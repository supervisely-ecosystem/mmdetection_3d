import supervisely as sly
import sly_globals as g
import os
import os.path as osp
import random
import numpy as np
import pickle as pkl
from itertools import groupby
from collections import namedtuple
import open3d as o3d

from supervisely.video_annotation.key_id_map import KeyIdMap
from supervisely.video_annotation.video_figure import VideoFigure

ItemInfo = namedtuple('ItemInfo', ['dataset_name', 'name', 'img_path'])
train_set = None
val_set = None

train_set_path = os.path.join(g.my_app.data_dir, "train.pkl")
val_set_path = os.path.join(g.my_app.data_dir, "val.pkl")

def init(project_info, project_meta: sly.ProjectMeta, data, state):
    data["randomSplit"] = [
        {"name": "train", "type": "success"},
        {"name": "val", "type": "primary"},
        {"name": "total", "type": "gray"},
    ]
    data["totalImagesCount"] = project_info.items_count

    train_percent = 80
    train_count = int(project_info.items_count / 100 * train_percent)
    if train_count < 1:
        train_count = 1
    elif project_info.items_count - train_count < 1:
        train_count = project_info.items_count - 1
    state["randomSplit"] = {
        "count": {
            "total": project_info.items_count,
            "train": train_count,
            "val": project_info.items_count - train_count
        },
        "percent": {
            "total": 100,
            "train": train_percent,
            "val": 100 - train_percent
        },
        "shareImagesBetweenSplits": False,
        "sliderDisabled": False,
    }

    state["splitMethod"] = "random"

    state["trainDatasets"] = []
    state["valDatasets"] = []
    state["splitInProgress"] = False
    state["trainImagesCount"] = None
    state["valImagesCount"] = None
    data["doneSplits"] = False
    state["collapsedSplits"] = True
    state["disabledSplits"] = True

    init_progress("ConvertTrain", state)
    init_progress("ConvertVal", state)


def restart(data, state):
    data["doneSplits"] = False


def init_progress(index, state):
    state[f"progress{index}"] = False
    state[f"progressCurrent{index}"] = 0
    state[f"progressTotal{index}"] = None
    state[f"progressPercent{index}"] = 0


def refresh_table():
    global items_to_ignore
    ignored_items_count = sum([len(ds_items) for ds_items in items_to_ignore.values()])
    total_items_count = g.project_fs.total_items - ignored_items_count
    train_percent = 80
    train_count = int(total_items_count / 100 * train_percent)
    if train_count < 1:
        train_count = 1
    elif g.project_info.items_count - train_count < 1:
        train_count = g.project_info.items_count - 1
    random_split_tab = {
        "count": {
            "total": total_items_count,
            "train": train_count,
            "val": total_items_count - train_count
        },
        "percent": {
            "total": 100,
            "train": train_percent,
            "val": 100 - train_percent
        },
        "shareImagesBetweenSplits": False,
        "sliderDisabled": False,
    }

    fields = [
        {'field': 'state.randomSplit', 'payload': random_split_tab},
        {'field': 'data.totalImagesCount', 'payload': total_items_count},
    ]
    g.api.app.set_fields(g.task_id, fields)



def get_train_val_splits_by_count(train_count, val_count):
    if g.project_fs.total_items != train_count + val_count:
        raise ValueError("total_count != train_count + val_count")
    all_items = []
    for dataset in g.project_fs.datasets:
        for item_name in dataset:
            all_items.append(ItemInfo(dataset_name=dataset.name,
                                name=item_name,
                                img_path=dataset.get_img_path(item_name)))
    random.shuffle(all_items)
    train_items = all_items[:train_count]
    val_items = all_items[train_count:]
    return train_items, val_items


def get_train_val_splits_by_dataset(train_datasets, val_datasets):
    def _add_items_to_list(datasets_names, items_list):
        for dataset_name in datasets_names:
            dataset = g.project_fs.datasets.get(dataset_name)
            if dataset is None:
                raise KeyError(f"Dataset '{dataset_name}' not found")
            for item_name in dataset:
                img_path, _ = dataset.get_item_paths(item_name)
                info = ItemInfo(dataset.name, item_name, img_path)
                items_list.append(info)

    train_items = []
    _add_items_to_list(train_datasets, train_items)
    val_items = []
    _add_items_to_list(val_datasets, val_items)
    return train_items, val_items


def get_train_val_sets(state):
    split_method = state["splitMethod"]
    if split_method == "random":
        train_count = state["randomSplit"]["count"]["train"]
        val_count = state["randomSplit"]["count"]["val"]
        train_set, val_set = get_train_val_splits_by_count(train_count, val_count)
    elif split_method == "datasets":
        train_datasets = state["trainDatasets"]
        val_datasets = state["valDatasets"]
        train_set, val_set = get_train_val_splits_by_dataset(train_datasets, val_datasets)
    else:
        raise ValueError(f"Unknown split method: {split_method}")
    return train_set, val_set


def verify_train_val_sets(train_set, val_set):
    if len(train_set) == 0:
        raise ValueError("Train set is empty, check or change split configuration")
    elif len(train_set) < 1:
        raise ValueError("Train set is not big enough, min size is 1.")
    if len(val_set) == 0:
        raise ValueError("Val set is empty, check or change split configuration")
    elif len(val_set) < 1:
        raise ValueError("Val set is not big enough, min size is 1.")


@g.my_app.callback("create_splits")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def create_splits(api: sly.Api, task_id, context, state, app_logger):
    step_done = False
    global train_set, val_set
    try:
        api.task.set_field(task_id, "state.splitInProgress", True)
        train_set, val_set = get_train_val_sets(state)
        verify_train_val_sets(train_set, val_set)
        
        if train_set is not None:
            sly.logger.info("Converting train annotations to mmdet3d format...")
            # if not osp.exists(train_set_path): # TODO: for debug
            save_set_to_annotation(train_set_path, train_set, state["selectedClasses"], "Train")
        if val_set is not None:
            sly.logger.info("Converting val annotations to mmdet3d format...")
            # if not osp.exists(val_set_path): # TODO: for debug
            save_set_to_annotation(val_set_path, val_set, state["selectedClasses"], "Val")
        step_done = True
    except Exception as e:
        train_set = None
        val_set = None
        step_done = False
        raise e
    finally:
        fields = [
            {"field": "state.splitInProgress", "payload": False},
            {"field": "data.doneSplits", "payload": step_done},
            {"field": "state.trainImagesCount", "payload": None if train_set is None else len(train_set)},
            {"field": "state.valImagesCount", "payload": None if val_set is None else len(val_set)},
        ]
        if step_done is True:
            fields.extend([
                {"field": "state.collapsedMonitoring", "payload": False},
                {"field": "state.disabledMonitoring", "payload": False},
                {"field": "state.activeStep", "payload": 7},
            ])
        g.api.app.set_fields(g.task_id, fields)


def save_set_to_annotation(save_path, items, selected_classes, split_name):
    fields = [
        {"field": f"state.progressConvert{split_name}", "payload": True},
        {"field": f"state.progressTotalConvert{split_name}", "payload": len(items)},
    ]
    g.api.app.set_fields(g.task_id, fields)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    annotations = []
    log_step = 10
    frames_to_pcds = {}
    pcds_to_frames = {}
    if len(items) < log_step:
        log_step = len(items)
    for idx, item in enumerate(items):
        if idx % log_step == 0:
            fields = [
                {"field": f"state.progressCurrentConvert{split_name}", "payload": idx},
                {"field": f"state.progressPercentConvert{split_name}", "payload": int(idx / len(items) * 100)}
            ]
            g.api.app.set_fields(g.task_id, fields)
        ptc_info = {
            'sample_idx': idx,
            'lidar_points': {},
            'annos': {
                'gt_bboxes_3d': [],
                'gt_names': [],
                'gt_labels_3d': [],
                'box_type_3d': 'LiDAR'
            }
        }
        filename = osp.join(item.dataset_name, "pointcloud", item.name)
        bin_filename = osp.join(item.dataset_name, "bin", f"{item.name}.bin")
        os.makedirs(osp.join(g.project_dir, osp.dirname(bin_filename)), exist_ok=True)
        pcd = o3d.io.read_point_cloud(osp.join(g.project_dir, filename))
        pcd_np = np.asarray(pcd.points)
        # point_dims = 3
        # if g.model_name in ["3DSSD", "PointRCNN"]:
        #     intensity = np.ones((pcd_np.shape[0], 1)).astype(np.float32) * 0.5
        #     intensity += np.random.normal(0, 0.1, size=intensity.shape)
        #     pcd_np = np.hstack((pcd_np, intensity))
        #     point_dims = 4
        pcd_np.astype(np.float32).tofile(osp.join(g.project_dir, bin_filename))
        ptc_info['lidar_points']['lidar_path'] = bin_filename

        if item.dataset_name not in frames_to_pcds.keys():
            frames_to_pcds[item.dataset_name] = sly.json.load_json_file(osp.join(g.project_dir, item.dataset_name, "frame_pointcloud_map.json"))
            pcds_to_frames[item.dataset_name] = {v: k for k, v in frames_to_pcds[item.dataset_name].items()}

        pcd_to_frame = pcds_to_frames[item.dataset_name]
        frame_number = pcd_to_frame[item.name]
        ann_path = osp.join(g.project_dir, item.dataset_name, "annotation.json")
        ann_json = sly.json.load_json_file(ann_path)
        # key_id_map = KeyIdMap().load_json(osp.join(g.project_dir, "key_id_map.json"))
        ann = sly.PointcloudEpisodeAnnotation.from_json(ann_json, g.project_meta)
        for frame in ann.frames:
            if frame.index != int(frame_number):
                continue
            for fig in frame.figures:
                if fig.video_object.obj_class.name not in selected_classes:
                    continue
                ptc_info['annos']['gt_names'].append(fig.video_object.obj_class.name)
                box_info = [] # x, y, z, dx, dy, dz, rot, [vel_x, vel_y]
                pos = fig.geometry.position
                box_info.extend([pos.x, pos.y, pos.z])
                dim = fig.geometry.dimensions
                box_info.extend([dim.x, dim.y, dim.z])
                box_info.extend([fig.geometry.rotation.z])
                # box_info.extend([0, 0]) # TODO: add vel
                ptc_info['annos']['gt_bboxes_3d'].append(box_info)
                ptc_info['annos']['gt_labels_3d'].append(selected_classes.index(fig.video_object.obj_class.name))
        ptc_info['annos']['gt_bboxes_3d'] = np.array(ptc_info['annos']['gt_bboxes_3d'], dtype=np.float32)
        ptc_info['annos']['gt_labels_3d'] = np.array(ptc_info['annos']['gt_labels_3d'], dtype=np.int32)
        annotations.append(ptc_info)

    g.api.app.set_field(g.task_id, f"state.progressConvert{split_name}", False)

    with open(save_path, 'wb') as f:
        pkl.dump(annotations, f)
