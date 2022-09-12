import supervisely as sly
import sly_globals as g
import os
import os.path as osp
import random
import numpy as np
import pickle as pkl
from collections import namedtuple
import open3d as o3d

ItemInfo = namedtuple('ItemInfo', ['dataset_name', 'name', 'img_path'])
train_set = None
val_set = None


def init(project_info, data, state):
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
    state["point_cloud_range"] = None
    state["point_cloud_dim"] = None

    init_progress("PointsRangeCalculation", state)

def restart(data, state):
    data["doneData"] = False


def init_progress(index, state):
    state[f"progress{index}"] = False
    state[f"progressCurrent{index}"] = 0
    state[f"progressTotal{index}"] = None
    state[f"progressPercent{index}"] = 0


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


def calculate_pcr(items, log_step=10):
    if len(items) < log_step:
        log_step = len(items)
    
    point_cloud_range = [10000, 10000, 10000, -10000, -10000, -10000]
    point_cloud_dim = [0, 0, 0]
    for idx, item in enumerate(items):
        if idx % log_step == 0:
            fields = [
                {"field": f"state.progressCurrentPointsRangeCalculation", "payload": idx},
                {"field": f"state.progressPercentPointsRangeCalculation", "payload": int(idx / len(items) * 100)}
            ]
            g.api.app.set_fields(g.task_id, fields)
        
        filename = osp.join(item.dataset_name, "pointcloud", item.name)
        pcd = o3d.io.read_point_cloud(osp.join(g.project_dir, filename))
        pcd_np = np.asarray(pcd.points)
        ptc_range = [
            pcd_np[:,0].min(), 
            pcd_np[:,1].min(), 
            pcd_np[:,2].min(),
            pcd_np[:,0].max(),
            pcd_np[:,1].max(),
            pcd_np[:,2].max()
        ]
        for i in range(3):
            if ptc_range[i] < point_cloud_range[i]:
                point_cloud_range[i] = ptc_range[i]
            if ptc_range[i + 3] > point_cloud_range[i + 3]:
                point_cloud_range[i + 3] = ptc_range[i + 3]
            if ptc_range[i + 3] - ptc_range[i] > point_cloud_dim[i]:
                point_cloud_dim[i] = ptc_range[i + 3] - ptc_range[i]

    return point_cloud_range, point_cloud_dim


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
        pcr, pcd = calculate_pcr(train_set + val_set)
        # TODO: change values from next step
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
            {"field": "state.valImagesCount", "payload": None if val_set is None else len(val_set)}
        ]
        if step_done is True:
            fields.extend([
                {"field": "state.collapsedData", "payload": False},
                {"field": "state.disabledData", "payload": False},
                {"field": "state.activeStep", "payload": 5},
                {"field": "state.point_cloud_range", "payload": pcr},
                {"field": "state.point_cloud_dim", "payload": pcd}
            ])
        g.api.app.set_fields(g.task_id, fields)


