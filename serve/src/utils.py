import supervisely as sly
import yaml
import pkg_resources
import os
import errno
import sly_globals as g
import requests
import numpy as np
import pathlib
import open3d as o3d
from mmcv import Config
from mmdet3d.apis import inference_detector, init_model
from supervisely.geometry.cuboid_3d import Cuboid3d
from supervisely.app.v1.widgets.progress_bar import ProgressBar


def get_pretrained_models(return_metrics=False):
    model_yamls = sly.json.load_json_file(os.path.join(g.root_source_path, "models", "model_meta.json"))
    model_config = {}
    all_metrics = []
    for model_group in model_yamls:
        group_name = model_group["group_name"]
        models = model_group["models"]
        for model_meta in models:
            with open(os.path.join(g.configs_dir, model_meta["yml_file"]), "r") as stream:
                model_info = yaml.safe_load(stream)
                model_config[model_meta["model_name"]] = {}
                model_config[model_meta["model_name"]]["checkpoints"] = []
                model_config[model_meta["model_name"]]["paper_from"] = model_meta["paper_from"]
                model_config[model_meta["model_name"]]["year"] = model_meta["year"]
                model_config[model_meta["model_name"]]["group_name"] = group_name
                # TODO: check how it works with master branch
                mmdet3d_ver = pkg_resources.get_distribution("mmdet3d").version
                model_config[model_meta["model_name"]][
                    "config_url"] = f"https://github.com/open-mmlab/mmdetection3d/tree/v{mmdet3d_ver}/configs/" + \
                                    model_meta["yml_file"].split("/")[0]
                checkpoint_keys = []
                for model in model_info["Models"]:
                    checkpoint_info = {}
                    if "exclude" in model_meta.keys():
                        if model_meta["exclude"].endswith("*"):
                            if model["Name"].startswith(model_meta["exclude"][:-1]):
                                continue
                    checkpoint_info["name"] = model["Name"]
                    checkpoint_info["method"] = model["In Collection"]
                    try:
                        checkpoint_info["training_memory"] = model["Metadata"]["Training Memory (GB)"]
                    except KeyError:
                        checkpoint_info["training_memory"] = "-"
                    checkpoint_info["config_file"] = str(pathlib.Path(g.configs_dir).parent / model["Config"])
                    checkpoint_info["dataset"] = model["Results"][0]["Dataset"]
                    for metric_name, metric_val in model["Results"][0]["Metrics"].items():
                        if metric_name not in all_metrics:
                            all_metrics.append(metric_name)
                        checkpoint_info[metric_name] = metric_val
                    try:
                        checkpoint_info["weights"] = model["Weights"]
                    except KeyError:
                        continue

                    for key in checkpoint_info.keys():
                        checkpoint_keys.append(key)
                    model_config[model_meta["model_name"]]["checkpoints"].append(checkpoint_info)
                model_config[model_meta["model_name"]]["all_keys"] = checkpoint_keys
    if return_metrics:
        return model_config, all_metrics
    return model_config


def get_table_columns(metrics):
    columns = [
        {"key": "name", "title": " ", "subtitle": None},
        {"key": "method", "title": "Method", "subtitle": None},
        {"key": "dataset", "title": "Dataset", "subtitle": None},
        {"key": "training_memory", "title": "Memory", "subtitle": "Training (GB)"},
    ]
    for metric in metrics:
        columns.append({"key": metric, "title": metric, "subtitle": "score"})
    return columns


def download_sly_file(remote_path, local_path, progress=None):
    file_info = g.api.file.get_info_by_path(g.team_id, remote_path)
    if file_info is None:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), remote_path)
    if progress is not None:
        progress.set_total(file_info.sizeb)
        progress_cb = progress.increment
    else:
        progress_cb = None
    g.api.file.download(g.team_id, remote_path, local_path, g.my_app.cache, progress_cb)
    if progress is not None:
        progress.reset_and_update()

    sly.logger.info(f"{remote_path} has been successfully downloaded",
                    extra={"weights": local_path})


def download_custom_config(state):
    weights_remote_dir = os.path.dirname(state["weightsPath"])
    model_config_local_path = os.path.join(g.my_app.data_dir, 'config.py')

    config_remote_dir = os.path.join(weights_remote_dir, 'config.py')
    if g.api.file.exists(g.team_id, config_remote_dir):
        download_sly_file(config_remote_dir, model_config_local_path)
    return model_config_local_path


def download_weights(state):
    progress = ProgressBar(g.task_id, g.api, "data.progressWeights", "Downloading weights", is_size=True,
                                           min_report_percent=5)
    if state["weightsInitialization"] == "custom":
        weights_path_remote = state["weightsPath"]
        if not weights_path_remote.endswith(".pth"):
            raise ValueError(f"Weights file has unsupported extension {sly.fs.get_file_ext(weights_path_remote)}. "
                                f"Supported: '.pth'")

        g.local_weights_path = os.path.join(g.my_app.data_dir, sly.fs.get_file_name_with_ext(weights_path_remote))
        if sly.fs.file_exists(g.local_weights_path):
            os.remove(g.local_weights_path)

        download_sly_file(weights_path_remote, g.local_weights_path, progress)
        g.model_config_local_path = download_custom_config(state)

    else:
        checkpoints_by_model = get_pretrained_models()[state["pretrainedModel"]]["checkpoints"]
        selected_model = next(item for item in checkpoints_by_model
                                if item["name"] == state["selectedModel"][state["pretrainedModel"]])

        weights_url = selected_model.get('weights')
        config_file = selected_model.get('config_file')
        if weights_url is not None:
            g.local_weights_path = os.path.join(g.my_app.data_dir, sly.fs.get_file_name_with_ext(weights_url))
            g.model_config_local_path = os.path.join(g.root_source_path, config_file)
            if sly.fs.file_exists(g.local_weights_path) is False:
                response = requests.head(weights_url, allow_redirects=True)
                sizeb = int(response.headers.get('content-length', 0))
                progress.set_total(sizeb)
                os.makedirs(os.path.dirname(g.local_weights_path), exist_ok=True)
                sly.fs.download(weights_url, g.local_weights_path, g.my_app.cache, progress.increment)
                progress.reset_and_update()
            sly.logger.info("Pretrained weights has been successfully downloaded",
                            extra={"weights": g.local_weights_path})


def expand_point_cloud_range(cfg):
    pcr = cfg.point_cloud_range
    for i in range(3):
        if abs(pcr[i] + pcr[i+3]) > 1e-5:
            cfg.point_cloud_range[i] = -max(abs(pcr[i]), abs(pcr[i+3]))
            cfg.point_cloud_range[i+3] = max(abs(pcr[i]), abs(pcr[i+3]))

    for idx, pipeline_step in enumerate(cfg.data.test.pipeline):
        if pipeline_step.type == "MultiScaleFlipAug3D":
            for tr_idx, transform in enumerate(pipeline_step.transforms):
                if transform.type == "PointsRangeFilter":
                    cfg.data.test.pipeline[idx].transforms[tr_idx].point_cloud_range=cfg.point_cloud_range
    
    if g.model_name == "CenterPoint":
        ss = cfg.model.pts_middle_encoder.sparse_shape
        cfg.voxel_size = [
            (pcr[3] - pcr[0]) / ss[1],
            (pcr[4] - pcr[1]) / ss[2],
            (pcr[5] - pcr[2]) / (ss[0] - 1),
        ]
        cfg.model.pts_voxel_layer.voxel_size = cfg.voxel_size
        cfg.model.pts_voxel_layer.point_cloud_range = cfg.point_cloud_range
        cfg.model.pts_bbox_head.bbox_coder.post_center_range = cfg.point_cloud_range
        cfg.model.pts_bbox_head.bbox_coder.pc_range = cfg.point_cloud_range
        cfg.model.pts_bbox_head.bbox_coder.voxel_size = cfg.voxel_size[:2]
        cfg.model.train_cfg.pts.point_cloud_range = cfg.point_cloud_range
        cfg.model.train_cfg.pts.voxel_size = cfg.voxel_size
        cfg.model.test_cfg.pts.post_center_limit_range = cfg.point_cloud_range
        cfg.model.test_cfg.pts.pc_range = cfg.point_cloud_range
        cfg.model.test_cfg.pts.voxel_size = cfg.voxel_size[:2]


def init_model_and_cfg(state):
    cfg = Config.fromfile(g.model_config_local_path)
    print(cfg.pretty_text) # TODO: for debug
    labels = cfg['class_names']
    g.model_name = state["pretrainedModel"]
    g.gt_index_to_labels = dict(enumerate(labels))
    g.gt_labels = {v: k for k, v in g.gt_index_to_labels.items()}
    obj_classes = sly.ObjClassCollection([sly.ObjClass(k, Cuboid3d) for k in labels])
    g.meta = sly.ProjectMeta(obj_classes=obj_classes)
    dims = 4
    
    if state["weightsInitialization"] == "pretrained" and state["expandPCR"]:
        expand_point_cloud_range(cfg)

    g.ptc_range_centered = is_pcr_centered(cfg.point_cloud_range, eps=1e-5)
    
    if hasattr(cfg, "center_coords"):
        g.train_data_centered = cfg.center_coords

    # TODO: doesn't work now
    if g.model_name == "Part-A2":
        cfg.model.type = "PartA2Fixed"
        cfg.model.voxel_layer.max_voxels=(800, 800)

    if g.model_name == "CenterPoint":
        cfg.model.type = "CenterPointFixed"
        cfg.model.pts_bbox_head.type = "CenterHeadWithVel"
        cfg.model.pts_middle_encoder.in_channels = dims
        cfg.model.pts_voxel_encoder.num_features = dims

    cfg.data.test.box_type_3d = 'lidar'

    cfg.data.test.pipeline[0].load_dim = dims
    cfg.data.test.pipeline[0].use_dim = dims

    for idx, pipeline_step in enumerate(cfg.data.test.pipeline):
        if pipeline_step.type == "LoadPointsFromMultiSweeps":
            del cfg.data.test.pipeline[idx]

    # TODO: add data pipeline fixes from train

    try:
        g.model = init_model(cfg, g.local_weights_path, state["device"]) 
    except FileNotFoundError:
        raise ValueError(f"File not exists: {g.local_weights_path}!")
    except Exception as e:
        sly.logger.exception(e)
        raise e

def rotate(source_angle, delta):
    result = source_angle + delta
    if result > np.pi:
        result = -np.pi + (result - np.pi)
    elif result < -np.pi:
        result = np.pi + (result + np.pi)
    return result

def get_per_box_predictions(result, score_thr, selected_classes, cfg, center_vec, input_slide_range):
    if 'pts_bbox' in result[0].keys():
        preds = result[0]['pts_bbox']
    else:
        preds = result[0]

    pred_scores = preds['scores_3d'].numpy()
    pred_bboxes = preds['boxes_3d'].tensor.numpy() # x, y, z, dx, dy, dz, rot, vel_x, vel_y
    pred_labels = preds['labels_3d'].numpy()
    
    inds = pred_scores > score_thr
    pred_bboxes = pred_bboxes[inds]
    pred_labels = pred_labels[inds]
    pred_scores = pred_scores[inds]
    
    assert len(pred_bboxes) == len(pred_scores) == len(pred_labels)
    results = []
    for i in range(len(pred_bboxes)):
        det = {}
        det["detection_name"] = g.gt_index_to_labels[pred_labels[i]]
        if selected_classes is not None and det["detection_name"] not in selected_classes:
            continue
        det["size"] = pred_bboxes[i,3:6].tolist()
        if cfg.dataset_type != "SuperviselyDataset":
            det["size"] = [det["size"][1], det["size"][0], det["size"][2]]
        det["translation"] = pred_bboxes[i,:3].tolist()
        # TODO: check these conditions in the future
        if cfg.dataset_type != "SuperviselyDataset":
            det["translation"][2] += det["size"][2] * 0.5
        if g.model_name == "CenterPoint":
            det["translation"][2] += det["size"][2] * 0.5
            
        for k in range(3):
            det["translation"][k] += center_vec[k]
        # skip boxes out of pointcloud range
        if det["translation"][0] < input_slide_range[0] or \
            det["translation"][0] > input_slide_range[3] or \
            det["translation"][1] < input_slide_range[1] or \
            det["translation"][1] > input_slide_range[4] or \
            det["translation"][2] < input_slide_range[2] or \
            det["translation"][2] > input_slide_range[5]:
            continue
        det["rotation"] = pred_bboxes[i,6].item()
        if cfg.dataset_type != "SuperviselyDataset":
            det["rotation"] = rotate(det["rotation"], -np.pi * 0.5)
        det["velocity"] = pred_bboxes[i,7:].tolist()
        det["detection_score"] = pred_scores[i].item()
        results.append(det)
    return results


def get_slide_boxes(pointcloud_range, model_pcr_dim, apply_sw):
    pcd = pointcloud_range.copy()
    ws = model_pcr_dim.copy()
    if isinstance(apply_sw, dict):
        apply_sw = list(apply_sw.values())
    slides = [0,0,0]
    for i in range(3):
        if apply_sw is not None and not apply_sw[i]:
            slides[i] = 1
            continue
        slides[i], overlap = divmod(pcd[i], ws[i])
        if overlap > 0:
            slides[i] += 1

    sboxes = []
    for z in range(int(slides[2])):
        for y in range(int(slides[1])):
            for x in range(int(slides[0])):
                sboxes.append([
                    ws[0] * x,
                    ws[0] * (x + 1),
                    ws[1] * y,
                    ws[1] * (y + 1),
                    ws[2] * z,
                    ws[2] * (z + 1),
                ])
    return sboxes


def is_pcr_centered(pcr, eps=0.01):
    return [
        abs(pcr[3] + pcr[0]) < eps,
        abs(pcr[4] + pcr[1]) < eps,
        abs(pcr[5] + pcr[2]) < eps
    ]


def inference_model(model, local_pointcloud_path, thresh=0.3, selected_classes=None,
                    apply_sw=None, center_ptc=None):
    if isinstance(center_ptc, dict):
        center_ptc = list(center_ptc.values())
    pcd = o3d.io.read_point_cloud(local_pointcloud_path)
    pcd_np = np.asarray(pcd.points)
    # check ptc ranges
    pcr = model.cfg.point_cloud_range
    pcr_dim = [pcr[3] - pcr[0], pcr[4] - pcr[1], pcr[5] - pcr[2]]
    input_ptc_dim = [
        pcd_np[:,0].max() - pcd_np[:,0].min(),
        pcd_np[:,1].max() - pcd_np[:,1].min(),
        pcd_np[:,2].max() - pcd_np[:,2].min()
    ]

    sboxes = get_slide_boxes(input_ptc_dim, pcr_dim, apply_sw)
    
    pcd_sboxes = []
    for sbox in sboxes:
        pcd_sbox = []
        for i in range(3):
            if center_ptc is None or center_ptc[i]:
                pcd_sbox.extend([
                    pcd_np[:,i].min() + sbox[i*2],
                    pcd_np[:,i].min() + sbox[i*2+1]
                ])
            else:
                pcd_sbox.extend([
                    pcr[i],
                    pcr[i+3]
                ])
        pcd_sboxes.append(pcd_sbox)

    results = []
    # TODO: is it possible to use batch inference here?
    for sbox in pcd_sboxes:
        pcd_eps = 1e-3
        pcd_slide = pcd_np[
            (pcd_np[:,0] > sbox[0] - pcd_eps) &
            (pcd_np[:,0] < sbox[1] + pcd_eps) &
            (pcd_np[:,1] > sbox[2] - pcd_eps) &
            (pcd_np[:,1] < sbox[3] + pcd_eps) &
            (pcd_np[:,2] > sbox[4] - pcd_eps) &
            (pcd_np[:,2] < sbox[5] + pcd_eps)
        ]
        if len(pcd_slide) == 0:
            continue
        center_vec = [0, 0, 0]
        input_slide_range = [
            pcd_slide[:,0].min(),
            pcd_slide[:,1].min(),
            pcd_slide[:,2].min(),
            pcd_slide[:,0].max(),
            pcd_slide[:,1].max(),
            pcd_slide[:,2].max()
        ]
        for i in range(3):
            if center_ptc is None or center_ptc[i]:
                dim_trans = input_slide_range[i] + (input_slide_range[i + 3] - input_slide_range[i]) * 0.5
                pcd_slide[:,i] -= dim_trans
                center_vec[i] = dim_trans

        intensity = np.zeros((pcd_slide.shape[0], 1), dtype=np.float32)
        pcd_slide = np.hstack((pcd_slide, intensity))
        pcd_slide.astype(np.float32).tofile(local_pointcloud_path)
        
        result, _ = inference_detector(model, local_pointcloud_path)
        result = get_per_box_predictions(result, thresh, selected_classes, model.cfg, center_vec, input_slide_range)
        results.extend(result)
    return results