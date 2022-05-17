import os
from mmdet3d.apis import inference_detector, init_model
import numpy as np
import supervisely_lib as sly
import sly_globals as g
import mmcv
import open3d as o3d
from supervisely.geometry.cuboid_3d import Cuboid3d


def construct_model_meta():
    labels = mmcv.Config.fromfile(g.local_config_path)['class_names']

    g.gt_index_to_labels = dict(enumerate(labels))
    g.gt_labels = {v: k for k, v in g.gt_index_to_labels.items()}

    g.meta = sly.ProjectMeta(obj_classes=sly.ObjClassCollection([sly.ObjClass(k, Cuboid3d) for k in labels]))
    sly.logger.info(g.meta.to_json())


@sly.timeit
def deploy_model():
    try:
        cfg = mmcv.Config.fromfile(g.local_config_path)
        # print(cfg.pretty_text) # TODO: for debug
        if hasattr(cfg.model, "pts_voxel_encoder") and hasattr(cfg.model.pts_voxel_encoder, "in_channels"):
            cfg.model.pts_voxel_encoder.in_channels = 3
        g.model = init_model(cfg, g.local_weights_path)
        sly.logger.info("Model has been successfully deployed")
    except FileNotFoundError:
        raise ValueError(f"File not exists: {g.local_weights_path}!")
    except Exception as e:
        sly.logger.exception(e)
        raise e


def get_per_box_predictions(result, score_thr, selected_classes):
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
        det["translation"] = pred_bboxes[i,:3].tolist()
        det["size"] = pred_bboxes[i,3:6].tolist()
        det["rotation"] = pred_bboxes[i,6].item()
        det["velocity"] = pred_bboxes[i,7:].tolist()
        det["detection_score"] = pred_scores[i].item()
        results.append(det)
    return results


def inference_model(model, local_pointcloud_path, thresh=0.3, selected_classes=None):
    """Inference 1 pointcloud with the detector.

    Args:
        model (nn.Module): The loaded detector (ObjectDetection pipeline instance).
        local_pointcloud_path: str: The pointcloud filename.
    Returns:
        result Pointcloud.annotation object`.
    """
    # TODO: use plyfile.PlyData instead of o3d if it is possible
    pcd = o3d.io.read_point_cloud(local_pointcloud_path)
    pcd_np = np.asarray(pcd.points)
    pcd_np.astype(np.float32).tofile(local_pointcloud_path)
    point_dims = 3

    model.cfg.data.test.box_type_3d = 'lidar'
    model.cfg.point_cloud_range = [
        pcd_np[:,0].min(), 
        pcd_np[:,1].min(), 
        pcd_np[:,2].min(), 
        pcd_np[:,0].max(), 
        pcd_np[:,1].max(), 
        pcd_np[:,2].max()
    ]

    model.cfg.data.test.pipeline[0].load_dim = point_dims
    model.cfg.data.test.pipeline[0].use_dim = point_dims
    for idx, pipeline_step in enumerate(model.cfg.data.test.pipeline):
        if pipeline_step.type == "LoadPointsFromMultiSweeps":
            del model.cfg.data.test.pipeline[idx]
    result, data = inference_detector(model, local_pointcloud_path)
    result = get_per_box_predictions(result, thresh, selected_classes)

    return result
