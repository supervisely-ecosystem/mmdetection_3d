import os
from mmdet3d.apis import inference_detector, init_model
import numpy as np
import supervisely_lib as sly
from supervisely.geometry.cuboid_3d import Cuboid3d, Vector3d
from supervisely.pointcloud_annotation.pointcloud_object_collection import PointcloudObjectCollection
import sly_globals as g
import mmcv
import open3d as o3d
# from mmdet3d.models.detectors.centerpoint import CenterPoint
# from mmdet3d.models.voxel_encoders.voxel_encoder import HardVFE

class Annotation:
    @staticmethod
    def pred_to_sly_geometry(labels, reverse=False):
        geometry = []
        for l in labels:
            x, y, z, dx, dy, dz, heading = l
            position = Vector3d(float(x), float(y), float(z * 0.5))

            if reverse:
                yaw = float(heading) - np.pi
                yaw = yaw - np.floor(yaw / (2 * np.pi) + 0.5) * 2 * np.pi
            else:
                yaw = -heading

            rotation = Vector3d(0, 0, float(yaw))
            dimension = Vector3d(float(dx), float(dy), float(dz))
            g = Cuboid3d(position, rotation, dimension)
            geometry.append(g)
        return geometry


    @staticmethod
    def create_annotation(bboxes, labels, meta):
        geometry_list = Annotation.pred_to_sly_geometry(bboxes)
        figures = []
        objs = []

        for label, geometry in zip(labels, geometry_list):  # by object in point cloud
            pcobj = sly.PointcloudObject(meta.get_obj_class(label))
            figures.append(sly.PointcloudFigure(pcobj, geometry))
            objs.append(pcobj)

        pc_annotation = sly.PointcloudAnnotation(PointcloudObjectCollection(objs), figures)
        return pc_annotation


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
        # cfg.model.pts_voxel_encoder.in_channels = 3
        g.model = init_model(cfg, g.local_weights_path)
        sly.logger.info("Model has been successfully deployed")
    except FileNotFoundError:
        raise ValueError(f"File not exists: {g.local_weights_path}!")
    except Exception as e:
        sly.logger.exception(e)
        raise e


def decode_prediction(result, labels, score_thr):
    if 'pts_bbox' in result[0].keys():
        pred_bboxes = result[0]['pts_bbox']['boxes_3d'].tensor.numpy()
        pred_scores = result[0]['pts_bbox']['scores_3d'].numpy()
        pred_labels = result[0]['pts_bbox']['labels_3d'].numpy()
    else:
        pred_bboxes = result[0]['boxes_3d'].tensor.numpy()
        pred_scores = result[0]['scores_3d'].numpy()
        pred_labels = result[0]['labels_3d'].numpy()

    if score_thr > 0:
        inds = pred_scores > score_thr
        pred_bboxes = pred_bboxes[inds]
        pred_labels = pred_labels[inds]
        pred_scores = pred_scores[inds]

    pred_bboxes = pred_bboxes[:, :7]  # x, y, z, x_size, y_size, z_size, yaw
    labels = [labels[x] for x in pred_labels]  # convert int labels to str
    return pred_bboxes, pred_scores, labels


def inference_model(model, local_pointcloud_path, thresh=0.3):
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
    model.cfg.point_cloud_range = [pcd_np[:,0].min(), pcd_np[:,1].min(), pcd_np[:,2].min(), pcd_np[:,0].max(), pcd_np[:,1].max(), pcd_np[:,2].max()]

    model.cfg.data.test.pipeline[0].load_dim = point_dims
    model.cfg.data.test.pipeline[0].use_dim = point_dims
    for idx, pipeline_step in enumerate(model.cfg.data.test.pipeline):
        if pipeline_step.type == "LoadPointsFromMultiSweeps":
            del model.cfg.data.test.pipeline[idx]
    result, data = inference_detector(model, local_pointcloud_path)
    pred_bboxes, pred_scores, labels = decode_prediction(result, g.gt_index_to_labels, thresh)
    annotation = Annotation.create_annotation(pred_bboxes, labels, g.meta)
    return annotation
