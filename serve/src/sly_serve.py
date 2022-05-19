import functools
import os

import utils
import supervisely as sly
from supervisely.geometry.cuboid_3d import Cuboid3d, Vector3d
from supervisely.pointcloud_annotation.pointcloud_object_collection import PointcloudObjectCollection
from supervisely.video_annotation.frame_collection import FrameCollection
from supervisely.video_annotation.video_tag_collection import VideoTagCollection
from supervisely.app.v1.widgets.progress_bar import ProgressBar

import sly_globals as g
import numpy as np
from collections import OrderedDict


def send_error_data(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        value = None
        try:
            value = func(*args, **kwargs)
        except Exception as e:
            request_id = kwargs["context"]["request_id"]
            g.my_app.send_response(request_id, data={"error": repr(e)})
        return value

    return wrapper


def init_state_and_data(data, state):
    state['pretrainedModel'] = 'CenterPoint'
    data["pretrainedModels"], metrics = utils.get_pretrained_models(return_metrics=True)
    model_select_info = {}
    for model_name, params in data["pretrainedModels"].items():
        if params["group_name"] not in model_select_info.keys():
            model_select_info[params["group_name"]] = []
        model_select_info[params["group_name"]].append({
            "name": model_name,
            "paper_from": params["paper_from"],
            "year": params["year"]
        })

    data["pretrainedModelsInfo"] = []
    for group_name, models in model_select_info.items():
        group_dict = {"group_name": group_name, "models": models}
        data["pretrainedModelsInfo"].append(group_dict)

    data["configLinks"] = {model_name: params["config_url"] for model_name, params in data["pretrainedModels"].items()}

    data["modelColumns"] = utils.get_table_columns(metrics)
    state["weightsInitialization"] = "pretrained"
    state["selectedModel"] = {pretrained_model: data["pretrainedModels"][pretrained_model]["checkpoints"][0]['name']
                              for pretrained_model in data["pretrainedModels"].keys()}
    state["device"] = "cuda:0"
    state["weightsPath"] = ""
    state["loading"] = False
    state["deployed"] = False
    ProgressBar(g.task_id, g.api, "data.progressWeights", "Downloading weights...", is_size=True,
                                min_report_percent=5).init_data(data)


@g.my_app.callback("get_custom_inference_settings")
@sly.timeit
@send_error_data
def get_custom_inference_settings(api: sly.Api, task_id, context, state, app_logger):
    # TODO: it should be YML with comments
    info = {
        "threshold": "(Float[0.0, 1.0], default 0.3) Boxes with confidence less than the threshold will"
                     " be skipped in the response."
    }

    request_id = context["request_id"]
    g.my_app.send_response(request_id, data=info)


@g.my_app.callback("get_output_classes_and_tags")
@sly.timeit
@send_error_data
def get_output_classes_and_tags(api: sly.Api, task_id, context, state, app_logger):
    request_id = context["request_id"]
    g.my_app.send_response(request_id, data=g.meta.to_json())


@g.my_app.callback("get_session_info")
@sly.timeit
@send_error_data
def get_session_info(api: sly.Api, task_id, context, state, app_logger):
    info = {
        "app": "Serve MMDetection3D",
        "weights": g.remote_weights_path,
        "device": g.device,
        "session_id": task_id,
        "classes_count": len(g.meta.obj_classes),
    }
    request_id = context["request_id"]
    g.my_app.send_response(request_id, data=info)


def _inference(api, pointcloud_id, threshold=None, selected_classes=None):
    local_pointcloud_path = os.path.join(g.my_app.data_dir, sly.rand_str(15) + ".pcd")

    api.pointcloud.download_path(pointcloud_id, local_pointcloud_path)

    result = utils.inference_model(g.model, local_pointcloud_path,
                                       thresh=threshold if threshold is not None else 0.3,
                                       selected_classes=selected_classes)
    sly.fs.silent_remove(local_pointcloud_path)
    return result


class Annotation:
    @staticmethod
    def pred_to_sly_geometry(labels, reverse=False):
        geometry = []
        for l in labels:
            x, y, z = l["translation"][0], l["translation"][1], l["translation"][2]
            dx, dy, dz = l["size"][0], l["size"][1], l["size"][2]
            yaw = l["rotation"]
            position = Vector3d(float(x), float(y), float(z * 0.5))
            rotation = Vector3d(0, 0, float(yaw))
            dimension = Vector3d(float(dx), float(dy), float(dz))
            g = Cuboid3d(position, rotation, dimension)
            geometry.append(g)
        return geometry


    @staticmethod
    def create_annotation(detections, meta, type):
        objects = []
        annotations = {"objects": [], "figures": {}}
        for ptc_id, preds in detections.items():
            geometry_list = Annotation.pred_to_sly_geometry(preds)
            figures = []
            for pred, geometry in zip(preds, geometry_list):  # by object in point cloud
                pcobj = sly.PointcloudObject(meta.get_obj_class(pred["detection_name"]))
                objects.append(pcobj)
                figures.append(sly.PointcloudFigure(pcobj, geometry))
                # TODO: add tag confidence

            annotations["figures"][ptc_id] = figures
        
        annotations["objects"] = PointcloudObjectCollection(objects)

        if type == "point_cloud_episodes":
            frames = []
            for frame_ind, (ptc_id, figures) in enumerate(annotations["figures"].items()):
                frames.append(sly.Frame(frame_ind, figures))
            anns = sly.PointcloudEpisodeAnnotation(
                frames_count=len(detections), 
                objects=annotations["objects"], 
                frames=FrameCollection(frames), 
                tags=VideoTagCollection([]))
            anns = anns.to_json()
        elif type == "point_clouds":
            anns = OrderedDict()
            for ptc_id, figures in annotations["figures"].items():
                ann = sly.PointcloudAnnotation(
                    annotations["objects"], 
                    figures, 
                    tags=VideoTagCollection([]))
                anns[ptc_id] = ann.to_json()
        return anns


@g.my_app.callback("inference_pointcloud_id")
@sly.timeit
@send_error_data
def inference_pointcloud_id(api: sly.Api, task_id, context, state, app_logger):
    app_logger.debug("Input data", extra={"state": state})
    try:
        raw_result = _inference(api, state["pointcloud_id"], state.get("threshold"), state.get("classes", None))
    except Exception as e:
        sly.logger.exception(e)

    ann = Annotation.create_annotation(
        {state["pointcloud_id"]: raw_result}, 
        g.meta, 
        type="point_clouds")

    results = {
        "annotation": ann[state["pointcloud_id"]], 
        "raw_results": raw_result
    }
    request_id = context["request_id"]
    g.my_app.send_response(request_id, data={"results": results})


@g.my_app.callback("inference_pointcloud_ids")
@sly.timeit
@send_error_data
def inference_pointcloud_ids(api: sly.Api, task_id, context, state, app_logger):
    assert state["project_type"] in ["point_cloud_episodes", "point_clouds"]
    app_logger.debug("Input data", extra={"state": state})
    raw_results = OrderedDict()
    for pointcloud_id in state["pointcloud_ids"]:
        try:
            tracking_result = _inference(api, pointcloud_id, state.get("threshold"), state.get("classes", None))
        except Exception as e:
            sly.logger.exception(e)

        sly.logger.info(f"Predict {pointcloud_id}")
        raw_results[pointcloud_id] = tracking_result

    anns = Annotation.create_annotation(raw_results, g.meta, state["project_type"])
    results = {"annotation": anns, "raw_results": raw_results}
    request_id = context["request_id"]
    g.my_app.send_response(request_id, data={"results": results})


@g.my_app.callback("run")
@g.my_app.ignore_errors_and_show_dialog_window()
def init_model(api: sly.Api, task_id, context, state, app_logger):
    g.remote_weights_path = state["weightsPath"]
    g.device = state["device"]
    utils.download_weights(state)
    utils.init_model_and_cfg()
    fields = [
        {"field": "state.loadingModel", "payload": False},
        {"field": "state.deployed", "payload": True},
    ]
    g.api.app.set_fields(g.task_id, fields)
    sly.logger.info("Model has been successfully deployed.")


def main():
    # TODO: mmdet3d version from master now. It is unstable.
    sly.logger.info("Script arguments", extra={
        "context.teamId": g.team_id,
        "context.workspaceId": g.workspace_id,
        "device": g.device,
        "modelWeightsOptions": g.modelWeightsOptions,
        "custom_weights": g.custom_weights,
        "pretrained_weights": g.pretrained_weights
    })

    data = {}
    state = {}

    init_state_and_data(data, state)

    g.my_app.compile_template(g.root_source_path)
    g.my_app.run(data=data, state=state)


if __name__ == "__main__":
    sly.main_wrapper("main", main)
