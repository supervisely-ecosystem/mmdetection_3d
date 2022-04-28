import functools
import os
import sys

sys.path.append('')
import supervisely as sly

import sly_globals as g
import nn_utils
import json
import yaml
import requests

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


@sly.timeit
def get_weights():
    # TODO: check and fix this! 
    # There is broken code from previous version here.
    if g.modelWeightsOptions == "pretrained":
        sly.logger.debug(g.pretrained_models_cfg)
        model_data = [x for x in g.pretrained_models_cfg if x["Model"] == g.pretrained_weights][0]
        g.local_config_path = model_data["config"]
        local_yml_path = os.path.join(os.path.dirname(model_data["config"]), "metafile.yml")
        with open(local_yml_path, "r") as stream:
            model_info = yaml.safe_load(stream)
        for model in model_info["Models"]:
            if model["Config"] == "/".join(g.local_config_path.split("/")[-3:]):
                g.remote_weights_path = model["Weights"]
        assert g.remote_weights_path is not None

    elif g.modelWeightsOptions == "custom":
        g.remote_weights_path = g.custom_weights
        g.remote_config_path = os.path.join(os.path.dirname(os.path.dirname(g.custom_weights)),
                                            "configs/model_config.py")
        g.local_config_path = os.path.join(g.my_app.data_dir, "model_config.py")

        progress = sly.Progress("Downloading config", 1, is_size=True, need_info_log=True)
        g.local_weights_path = os.path.join(g.my_app.data_dir, "weights.pt")

        file_info = g.my_app.public_api.file.get_info_by_path(g.team_id, g.remote_config_path)
        progress.set(current=0, total=file_info.sizeb)
        g.my_app.public_api.file.download(g.team_id, g.remote_config_path,
                                          g.local_config_path, g.my_app.cache,
                                          progress.iters_done_report)

    else:
        raise ValueError("Unknown weights option {!r}".format(g.modelWeightsOptions))

    # progress = sly.Progress("Downloading weights", 1, is_size=True, need_info_log=True)
    g.local_weights_path = os.path.join(g.my_app.data_dir, "weights.pt")

    # file_info = g.my_app.public_api.file.get_info_by_path(g.team_id, g.remote_weights_path)
    # response = requests.head(g.remote_weights_path, allow_redirects=True)
    # sizeb = int(response.headers.get('content-length', 0))
    # progress.set(current=0, total=sizeb)

    sly.fs.download(g.remote_weights_path, g.local_weights_path, g.my_app.cache)

    # sly.logger.info(f"Model {g.pretrained_weights} has been "
    #                 f"successfully downloaded with weights: {g.remote_weights_path} and config {g.remote_config_path}")

    sly.logger.debug(f"Local weights {g.local_weights_path}")
    sly.logger.debug(f"Local config path {g.local_config_path}")
    sly.logger.info("Model has been successfully downloaded")


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


def _inference(api, pointcloud_id, threshold=None):
    local_pointcloud_path = os.path.join(g.my_app.data_dir, sly.rand_str(15) + ".pcd")

    api.pointcloud.download_path(pointcloud_id, local_pointcloud_path)

    results = nn_utils.inference_model(g.model, local_pointcloud_path,
                                       thresh=threshold if threshold is not None else 0.3)
    sly.fs.silent_remove(local_pointcloud_path)
    return results


@g.my_app.callback("inference_pointcloud_id")
@sly.timeit
@send_error_data
def inference_pointcloud_id(api: sly.Api, task_id, context, state, app_logger):
    app_logger.debug("Input data", extra={"state": state})
    results = _inference(api, state["pointcloud_id"], state.get("threshold"))
    request_id = context["request_id"]
    g.my_app.send_response(request_id, data={"results": results.to_json()})


@g.my_app.callback("inference_pointcloud_ids")
@sly.timeit
@send_error_data
def inference_pointcloud_ids(api: sly.Api, task_id, context, state, app_logger):
    app_logger.debug("Input data", extra={"state": state})
    results = []
    for pointcloud_id in state["pointcloud_ids"]:
        try:
            result = _inference(api, pointcloud_id, state.get("threshold"))
        except Exception as e:
            sly.logger.exception(e)

        sly.logger.info(f"Predict {pointcloud_id}")
        results.append(result.to_json())

    request_id = context["request_id"]
    g.my_app.send_response(request_id, data={"results": results})


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

    get_weights()

    nn_utils.construct_model_meta()
    nn_utils.deploy_model()
    g.my_app.run()


if __name__ == "__main__":
    sly.main_wrapper("main", main)
