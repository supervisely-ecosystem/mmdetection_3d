import errno
import os
import pathlib
import requests
import yaml
import pkg_resources
import sly_globals as g
import supervisely_lib as sly
from mmcv import Config
from supervisely.app.v1.widgets.progress_bar import ProgressBar

cfg = None

def init(data, state):
    state['pretrainedModel'] = 'CenterPoint'
    data["pretrainedModels"], metrics = get_pretrained_models(return_metrics=True)
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

    data["modelColumns"] = get_table_columns(metrics)

    state["selectedModel"] = {pretrained_model: data["pretrainedModels"][pretrained_model]["checkpoints"][0]['name']
                              for pretrained_model in data["pretrainedModels"].keys()}

    state["weightsInitialization"] = "pretrained"  # "custom"
    state["collapsedModels"] = True
    state["disabledModels"] = True
    state["weightsPath"] = ""
    data["doneModels"] = False
    state["loadingModel"] = False

    ProgressBar(g.task_id, g.api, "data.progressWeights", "Download weights", is_size=True,
                                min_report_percent=5).init_data(data)

def get_pretrained_models(return_metrics=False):
    model_yamls = sly.json.load_json_file(os.path.join(g.root_source_dir, "models", "model_meta.json"))
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


@g.my_app.callback("download_weights")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def download_weights(api: sly.Api, task_id, context, state, app_logger):
    progress = ProgressBar(g.task_id, g.api, "data.progressWeights", "Downloading weights", is_size=True,
                                           min_report_percent=5)
    
    try:
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
                g.model_config_local_path = os.path.join(g.root_source_dir, config_file)
                if sly.fs.file_exists(g.local_weights_path) is False:
                    response = requests.head(weights_url, allow_redirects=True)
                    sizeb = int(response.headers.get('content-length', 0))
                    progress.set_total(sizeb)
                    os.makedirs(os.path.dirname(g.local_weights_path), exist_ok=True)
                    sly.fs.download(weights_url, g.local_weights_path, g.my_app.cache, progress.increment)
                    progress.reset_and_update()
                sly.logger.info("Pretrained weights has been successfully downloaded",
                                extra={"weights": g.local_weights_path})
    except Exception as e:
        progress.reset_and_update()
        raise e

    fields = [
        {"field": "state.loadingModel", "payload": False},
        {"field": "data.doneModels", "payload": True},
        {"field": "state.collapsedClasses", "payload": False},
        {"field": "state.disabledClasses", "payload": False},
        {"field": "state.activeStep", "payload": 3},
    ]

    global cfg
    if g.model_config_local_path is None:
        raise ValueError("Model config file not found!")
    cfg = Config.fromfile(g.model_config_local_path)

    if state["weightsInitialization"] != "custom":
        cfg.pretrained_model = state["pretrainedModel"]

    # print(f'Initial config:\n{cfg.pretty_text}') # TODO: debug

    g.api.app.set_fields(g.task_id, fields)

def restart(data, state):
    data["doneModels"] = False