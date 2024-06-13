import supervisely as sly
import sly_globals as g
import os, os.path as osp
from functools import partial

from mmdet3d.apis import train_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
# from mmcv.runner import load_checkpoint
from init_cfg import init_cfg
from sly_train_progress import init_progress, _update_progress_ui

import centerpoint
import sly_dataset
import sly_logger_hook

_open_lnk_name = "open_app.lnk"

def init(data, state):
    init_progress("Epoch", data)
    init_progress("Iter", data)
    init_progress("UploadDir", data)

    init_charts(data, state)

    state["collapsedMonitoring"] = True
    state["disabledMonitoring"] = True
    state["doneMonitoring"] = False
    data["eta"] = None
    state["isValidation"] = False
    data["outputName"] = None
    data["outputUrl"] = None
    state["started"] = False


def init_chart(title, names, xs, ys, smoothing=None, yrange=None, decimals=None, xdecimals=None):
    series = []
    for name, x, y in zip(names, xs, ys):
        series.append({
            "name": name,
            "data": [[px, py] for px, py in zip(x, y)]
        })
    result = {
        "options": {
            "title": title
        },
        "series": series
    }
    if smoothing is not None:
        result["options"]["smoothingWeight"] = smoothing
    if yrange is not None:
        result["options"]["yaxisInterval"] = yrange
    if decimals is not None:
        result["options"]["decimalsInFloat"] = decimals
    if xdecimals is not None:
        result["options"]["xaxisDecimalsInFloat"] = xdecimals
    return result


def init_charts(data, state):
    state["smoothing"] = 0.6
    # train charts
    state["chartLR"] = init_chart("LR", names=["lr"], xs = [[]], ys = [[]], smoothing=None, decimals=6, xdecimals=2)
    state["chartLoss"] = init_chart("Loss", names=["total"], xs=[[]], ys=[[]], smoothing=state["smoothing"], decimals=6, xdecimals=2)
    
    # val charts
    state["chartMAP_25"] = init_chart("mAP score (0.25)", names=["total"], xs=[[]], ys=[[]], smoothing=state["smoothing"], decimals=6, xdecimals=2, yrange=[0, 1])
    state["chartMAR_25"] = init_chart("mAR score (0.25)", names=["total"], xs=[[]], ys=[[]], smoothing=state["smoothing"], decimals=6, xdecimals=2, yrange=[0, 1])
    state["chartAP_25"] = init_chart("AP score (0.25)", names=[], xs=[], ys=[], smoothing=state["smoothing"], decimals=6, xdecimals=2, yrange=[0, 1])
    state["chartAR_25"] = init_chart("AR score (0.25)", names=[], xs=[], ys=[], smoothing=state["smoothing"], decimals=6, xdecimals=2, yrange=[0, 1])
    state["chartMAP_5"] = init_chart("mAP score (0.50)", names=["total"], xs=[[]], ys=[[]], smoothing=state["smoothing"], decimals=6, xdecimals=2, yrange=[0, 1])
    state["chartMAR_5"] = init_chart("mAR score (0.50)", names=["total"], xs=[[]], ys=[[]], smoothing=state["smoothing"], decimals=6, xdecimals=2, yrange=[0, 1])
    state["chartAP_5"] = init_chart("AP score (0.50)", names=[], xs=[], ys=[], smoothing=state["smoothing"], decimals=6, xdecimals=2, yrange=[0, 1])
    state["chartAR_5"] = init_chart("AR score (0.50)", names=[], xs=[], ys=[], smoothing=state["smoothing"], decimals=6, xdecimals=2, yrange=[0, 1])
    
    # system charts
    state["chartTime"] = init_chart("Time", names=["time"], xs=[[]], ys=[[]], xdecimals=2)
    state["chartDataTime"] = init_chart("Data Time", names=["data_time"], xs=[[]], ys=[[]], xdecimals=2)
    state["chartMemory"] = init_chart("Memory", names=["memory"], xs=[[]], ys=[[]], xdecimals=2)


@g.my_app.callback("change_smoothing")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def change_smoothing(api: sly.Api, task_id, context, state, app_logger):
    fields = [
        {"field": "state.chartLoss.options.smoothingWeight", "payload": state["smoothing"]},
        {"field": "state.chartMAP_25.options.smoothingWeight", "payload": state["smoothing"]},
        {"field": "state.chartMAR_25.options.smoothingWeight", "payload": state["smoothing"]},
        {"field": "state.chartAP_25.options.smoothingWeight", "payload": state["smoothing"]},
        {"field": "state.chartAR_25.options.smoothingWeight", "payload": state["smoothing"]},
        {"field": "state.chartMAP_5.options.smoothingWeight", "payload": state["smoothing"]},
        {"field": "state.chartMAR_5.options.smoothingWeight", "payload": state["smoothing"]},
        {"field": "state.chartAP_5.options.smoothingWeight", "payload": state["smoothing"]},
        {"field": "state.chartAR_5.options.smoothingWeight", "payload": state["smoothing"]}
    ]
    g.api.app.set_fields(g.task_id, fields)


def _save_link_to_ui(local_dir, app_url):
    # save report to file *.lnk (link to report)
    local_path = os.path.join(local_dir, _open_lnk_name)
    sly.fs.ensure_base_path(local_path)
    with open(local_path, "w") as text_file:
        print(app_url, file=text_file)


def upload_artifacts_and_log_progress():
    _save_link_to_ui(g.artifacts_dir, g.my_app.app_url)

    def upload_monitor(monitor, api: sly.Api, task_id, progress: sly.Progress):
        if progress.total == 0:
            progress.set(monitor.bytes_read, monitor.len, report=False)
        else:
            progress.set_current_value(monitor.bytes_read, report=False)
        _update_progress_ui("UploadDir", g.api, g.task_id, progress)

    progress = sly.Progress("Upload directory with training artifacts to Team Files", 0, is_size=True)
    progress_cb = partial(upload_monitor, api=g.api, task_id=g.task_id, progress=progress)

    remote_dir = f"/MMDetection3D/{g.task_id}_{g.project_info.name}"
    res_dir = g.api.file.upload_directory(g.team_id, g.artifacts_dir, remote_dir, progress_size_cb=progress_cb)
    return res_dir


def init_class_charts_series(state):
    per_class_series = []
    for class_name in state["selectedClasses"]:
        per_class_series.append({
            "name": class_name,
            "data": []
        })
    fields = [
        {"field": "state.chartAP_25.series", "payload": per_class_series},
        {"field": "state.chartAR_25.series", "payload": per_class_series},
        {"field": "state.chartAP_5.series", "payload": per_class_series},
        {"field": "state.chartAR_5.series", "payload": per_class_series}
    ]
    g.api.app.set_fields(g.task_id, fields)


@g.my_app.callback("train")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def train(api: sly.Api, task_id, context, state, app_logger):
    try:
        init_class_charts_series(state)
        sly.json.dump_json_file(state, os.path.join(g.info_dir, "ui_state.json"))
        
        cfg = init_cfg(state)
        print(cfg.pretty_text) # TODO: for debug
        print(cfg)
        print(cfg['min_radius'])

        # TODO: bug: save latest even if 'latest' false
        os.makedirs(os.path.join(g.checkpoints_dir, cfg.work_dir.split('/')[-1]), exist_ok=True)
        cfg.dump(osp.join(g.checkpoints_dir, cfg.work_dir.split('/')[-1], "config.py"))
        
        # debug upload config
        api.file.upload(g.team_id, os.path.join(g.checkpoints_dir, cfg.work_dir.split('/')[-1], "config.py"), f"/MMDetection3DCONFIG/{task_id}/config.py")
        
        model = build_model(
            cfg.model,
            train_cfg=cfg.get('train_cfg'),
            test_cfg=cfg.get('test_cfg'))
        
        datasets = [build_dataset(cfg.data.train)]
        model.CLASSES = datasets[0].CLASSES
        train_model(
            model,
            datasets,
            cfg,
            distributed=False,
            validate=True)

        fields = [
            {"field": "data.progressEpoch", "payload": None},
            {"field": "data.progressIter", "payload": None},
            {"field": "data.eta", "payload": None},
        ]
        g.api.app.set_fields(g.task_id, fields)

        remote_dir = upload_artifacts_and_log_progress()
        file_info = api.file.get_info_by_path(g.team_id, os.path.join(remote_dir, _open_lnk_name))
        api.task.set_output_directory(task_id, file_info.id, remote_dir)

        fields = [
            {"field": "data.outputUrl", "payload": g.api.file.get_url(file_info.id)},
            {"field": "data.outputName", "payload": remote_dir},
            {"field": "state.doneMonitoring", "payload": True},
            {"field": "state.started", "payload": False},
        ]
        g.api.app.set_fields(g.task_id, fields)

        # stop application
        g.my_app.stop()

    except Exception as e:
        g.api.app.set_field(task_id, "state.started", False)
        sly.logger.info(e)
        raise e  # app will handle this error and show modal window