import os
import random
from collections import namedtuple
import supervisely as sly
import sly_globals as g
from sly_train_progress import get_progress_cb, reset_progress, init_progress

progress_index = 1

def init(data, state):
    data["projectId"] = g.project_info.id
    data["projectName"] = g.project_info.name
    data["projectImagesCount"] = g.project_info.items_count
    init_progress(progress_index, data)
    data["doneProject"] = False
    state["collapsedProject"] = False


@g.my_app.callback("download_project")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def download(api: sly.Api, task_id, context, state, app_logger):
    try:
        if not sly.fs.dir_exists(g.project_dir):
            sly.fs.mkdir(g.project_dir)
            sly.download_pointcloud_episode_project(g.api, g.project_id, g.project_dir, dataset_ids=None, download_realated_images=False, log_progress=True)

        g.project_fs = sly.PointcloudEpisodeProject(g.project_dir, sly.OpenMode.READ)
    except Exception as e:
        reset_progress(progress_index)
        raise e

    fields = [
        {"field": "data.doneProject", "payload": True},
        {"field": "state.collapsedModels", "payload": False},
        {"field": "state.disabledModels", "payload": False},
        {"field": "state.activeStep", "payload": 2},
    ]
    g.api.app.set_fields(g.task_id, fields)

