import supervisely as sly
import sly_globals as g
import input_project
import models
import classes
import splits
import data_preparation as data
import augs
import params
import monitoring


@sly.timeit
def init(data, state):
    state["activeStep"] = 1
    state["restartFrom"] = None
    input_project.init(data, state)
    models.init(data, state)
    classes.init(g.api, data, state, g.project_id, g.project_meta)
    splits.init(g.project_info, data, state)
    data.init(data, state)
    augs.init(data, state)
    params.init(data, state)
    monitoring.init(data, state)


@g.my_app.callback("restart")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def restart(api: sly.Api, task_id, context, state, app_logger):
    restart_from_step = state["restartFrom"]
    data = {}
    state = {}

    if restart_from_step <= 2:
        if restart_from_step == 2:
            models.restart(data, state)
        else:
            models.init(data, state)

    if restart_from_step <= 3:
        if restart_from_step == 3:
            classes.restart(data, state)
        else:
            classes.init(g.api, data, state, g.project_id, g.project_meta)

    if restart_from_step <= 4:
        if restart_from_step == 4:
            splits.restart(data, state)
        else:
            splits.init(g.project_info, g.project_meta, data, state)
    
    if restart_from_step <= 5:
        if restart_from_step == 5:
            data.restart(data, state)
        else:
            data.init(data, state)
    
    if restart_from_step <= 6:
        if restart_from_step == 6:
            augs.restart(data, state)
        else:
            augs.init(data, state)
    
    if restart_from_step <= 7:
        if restart_from_step == 7:
            params.restart(data, state)
        else:
            params.init(data, state)

    monitoring.init(data, state)

    fields = [
        {"field": "data", "payload": data, "append": True, "recursive": False},
        {"field": "state", "payload": state, "append": True, "recursive": False},
        {"field": "state.restartFrom", "payload": None},
        {"field": "state.activeStep", "payload": restart_from_step},
        {"field": "data.scrollIntoView", "payload": f"step{restart_from_step}"},
    ]
    g.api.app.set_fields(g.task_id, fields)