import supervisely as sly
import sly_globals as g
import input_project
import models
import splits
import data_preparation
# import augs
# import architectures
# import hyperparameters
import classes
import monitoring
# import task


@sly.timeit
def init(data, state):
    state["activeStep"] = 1
    state["restartFrom"] = None
    input_project.init(data, state)
    models.init(data, state)
    # task.init(data, state)
    classes.init(g.api, data, state, g.project_id, g.project_meta)
    splits.init(g.project_info, data, state)
    data_preparation.init(data, state)
    # augs.init(data, state)
    # hyperparameters.init(data, state)
    monitoring.init(data, state)


@g.my_app.callback("restart")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def restart(api: sly.Api, task_id, context, state, app_logger):
    restart_from_step = state["restartFrom"]
    data = {}
    state = {}

    # if restart_from_step <= 2:
    #     if restart_from_step == 2:
    #         task.restart(data, state)
    #     else:
    #         task.init(data, state)

    # if restart_from_step <= 3:
    #     if restart_from_step == 3:
    #         architectures.restart(data, state)
    #     else:
    #         architectures.init(data, state)

    # if restart_from_step <= 4:
    #     if restart_from_step == 4:
    #         classes.restart(data, state)
    #     else:
    #         classes.init(g.api, data, state, g.project_id, g.project_meta)
    
    # if restart_from_step <= 5:
    #     if restart_from_step == 5:
    #         splits.restart(data, state)
    #     else:
    #         splits.init(g.project_info, g.project_meta, data, state)
    
    # if restart_from_step <= 6:
    #     if restart_from_step == 6:
    #         augs.restart(data, state)
    #     else:
    #         augs.init(data, state)
    
    # if restart_from_step <= 7:
    #     if restart_from_step == 7:
    #         hyperparameters.restart(data, state)
    #     else:
    #         hyperparameters.init(data, state)

    # monitoring.init(data, state)

    fields = [
        {"field": "data", "payload": data, "append": True, "recursive": False},
        {"field": "state", "payload": state, "append": True, "recursive": False},
        {"field": "state.restartFrom", "payload": None},
        {"field": "state.activeStep", "payload": restart_from_step},
        {"field": "data.scrollIntoView", "payload": f"step{restart_from_step}"},
    ]
    g.api.app.set_fields(g.task_id, fields)