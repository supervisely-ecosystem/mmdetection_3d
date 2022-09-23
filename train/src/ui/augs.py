import os, os.path as osp
import supervisely as sly
import sly_globals as g
import numpy as np

aug_templates = [{
        'name': 'No Augmentations',
        'augs': {
            # TODO: Add PointSample / VoxelBasedPointSampler?
            'global_rot_range': [0., 0.],
            'global_scale_range': [1.0, 1.0],
            'global_translation_std': [0., 0., 0.],
            'flip_horizontal': 0., 
            'flip_vertical': 0.
        }
    },{
        'name': 'Light',
        'augs': {
            'global_rot_range': [-np.pi * 0.1, np.pi * 0.1],
            'global_scale_range': [0.95, 1.05],
            'global_translation_std': [0.5, 0.5, 0.5],
            'flip_horizontal': 0.2,
            'flip_vertical': 0.2
        }
    },{
        'name': 'Medium',
        'augs': {
            'global_rot_range': [-np.pi * 0.5, np.pi * 0.5],
            'global_scale_range': [0.8, 1.2],
            'global_translation_std': [1.0, 1.0, 1.0],
            'flip_horizontal': 0.5,
            'flip_vertical': 0.5
        }
    },{
        'name': 'Hard',
        'augs': {
            'global_rot_range': [-np.pi, np.pi],
            'global_scale_range': [0.5, 1.5],
            'global_translation_std': [3.0, 3.0, 3.0],
            'flip_horizontal': 0.5,
            'flip_vertical': 0.5
        }
    }]

def init(data, state):
    state["augsTemplateName"] = 'Light'
    state["collapsedAugs"] = True
    state["disabledAugs"] = True
    data["doneAugs"] = False
    data["augTemplates"] = aug_templates
    state["selectedAugs"] = data["augTemplates"][1]['augs'].copy() # Light

def restart(data, state):
    data["doneAugs"] = False


@g.my_app.callback("select_template")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def use_augs(api: sly.Api, task_id, context, state, app_logger):
    global aug_templates
    augs = None
    for template in aug_templates:
        if template['name'] == state["augsTemplateName"]:
            augs = template['augs']
    if augs is None:
        augs = aug_templates[0]['augs']
    g.api.app.set_field(g.task_id, "state.selectedAugs", augs)


@g.my_app.callback("use_augs")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def use_augs(api: sly.Api, task_id, context, state, app_logger):
    fields = [
        {"field": "data.doneAugs", "payload": True},
        {"field": "state.collapsedParams", "payload": False},
        {"field": "state.disabledParams", "payload": False},
        {"field": "state.activeStep", "payload": 7},
    ]
    g.api.app.set_fields(g.task_id, fields)