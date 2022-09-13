import splits
import supervisely_lib as sly
import sly_globals as g
from supervisely.geometry.cuboid_3d import Cuboid3d

selected_classes = None

def init(api: sly.Api, data, state, project_id, project_meta: sly.ProjectMeta):
    stats = api.project.get_stats(project_id)
    class_images = {}
    for item in stats["images"]["objectClasses"]:
        class_images[item["objectClass"]["name"]] = item["total"]

    class_objects = {}
    for item in stats["objects"]["items"]:
        class_objects[item["objectClass"]["name"]] = item["total"]

    selected_classes_json = []
    for obj_class in project_meta.obj_classes:
        if obj_class.geometry_type == Cuboid3d:
            selected_classes_json.append(obj_class.to_json())

    for obj_class in selected_classes_json:
        obj_class["imagesCount"] = class_images[obj_class["title"]]
        obj_class["objectsCount"] = class_objects[obj_class["title"]]

    data["classes"] = selected_classes_json
    state["selectedClasses"] = []
    state["classes"] = len(selected_classes_json) * [True]
    state["totalItems"] = g.project_info.items_count
    data["doneClasses"] = False
    state["collapsedClasses"] = True
    state["disabledClasses"] = True


@g.my_app.callback("use_classes")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def use_classes(api: sly.Api, task_id, context, state, app_logger):
    global selected_classes
    selected_classes = state["selectedClasses"]

    fields = [
        {"field": "state.selectedClasses", "payload": state["selectedClasses"]},
        {"field": "data.doneClasses", "payload": True},
        {"field": "state.collapsedSplits", "payload": False},
        {"field": "state.disabledSplits", "payload": False},
        {"field": "state.activeStep", "payload": 4},
    ]
    g.api.app.set_fields(g.task_id, fields)


def restart(data, state):
    data["doneClasses"] = False