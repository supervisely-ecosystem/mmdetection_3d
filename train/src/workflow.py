# Description: This file contains Versioning and Workflow features for the Train MMDetection3D custom model

import os
import supervisely as sly

from supervisely.api.file_api import FileInfo

def workflow_input(api: sly.Api, project_info: sly.ProjectInfo, state: dict = None):   
    try:       
        if project_info.type != sly.ProjectType.IMAGES.__str__():
            sly.logger.info(f"{project_info.type =} is not '{sly.ProjectType.IMAGES.__str__()}'. Project version will not be created.")
            project_version_id = None
        else:
            project_version_id = api.project.version.create(
                project_info, "Train MMDetection", f"This backup was created automatically by Supervisely before the Train MMDetection task with ID: {api.task_id}"
            )
    except Exception as e:
        sly.logger.warning(f"Failed to create a project version: {repr(e)}")
        project_version_id = None

    try:
        file_info = None
        if project_version_id is None:
            project_version_id = project_info.version.get("id", None) if project_info.version else None
        api.app.workflow.add_input_project(project_info.id, version_id=project_version_id)
        if state.get("weightsInitialization", None) == "custom":
            file_path = state.get("weightsPath", None)
            if file_path is None:
                sly.logger.debug("Workflow Input: weights file path is not specified. Cannot add input file to the workflow.")
            file_info = api.file.get_info_by_path(sly.env.team_id(), file_path)
        if file_info is not None:
            api.app.workflow.add_input_file(file_info, model_weight=True)
        sly.logger.debug(f"Workflow Input: Project ID - {project_info.id}, Project Version ID - {project_version_id}, Input File - {True if file_info else False}")
    except Exception as e:
        sly.logger.debug(f"Failed to add input to the workflow: {repr(e)}")


def workflow_output(api: sly.Api, remote_dir: str, state:dict):
    try:
        remote_dir = os.path.join(remote_dir, "checkpoints", "data")
        file_infos_list = api.file.list(sly.env.team_id(), remote_dir, recursive=False, return_type="fileinfo")
        all_checkpoints = []
        best_checkpoints = []
        for info in file_infos_list:
            if "best" in info.name:
                best_checkpoints.append(info)
            elif ".pth" in info.name:
                all_checkpoints.append(info)
        if len(best_checkpoints) > 1:
            best_file_info = sorted(best_checkpoints, key=lambda x: x.name, reverse=True)[0]
        elif len(best_checkpoints) == 1:
            best_file_info = best_checkpoints[0]
        else:
            best_file_info = None
        
        if len(all_checkpoints) > 1:
            last_file_info = sorted(all_checkpoints, key=lambda x: x.name, reverse=True)[0]
        elif len(all_checkpoints) == 1:
            last_file_info = all_checkpoints[0]
        else:
            last_file_info = None

        if best_file_info is None and last_file_info is not None:
            best_file_info = last_file_info
        elif best_file_info is None and last_file_info is None:
            sly.logger.debug(
                f"Workflow Output: No checkpoint files found in Team Files. Cannot set workflow output."
            )
            return

        
        module_id = api.task.get_info_by_id(api.task_id).get("meta", {}).get("app", {}).get("id")
        
        if state.get("weightsInitialization", None) == "custom":
            node_custom_title = "Train Custom Model"
        else:
            node_custom_title = None
        if best_file_info:
            node_settings = sly.WorkflowSettings(
                    title=node_custom_title, 
                    url = f"/apps/{module_id}/sessions/{api.task_id}" if module_id else f"apps/sessions/{api.task_id}",
                    url_title= "Show Results",
                )
            relation_settings = sly.WorkflowSettings(
                    title="Checkpoints",
                    icon="folder",
                    icon_color = "#FFA500",
                    icon_bg_color ="#FFE8BE",
                    url=f"/files/{best_file_info.id}/true", 
                    url_title = "Open Folder",
                )
            meta = sly.WorkflowMeta(relation_settings, node_settings)
            api.app.workflow.add_output_file(best_file_info, model_weight=True, meta=meta)
            sly.logger.debug(f"Workflow Output: Node custom title - {node_custom_title}, Best filename - {best_file_info}")
            sly.logger.debug(f"Workflow Output: Meta \n    {meta.as_dict}")
        else:
            sly.logger.debug(f"File {best_file_info} not found in Team Files. Cannot set workflow output.")
    except Exception as e:
        sly.logger.debug(f"Failed to add output to the workflow: {repr(e)}")
