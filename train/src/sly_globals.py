import os
import shutil
import supervisely_lib as sly
from supervisely.app.v1.app_service import AppService
import pathlib
import sys
import json
# from dotenv import load_dotenv
# import pkg_resources
# import zipfile
from pathlib import Path

root_source_dir = str(Path(sys.argv[0]).parents[2])
sly.logger.info(f"Root source directory: {root_source_dir}")
sys.path.append(root_source_dir)
source_path = str(Path(sys.argv[0]).parents[0])
sly.logger.info(f"App source directory: {source_path}")
sys.path.append(source_path)
ui_sources_dir = os.path.join(source_path, "ui")
sly.logger.info(f"UI source directory: {ui_sources_dir}")
sys.path.append(ui_sources_dir)
sly.logger.info(f"Added to sys.path: {ui_sources_dir}")

# @TODO: for debug
# debug_env_path = os.path.join(root_source_dir, "train", "debug.env")
# secret_debug_env_path = os.path.join(root_source_dir, "train", "secret_debug.env")
# load_dotenv(debug_env_path)
# load_dotenv(secret_debug_env_path, override=True)  

my_app = AppService()
api = my_app.public_api
task_id = my_app.task_id

# @TODO: for debug
# sly.fs.clean_dir(my_app.data_dir)

team_id = int(os.environ['context.teamId'])
workspace_id = int(os.environ['context.workspaceId'])
project_id = int(os.environ['modal.state.slyProjectId'])
project_info = api.project.get_info_by_id(project_id)
project_type = project_info.type

project_dir = os.path.join(my_app.data_dir, "sly_project")
project_meta = sly.ProjectMeta.from_json(api.project.get_meta(project_id))
project_fs: sly.Project = None

artifacts_dir = os.path.join(my_app.data_dir, "artifacts")
sly.fs.mkdir(artifacts_dir)
info_dir = os.path.join(artifacts_dir, "info")
sly.fs.mkdir(info_dir)
checkpoints_dir = os.path.join(artifacts_dir, "checkpoints")
sly.fs.mkdir(checkpoints_dir)

meta: sly.ProjectMeta = None

configs_dir = "/tmp/mmdet3d/configs"
if not os.path.exists(configs_dir):
    repo_dir = "/tmp/mmdet3d/mmdetection3d"
    if os.path.isdir(repo_dir):
        shutil.move(os.path.join(repo_dir, "configs"), configs_dir)
os.makedirs(configs_dir, exist_ok=True)
config_folders_cnt = len(os.listdir(configs_dir))
sly.logger.info(f"Found {config_folders_cnt} folders in {configs_dir} directory.")
