import os
import shutil
import supervisely_lib as sly
from supervisely.app.v1.app_service import AppService
import pathlib
import sys
import json
from dotenv import load_dotenv
# import pkg_resources
# import zipfile

root_source_path = str(pathlib.Path(sys.argv[0]).parents[2])
sly.logger.info(f"Root source directory: {root_source_path}")
sys.path.append(root_source_path)

# @TODO: for debug
debug_env_path = os.path.join(root_source_path, "serve", "debug.env")
secret_debug_env_path = os.path.join(root_source_path, "serve", "secret_debug.env")
load_dotenv(debug_env_path)
load_dotenv(secret_debug_env_path, override=True)  

my_app = AppService()
api = my_app.public_api
task_id = my_app.task_id

# @TODO: for debug
# sly.fs.clean_dir(my_app.data_dir)

team_id = int(os.environ['context.teamId'])
workspace_id = int(os.environ['context.workspaceId'])

remote_weights_path = None
device = None
model = None
meta: sly.ProjectMeta = None
local_weights_path = None
model_config_local_path = None
model_name = None
ptc_range_centered = []
train_data_centered = None

configs_dir = "/tmp/mmdet3d/configs"
if not os.path.exists(configs_dir):
    repo_dir = "/tmp/mmdet3d/mmdetection3d"
    if os.path.isdir(repo_dir):
        shutil.move(os.path.join(repo_dir, "configs"), configs_dir)
os.makedirs(configs_dir, exist_ok=True)
config_folders_cnt = len(os.listdir(configs_dir))
sly.logger.info(f"Found {config_folders_cnt} folders in {configs_dir} directory.")