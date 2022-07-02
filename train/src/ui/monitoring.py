import supervisely as sly
import sly_globals as g
import os.path as osp

from mmcv import Config, ConfigDict
from mmdet.utils import setup_multi_processes
from mmdet3d.apis import init_random_seed, train_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet.apis import set_random_seed
from mmcv.runner import load_checkpoint
import centerpoint
from mmdet3d.datasets.custom_3d import Custom3DDataset
from mmcv.runner.hooks.optimizer import OptimizerHook


def init(data, state):
    state["collapsedMonitoring"] = True
    state["disabledMonitoring"] = True
    state["doneMonitoring"] = False

    state["started"] = False


@g.my_app.callback("train")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def train(api: sly.Api, task_id, context, state, app_logger):
    cfg_path = osp.join(g.source_path, 'centerpoint_custom_config.py')
    # cfg_path = osp.join(g.configs_dir, "centerpoint", "centerpoint_01voxel_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus.py")
    cfg = Config.fromfile(cfg_path)

    # print(cfg.pretty_text)

    setup_multi_processes(cfg)
    cfg.work_dir = g.my_app.data_dir
    cfg.gpu_ids = range(1)
    cfg.dataset_type = 'Custom3DDataset'
    cfg.data_root = g.project_dir
    cfg.class_names = state["selectedClasses"]
    if not hasattr(cfg, "runner"):
        cfg.runner = ConfigDict()
        cfg.runner.type = "EpochBasedRunner"
    if not hasattr(cfg.runner, "max_epochs"):
        cfg.runner.max_epochs = 10

    # TODO: ptc ranges in model config

    seed = init_random_seed(0)
    set_random_seed(seed, deterministic=False)
    cfg.seed = seed

    cfg.data.train.pipeline[0].load_dim = 3
    cfg.data.train.pipeline[0].use_dim = 3
    cfg.data.train.type = cfg.dataset_type
    cfg.data.train.data_root = cfg.data_root
    cfg.data.train.ann_file = osp.join(g.my_app.data_dir, "train.pkl")
    cfg.data.train.classes = cfg.class_names
    if hasattr(cfg.data.train, "dataset"):
        delattr(cfg.data.train, "dataset")

    cfg.data.val.pipeline[0].load_dim = 3
    cfg.data.val.pipeline[0].use_dim = 3
    cfg.data.val.type = cfg.dataset_type
    cfg.data.val.data_root = cfg.data_root
    cfg.data.val.ann_file = osp.join(g.my_app.data_dir, "val.pkl")
    cfg.data.val.classes = cfg.class_names
    if hasattr(cfg.data.val, "dataset"):
        delattr(cfg.data.val, "dataset")

    cfg.data.test.pipeline[0].load_dim = 3
    cfg.data.test.pipeline[0].use_dim = 3
    cfg.data.test.type = cfg.dataset_type
    cfg.data.test.data_root = cfg.data_root
    cfg.data.test.ann_file = osp.join(g.my_app.data_dir, "val.pkl")
    cfg.data.test.classes = cfg.class_names
    if hasattr(cfg.data.test, "dataset"):
        delattr(cfg.data.test, "dataset")

    cfg.model.type = "CenterPointFixed"
    cfg.model.pts_bbox_head.type = "CenterHeadWithVel"
    cfg.model.pts_middle_encoder.in_channels = 3

    class_dicts = []
    for selected_class in cfg.class_names:
        class_dicts.append(dict(num_class=1, class_names=[selected_class]))
    cfg.model.pts_bbox_head.tasks = class_dicts

    train_pipeline_len = len(cfg.data.train.pipeline)
    for idx, pipeline_step in enumerate(cfg.data.train.pipeline[::-1]):
        if pipeline_step.type == "LoadPointsFromMultiSweeps":
            del cfg.data.train.pipeline[train_pipeline_len - 1 - idx]
        elif pipeline_step.type == "ObjectNameFilter":
            cfg.data.train.pipeline[train_pipeline_len - 1 - idx].classes = cfg.class_names
        elif pipeline_step.type == "DefaultFormatBundle3D":
            cfg.data.train.pipeline[train_pipeline_len - 1 - idx].class_names = cfg.class_names

    val_pipeline_len = len(cfg.data.val.pipeline)
    for idx, pipeline_step in enumerate(cfg.data.val.pipeline[::-1]):
        if pipeline_step.type == "LoadPointsFromMultiSweeps":
            del cfg.data.val.pipeline[val_pipeline_len - 1 - idx]
        elif pipeline_step.type == "MultiScaleFlipAug3D":
            for tr_idx, transform in enumerate(pipeline_step.transforms):
                if transform.type == "PointsRangeFilter":
                    cfg.data.val.pipeline[val_pipeline_len - 1 - idx].transforms[tr_idx].point_cloud_range=cfg.point_cloud_range
                elif transform.type == "DefaultFormatBundle3D":
                    cfg.data.val.pipeline[val_pipeline_len - 1 - idx].transforms[tr_idx].class_names=cfg.class_names
        
    test_pipeline_len = len(cfg.data.test.pipeline)
    for idx, pipeline_step in enumerate(cfg.data.test.pipeline[::-1]):
        if pipeline_step.type == "LoadPointsFromMultiSweeps":
            del cfg.data.test.pipeline[test_pipeline_len - 1 - idx]
        elif pipeline_step.type == "MultiScaleFlipAug3D":
            for tr_idx, transform in enumerate(pipeline_step.transforms):
                if transform.type == "PointsRangeFilter":
                    cfg.data.test.pipeline[test_pipeline_len - 1 - idx].transforms[tr_idx].point_cloud_range=cfg.point_cloud_range
                elif transform.type == "DefaultFormatBundle3D":
                    cfg.data.test.pipeline[test_pipeline_len - 1 - idx].transforms[tr_idx].class_names=cfg.class_names

    if hasattr(cfg, "db_sampler"):
        cfg.db_sampler = None
        # delattr(cfg, "db_sampler")

    # TODO: cfg.model.pts_middle_encoder -> fix args
    # TODO: eval pipeline

    cfg.dump(osp.join(cfg.work_dir, osp.basename(cfg_path)))
    model = build_model(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    #model.init_weights()
    
    weights_path = osp.join(cfg.work_dir, "centerpoint_01voxel_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus_20201004_075317-26d8176c.pth")
    checkpoint = load_checkpoint(model, weights_path, map_location='cuda:0')
    datasets = [build_dataset(cfg.data.train)]
    model.CLASSES = datasets[0].CLASSES
    train_model(
        model,
        datasets,
        cfg,
        distributed=False,
        validate=True)
    print('The end')
