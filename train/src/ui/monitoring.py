import supervisely as sly
import sly_globals as g
import os, os.path as osp

from mmcv import Config, ConfigDict
from mmdet.utils import setup_multi_processes
from mmdet3d.apis import init_random_seed, train_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet.apis import set_random_seed
from mmcv.runner import load_checkpoint
import centerpoint
import sly_dataset



def init(data, state):
    state["collapsedMonitoring"] = True
    state["disabledMonitoring"] = True
    state["doneMonitoring"] = False

    state["started"] = False


@g.my_app.callback("train")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def train(api: sly.Api, task_id, context, state, app_logger):
    # cfg_path = osp.join(g.source_path, 'centerpoint_custom_config.py')
    # cfg_path = osp.join(g.configs_dir, "centerpoint", "centerpoint_01voxel_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus.py")
    # cfg_path = osp.join(g.configs_dir, "pointpillars", "hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d.py")
    cfg = Config.fromfile(g.model_config_local_path)

    dims = 4
    setup_multi_processes(cfg)
    cfg.work_dir = g.my_app.data_dir
    cfg.gpu_ids = range(1)
    cfg.dataset_type = "SuperviselyDataset"
    cfg.data_root = g.project_dir
    cfg.class_names = state["selectedClasses"]
    cfg.center_coords = state["center_coords"] # Custom parameter, [x, y, z]
    if not hasattr(cfg, "runner"):
        cfg.runner = ConfigDict()
        cfg.runner.type = "EpochBasedRunner"
    # if not hasattr(cfg.runner, "max_epochs"):
    cfg.runner.max_epochs = 30

    cfg.log_config.interval = 1
    cfg.optimizer.lr = 1e-3
    cfg.lr_config = ConfigDict()
    cfg.lr_config.policy = 'step'
    cfg.lr_config.step = 5
    cfg.lr_config.gamma = 0.1
    cfg.lr_config.min_lr = 1e-7

    # cfg.lr_config.target_ratio = (100, 1e-6)
    # cfg.lr_config.cyclic_times = 1
    # cfg.lr_config.step_ratio_up = 0.2
    # cfg.lr_config.anneal_strategy = 'linear'

    eval_pipeline_len = len(cfg.eval_pipeline)
    for idx, pipeline_step in enumerate(cfg.eval_pipeline[::-1]):
        if pipeline_step.type == "LoadPointsFromMultiSweeps":
            del cfg.eval_pipeline[eval_pipeline_len - 1 - idx]
        elif pipeline_step.type == "LoadPointsFromFile":
            cfg.eval_pipeline[eval_pipeline_len - 1 - idx].load_dim = dims
            cfg.eval_pipeline[eval_pipeline_len - 1 - idx].use_dim = dims
        elif pipeline_step.type == "DefaultFormatBundle3D":
            cfg.eval_pipeline[eval_pipeline_len - 1 - idx].class_names = cfg.class_names

    cfg.evaluation.interval = 5
    cfg.evaluation.pipeline = cfg.eval_pipeline
    # cfg.evaluation.save_best = "auto" if state["saveBest"] else None
    # cfg.evaluation.rule = "greater"
    cfg.evaluation.out_dir = g.checkpoints_dir
    cfg.evaluation.by_epoch = True
    # cfg.evaluation.classwise = True
    cfg.checkpoint_config.interval=5

    seed = init_random_seed(0)
    set_random_seed(seed, deterministic=True)
    cfg.seed = seed

    # TODO: choice: based on config or based on data (and any checks maybe)
    
    cfg.point_cloud_range = state["point_cloud_range"]
    ss = cfg.model.pts_middle_encoder.sparse_shape
    pcr = cfg.point_cloud_range
    cfg.voxel_size = [
        (pcr[3] - pcr[0]) / ss[1],
        (pcr[4] - pcr[1]) / ss[2],
        (pcr[5] - pcr[2]) / (ss[0] - 1),
    ]

    # TODO: add to serve?
    train_pipeline_len = len(cfg.train_pipeline)
    for idx, pipeline_step in enumerate(cfg.train_pipeline[::-1]):
        if pipeline_step.type == "LoadPointsFromMultiSweeps":
            del cfg.train_pipeline[train_pipeline_len - 1 - idx]
        elif pipeline_step.type == "LoadPointsFromFile":
            cfg.train_pipeline[train_pipeline_len - 1 - idx].load_dim = dims
            cfg.train_pipeline[train_pipeline_len - 1 - idx].use_dim = dims
        elif pipeline_step.type == "ObjectNameFilter":
            cfg.train_pipeline[train_pipeline_len - 1 - idx].classes = cfg.class_names
        elif pipeline_step.type == "DefaultFormatBundle3D":
            cfg.train_pipeline[train_pipeline_len - 1 - idx].class_names = cfg.class_names
        elif pipeline_step.type == "ObjectSample":
            # TODO: change in the future
            del cfg.train_pipeline[train_pipeline_len - 1 - idx]
            # cfg.train_pipeline[train_pipeline_len - 1 - idx].db_sampler = None
        # TODO: remove/change in the future
        elif pipeline_step.type == "RandomFlip3D":
            cfg.train_pipeline[train_pipeline_len - 1 - idx].flip_ratio_bev_horizontal = 0.
            cfg.train_pipeline[train_pipeline_len - 1 - idx].flip_ratio_bev_vertical = 0.
        elif pipeline_step.type == "GlobalRotScaleTrans":
            cfg.train_pipeline[train_pipeline_len - 1 - idx].rot_range = [0., 0.]
            cfg.train_pipeline[train_pipeline_len - 1 - idx].scale_ratio_range = [1., 1.]
        elif pipeline_step.type == "PointsRangeFilter":
            cfg.train_pipeline[train_pipeline_len - 1 - idx].point_cloud_range = cfg.point_cloud_range
        elif pipeline_step.type == "ObjectRangeFilter":
            cfg.train_pipeline[train_pipeline_len - 1 - idx].point_cloud_range = cfg.point_cloud_range
    
    test_pipeline_len = len(cfg.test_pipeline)
    for idx, pipeline_step in enumerate(cfg.test_pipeline[::-1]):
        if pipeline_step.type == "LoadPointsFromMultiSweeps":
            del cfg.test_pipeline[test_pipeline_len - 1 - idx]
        elif pipeline_step.type == "LoadPointsFromFile":
            cfg.test_pipeline[test_pipeline_len - 1 - idx].load_dim = dims
            cfg.test_pipeline[test_pipeline_len - 1 - idx].use_dim = dims
        elif pipeline_step.type == "MultiScaleFlipAug3D":
            for tr_idx, transform in enumerate(pipeline_step.transforms):
                if transform.type == "PointsRangeFilter":
                    cfg.test_pipeline[test_pipeline_len - 1 - idx].transforms[tr_idx].point_cloud_range=cfg.point_cloud_range
                if transform.type == "DefaultFormatBundle3D":
                    cfg.test_pipeline[test_pipeline_len - 1 - idx].transforms[tr_idx].class_names=cfg.class_names

    cfg.data.samples_per_gpu = 2
    cfg.data.workers_per_gpu = 2
    cfg.data.persistent_workers = True

    cfg.data.train.type = cfg.dataset_type
    cfg.data.train.pipeline = cfg.train_pipeline
    cfg.data.train.data_root = cfg.data_root
    cfg.data.train.ann_file = osp.join(g.my_app.data_dir, "train.pkl")
    cfg.data.train.classes = cfg.class_names
    cfg.data.train.filter_empty_gt = False
    if hasattr(cfg.data.train, "dataset"):
        delattr(cfg.data.train, "dataset")

    cfg.data.val.type = cfg.dataset_type
    cfg.data.val.pipeline = cfg.test_pipeline
    cfg.data.val.data_root = cfg.data_root
    cfg.data.val.ann_file = osp.join(g.my_app.data_dir, "val.pkl")
    cfg.data.val.classes = cfg.class_names
    if hasattr(cfg.data.val, "dataset"):
        delattr(cfg.data.val, "dataset")

    cfg.data.test.type = cfg.dataset_type
    cfg.data.test.pipeline = cfg.test_pipeline
    cfg.data.test.data_root = cfg.data_root
    cfg.data.test.ann_file = osp.join(g.my_app.data_dir, "val.pkl")
    cfg.data.test.classes = cfg.class_names
    if hasattr(cfg.data.test, "dataset"):
        delattr(cfg.data.test, "dataset")

    cfg.model.type = "CenterPointFixed"
    cfg.model.pts_bbox_head.type = "CenterHeadWithVel"
    cfg.model.pts_middle_encoder.in_channels = dims
    cfg.model.pts_voxel_encoder.num_features = dims

    # TODO: check all model params below
    cfg.model.pts_voxel_layer.voxel_size = cfg.voxel_size
    # cfg.model.pts_voxel_layer.max_voxels = [90000, 120000] ??
    cfg.model.pts_voxel_layer.point_cloud_range = cfg.point_cloud_range
    cfg.model.pts_bbox_head.bbox_coder.post_center_range = cfg.point_cloud_range
    cfg.model.pts_bbox_head.bbox_coder.pc_range = cfg.point_cloud_range
    cfg.model.pts_bbox_head.bbox_coder.voxel_size = cfg.voxel_size[:2]
    cfg.model.train_cfg.pts.code_weights = [1., 1., 1., 1., 1., 1., 1., 1., 0.1, 0.1] 
    cfg.model.train_cfg.pts.point_cloud_range = cfg.point_cloud_range
    cfg.model.train_cfg.pts.voxel_size = cfg.voxel_size
    # cfg.model.train_cfg.pts.min_radius = 2
    cfg.model.test_cfg.pts.post_center_limit_range = cfg.point_cloud_range
    # cfg.model.test_cfg.pts.min_radius = [] ????
    # cfg.model.test_cfg.pts.score_threshold = 0
    cfg.model.test_cfg.pts.pc_range = cfg.point_cloud_range
    cfg.model.test_cfg.pts.voxel_size = cfg.voxel_size[:2]

    class_dicts = []
    for selected_class in cfg.class_names:
        class_dicts.append(dict(num_class=1, class_names=[selected_class]))
    cfg.model.pts_bbox_head.tasks = class_dicts

    # TODO: check that centerize works on val correctly

    if hasattr(cfg, "db_sampler"):
        cfg.db_sampler = None
    # TODO: check samples_per_gpu for train and eval
    # TODO: db_sampler settings

    print(cfg.pretty_text)

    os.makedirs(os.path.join(g.checkpoints_dir, cfg.work_dir.split('/')[-1]), exist_ok=True)
    cfg.dump(osp.join(g.checkpoints_dir, cfg.work_dir.split('/')[-1], "config.py"))
    model = build_model(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    
    # weights_path = osp.join(cfg.work_dir, "weights.pth")
    checkpoint = load_checkpoint(model, g.local_weights_path, map_location='cuda:0')
    datasets = [build_dataset(cfg.data.train)]
    model.CLASSES = datasets[0].CLASSES
    train_model(
        model,
        datasets,
        cfg,
        distributed=False,
        validate=True)
    print('The end')
