import sly_globals as g
import models
from mmcv import ConfigDict
from mmdet.apis import set_random_seed
import os.path as osp


def init_cfg_optimizer(cfg, state):
    cfg.optimizer.type = state["optimizer"]
    cfg.optimizer.lr = state["lr"]
    cfg.optimizer.weight_decay = state["weightDecay"]
    if state["gradClipEnabled"]:
        if not hasattr(cfg, "optimizer_config"):
            cfg.optimizer_config = ConfigDict(
                grad_clip=ConfigDict(max_norm=state["maxNorm"], norm_type=2)
            )
        else:
            cfg.optimizer_config.grad_clip=ConfigDict(max_norm=state["maxNorm"], norm_type=2)
    if hasattr(cfg.optimizer, "eps"):
        delattr(cfg.optimizer, "eps")

    if state["optimizer"] == "SGD":
        if hasattr(cfg.optimizer, "betas"):
            delattr(cfg.optimizer, "betas")
        cfg.optimizer.momentum = state["momentum"]
        cfg.optimizer.nesterov = state["nesterov"]
    elif state["optimizer"] in ["Adam", "Adamax", "AdamW", "NAdam", "RAdam"]:
        if hasattr(cfg.optimizer, "momentum"):
            delattr(cfg.optimizer, "momentum")
        if hasattr(cfg.optimizer, "nesterov"):
            delattr(cfg.optimizer, "nesterov")
        cfg.optimizer.betas = (state["beta1"], state["beta2"])
        if state["optimizer"] in ["Adam", "AdamW"]:
            cfg.optimizer.amsgrad = state["amsgrad"]
        if state["optimizer"] == "NAdam":
            cfg.optimizer.momentum_decay = state["momentumDecay"]

def init_cfg_pipelines(cfg, state, dims):
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
        # elif pipeline_step.type == "ObjectNoise":
        #     # TODO: change in the future
        #     del cfg.train_pipeline[train_pipeline_len - 1 - idx]
        # elif pipeline_step.type == "PointSample":
        #     # TODO: change in the future
        #     del cfg.train_pipeline[train_pipeline_len - 1 - idx]
        elif pipeline_step.type == "RandomFlip3D":
            cfg.train_pipeline[train_pipeline_len - 1 - idx].flip_ratio_bev_horizontal = float(state["selectedAugs"]["flip_horizontal"])
            cfg.train_pipeline[train_pipeline_len - 1 - idx].flip_ratio_bev_vertical = float(state["selectedAugs"]["flip_vertical"])
        elif pipeline_step.type == "GlobalRotScaleTrans":
            cfg.train_pipeline[train_pipeline_len - 1 - idx].rot_range = [
                float(state["selectedAugs"]["global_rot_range"][0]),
                float(state["selectedAugs"]["global_rot_range"][1])
            ]
            cfg.train_pipeline[train_pipeline_len - 1 - idx].scale_ratio_range = [
                float(state["selectedAugs"]["global_scale_range"][0]),
                float(state["selectedAugs"]["global_scale_range"][1])
            ]
            cfg.train_pipeline[train_pipeline_len - 1 - idx].translation_std=[
                float(state["selectedAugs"]["global_translation_std"][0]),
                float(state["selectedAugs"]["global_translation_std"][1]),
                float(state["selectedAugs"]["global_translation_std"][2])
            ]
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
                elif transform.type == "DefaultFormatBundle3D":
                    cfg.test_pipeline[test_pipeline_len - 1 - idx].transforms[tr_idx].class_names=cfg.class_names
                # elif transform.type == "PointSample":
                #     # TODO: maybe change in the future
                #     del cfg.test_pipeline[test_pipeline_len - 1 - idx].transforms[tr_idx]

    eval_pipeline_len = len(cfg.eval_pipeline)
    for idx, pipeline_step in enumerate(cfg.eval_pipeline[::-1]):
        if pipeline_step.type == "LoadPointsFromMultiSweeps":
            del cfg.eval_pipeline[eval_pipeline_len - 1 - idx]
        elif pipeline_step.type == "LoadPointsFromFile":
            cfg.eval_pipeline[eval_pipeline_len - 1 - idx].load_dim = dims
            cfg.eval_pipeline[eval_pipeline_len - 1 - idx].use_dim = dims
        elif pipeline_step.type == "DefaultFormatBundle3D":
            cfg.eval_pipeline[eval_pipeline_len - 1 - idx].class_names = cfg.class_names


def init_cfg_splits(cfg):
    cfg.dataset_type = "SuperviselyDataset"
    cfg.data_root = g.project_dir

    cfg.data.train = ConfigDict()
    cfg.data.val = ConfigDict()
    cfg.data.test = ConfigDict()
    train_dataset = cfg.data.train
    val_dataset = cfg.data.val
    test_dataset = cfg.data.test

    train_dataset.pipeline = cfg.train_pipeline
    train_dataset.data_root = cfg.data_root
    train_dataset.type = cfg.dataset_type
    train_dataset.ann_file = osp.join(g.my_app.data_dir, "train.pkl")
    train_dataset.test_mode = False
    train_dataset.classes = cfg.class_names
    train_dataset.filter_empty_gt = False

    val_dataset.pipeline = cfg.test_pipeline
    val_dataset.data_root = cfg.data_root
    val_dataset.type = cfg.dataset_type
    val_dataset.ann_file = osp.join(g.my_app.data_dir, "val.pkl")
    val_dataset.test_mode = True
    val_dataset.classes = cfg.class_names
    
    test_dataset.pipeline = cfg.test_pipeline
    test_dataset.data_root = cfg.data_root
    test_dataset.type = cfg.dataset_type
    test_dataset.ann_file = osp.join(g.my_app.data_dir, "val.pkl")
    test_dataset.test_mode = True
    test_dataset.classes = cfg.class_names


def init_cfg_training(cfg, state):
    cfg.seed = state["randomSeed"]
    set_random_seed(cfg.seed, deterministic=True)

    cfg.data.samples_per_gpu = state["batchSizePerGPU"]
    cfg.data.workers_per_gpu = state["workersPerGPU"]
    cfg.data.persistent_workers = True

    # TODO: sync with state["gpusId"] if it will be needed
    cfg.gpu_ids = range(1)
    cfg.load_from = g.local_weights_path
    cfg.work_dir = g.my_app.data_dir

    if not hasattr(cfg, "runner"):
        cfg.runner = ConfigDict()
    cfg.runner.type = "EpochBasedRunner"
    cfg.runner.max_epochs = state["epochs"]
    if hasattr(cfg, "total_epochs"):
        cfg.total_epochs = state["epochs"]
    if hasattr(cfg.runner, "max_iters"):
        delattr(cfg.runner, "max_iters")

    cfg.log_config.interval = state["logConfigInterval"]
    cfg.log_config.hooks = [
        dict(type='SuperviselyLoggerHook', by_epoch=False)
    ]

def init_cfg_eval(cfg, state):
    cfg.evaluation.pipeline = cfg.eval_pipeline
    cfg.evaluation.interval = state["valInterval"]
    cfg.evaluation.save_best = "auto" if state["saveBest"] else None
    cfg.evaluation.rule = "greater"
    cfg.evaluation.out_dir = g.checkpoints_dir
    cfg.evaluation.by_epoch = True

def init_cfg_checkpoint(cfg, state):
    cfg.checkpoint_config.interval = state["checkpointInterval"]
    cfg.checkpoint_config.by_epoch = True
    cfg.checkpoint_config.max_keep_ckpts = state["maxKeepCkpts"] if state["maxKeepCkptsEnabled"] else None
    cfg.checkpoint_config.out_dir = g.checkpoints_dir


def init_cfg_lr(cfg, state):
    lr_config = ConfigDict(
        policy=state["lrPolicy"],
        by_epoch=state["schedulerByEpochs"],
        warmup=state["warmup"] if state["useWarmup"] else None,
        warmup_by_epoch=state["warmupByEpoch"],
        warmup_iters=state["warmupIters"],
        warmup_ratio=state["warmupRatio"]
    )
    if state["lrPolicy"] == "Step":
        lr_config["step"] = state["lr_step"]
        lr_config["gamma"] = state["gamma"]
        lr_config["min_lr"] = state["minLR"]
    elif state["lrPolicy"] == "Exp":
        lr_config["gamma"] = state["gamma"]
    elif state["lrPolicy"] == "Poly":
        lr_config["min_lr"] = state["minLR"]
        lr_config["power"] = state["power"]
    elif state["lrPolicy"] == "Inv":
        lr_config["gamma"] = state["gamma"]
        lr_config["power"] = state["power"]
    elif state["lrPolicy"] == "CosineAnnealing":
        lr_config["min_lr"] = state["minLR"] if state["minLREnabled"] else None
        lr_config["min_lr_ratio"] = state["minLRRatio"] if not state["minLREnabled"] else None
    elif state["lrPolicy"] == "FlatCosineAnnealing":
        lr_config["min_lr"] = state["minLR"] if state["minLREnabled"] else None
        lr_config["min_lr_ratio"] = state["minLRRatio"] if not state["minLREnabled"] else None
        lr_config["start_percent"] = state["startPercent"]
    elif state["lrPolicy"] == "CosineRestart":
        lr_config["min_lr"] = state["minLR"] if state["minLREnabled"] else None
        lr_config["min_lr_ratio"] = state["minLRRatio"] if not state["minLREnabled"] else None
        lr_config["periods"] = [int(period) for period in state["periods"].split(",")]
        lr_config["restart_weights"] = [float(weight) for weight in state["restartWeights"].split(",")]
    elif state["lrPolicy"] == "Cyclic":
        lr_config["target_ratio"] = (state["highestLRRatio"], state["lowestLRRatio"])
        lr_config["cyclic_times"] = state["cyclicTimes"]
        lr_config["step_ratio_up"] = state["stepRatioUp"]
        lr_config["anneal_strategy"] = state["annealStrategy"]
        lr_config["gamma"] = state["cyclicGamma"]
    elif state["lrPolicy"] == "OneCycle":
        lr_config["anneal_strategy"] = state["annealStrategy"]
        lr_config["max_lr"] = [float(maxlr) for maxlr in state["maxLR"].split(",")]
        lr_config["total_steps"] = state["totalSteps"] if state["totalStepsEnabled"] else None
        lr_config["pct_start"] = state["pctStart"]
        lr_config["div_factor"] = state["divFactor"]
        lr_config["final_div_factor"] = state["finalDivFactor"]
        lr_config["three_phase"] = state["threePhase"]
    cfg.lr_config = lr_config


def init_model(cfg, dims):

    # TODO: add in the future
    if hasattr(cfg, "db_sampler"):
        cfg.db_sampler = None
    # if cfg.pretrained_model == "3DSSD":
    #     cfg.model.test_cfg.max_output_num = 500 # instead of default 100
    if cfg.pretrained_model == "CenterPoint":
        ss = cfg.model.pts_middle_encoder.sparse_shape
        pcr = cfg.point_cloud_range
        cfg.voxel_size = [
            (pcr[3] - pcr[0]) / ss[1],
            (pcr[4] - pcr[1]) / ss[2],
            (pcr[5] - pcr[2]) / (ss[0] - 1),
        ]

        cfg.model.type = "CenterPointFixed"
        cfg.model.pts_bbox_head.type = "CenterHeadWithVel"
        cfg.model.pts_middle_encoder.in_channels = dims
        cfg.model.pts_voxel_encoder.num_features = dims

        cfg.model.pts_voxel_layer.voxel_size = cfg.voxel_size
        # cfg.model.pts_voxel_layer.max_voxels = [90000, 120000] ??
        cfg.model.pts_voxel_layer.point_cloud_range = pcr
        cfg.model.pts_bbox_head.bbox_coder.post_center_range = pcr
        cfg.model.pts_bbox_head.bbox_coder.pc_range = pcr
        cfg.model.pts_bbox_head.bbox_coder.voxel_size = cfg.voxel_size[:2]
        cfg.model.pts_bbox_head.bbox_coder.code_size = 7
        if hasattr(cfg.model.pts_bbox_head.common_heads, "vel"):
            delattr(cfg.model.pts_bbox_head.common_heads, "vel")
        # TODO: maybe allow to customize by user?
        cfg.model.train_cfg.pts.code_weights = [1., 1., 1., 1., 1., 1., 1., 1.] 
        cfg.model.train_cfg.pts.point_cloud_range = pcr
        cfg.model.train_cfg.pts.voxel_size = cfg.voxel_size
        cfg.model.test_cfg.pts.post_center_limit_range = pcr
        #cfg.model.test_cfg.pts.score_threshold = 0
        cfg.model.test_cfg.pts.pc_range = pcr
        cfg.model.test_cfg.pts.voxel_size = cfg.voxel_size[:2]

        class_dicts = []
        for selected_class in cfg.class_names:
            class_dicts.append(dict(num_class=1, class_names=[selected_class]))
        cfg.model.pts_bbox_head.tasks = class_dicts
    else:
        raise NotImplementedError(f"Current model {cfg.pretrained_model} is not supported now.")

def init_cfg(state):
    cfg = models.cfg
    dims = 4

    cfg.class_names = state["selectedClasses"]
    cfg.center_coords = state["center_coords"] # Custom parameter, [x, y, z]
    cfg.point_cloud_range = state["point_cloud_range"]

    init_model(cfg, dims)
    init_cfg_optimizer(cfg, state)
    init_cfg_training(cfg, state)
    init_cfg_pipelines(cfg, state, dims)
    init_cfg_splits(cfg)
    init_cfg_eval(cfg, state)
    init_cfg_checkpoint(cfg, state)
    init_cfg_lr(cfg, state)

    return cfg