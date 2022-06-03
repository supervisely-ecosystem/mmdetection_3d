from mmdet3d.models.builder import DETECTORS
from mmdet3d.models.detectors.parta2 import PartA2
from mmdet3d.models.roi_heads.part_aggregation_roi_head import PartAggregationROIHead
from mmdet3d.models.roi_heads.mask_heads import PointwiseSemanticHead
from mmdet3d.models.builder import HEADS
import torch.nn as nn

@DETECTORS.register_module()
class PartA2Fixed(PartA2):
    def __init__(self,
                 voxel_layer,
                 voxel_encoder,
                 middle_encoder,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(PartA2Fixed, self).__init__(
            voxel_layer=voxel_layer,
            voxel_encoder=voxel_encoder,
            middle_encoder=middle_encoder,
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

