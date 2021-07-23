import warnings

from torch import nn
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .two_stage import TwoStageDetector


@DETECTORS.register_module()
class FasterRCNNDual(TwoStageDetector):
    """Implementation of `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_"""

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck = None,
                 pretrained = None,
                 init_cfg = None):
        super(TwoStageDetector, self).__init__(init_cfg)
        self.stages = ['acid', 'iodine']
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained
        if self.stages[0] not in backbone:
            backbone = {stage: backbone for stage in self.stages}
        self.backbone = nn.ModuleDict({stage: build_backbone(backbone[stage]) for stage in self.stages})
        if neck is not None:
            if self.stages[0] not in neck:
                neck = {stage: neck for stage in self.stages}
            self.neck = nn.ModuleDict({stage: build_neck(neck[stage]) for stage in self.stages})
        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            self.rpn_head = build_head(rpn_head_)

        if roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            roi_head.update(train_cfg=rcnn_train_cfg)
            roi_head.update(test_cfg=test_cfg.rcnn)
            roi_head.pretrained = pretrained
            self.roi_head = build_head(roi_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
