import warnings
import torch
from torch import nn

from ..builder import DETECTORS, build_head, build_backbone, build_neck
from .two_stage_cervix import TwoStageCervixDetector
from .two_stage import TwoStageDetector


@DETECTORS.register_module()
class FasterRCNNMiddleFusionChannelAttention(TwoStageCervixDetector):
    """Implementation of `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_"""

    def __init__(self,
                 backbone,
                 neck = None,
                 rpn_head = None,
                 roi_head = None,
                 train_cfg = None,
                 test_cfg = None,
                 pretrained = None,
                 prim = None,
                 init_cfg = None):
        super(TwoStageDetector, self).__init__(init_cfg)

        if prim is None:
            self.prim = ['acid', 'iodine']
        elif not isinstance(prim, list):
            self.prim = [prim]
        else:
            self.prim = prim

        self.stages = ['acid', 'iodine']

        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
        if self.stages[0] not in backbone:
            if pretrained:
                backbone.pretrained = pretrained
            backbone = {stage: backbone for stage in self.stages}
        else:
            for stage in self.stages:
                if pretrained:
                    backbone[stage].pretrained = pretrained
        self.backbone = nn.ModuleDict({stage: build_backbone(backbone[stage]) for stage in self.stages})

        if neck is not None:
            if self.stages[0] not in neck:
                neck = {stage: neck for stage in self.stages}
            self.neck = nn.ModuleDict({stage: build_neck(neck[stage]) for stage in self.stages})

        pool_size = 1
        pool_area = pool_size ** 2
        middle_channels = 256
        self.max_pool = nn.AdaptiveMaxPool2d((pool_size, pool_size))
        self.avg_pool = nn.AdaptiveAvgPool2d((pool_size, pool_size))
        self.fusion_module = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(self.neck[self.stages[0]].out_channels * 2 * pool_area, middle_channels),
                nn.ReLU(),
                nn.Linear(middle_channels, middle_channels),
                nn.ReLU(),
                nn.Linear(middle_channels, self.neck[self.stages[0]].out_channels * 2 * pool_area),
                nn.Sigmoid()
            ])
            for _ in range(self.neck[self.stages[0]].num_outs)
        ])
        self.down_channel_module = nn.ModuleList([
            nn.Conv2d(self.neck[self.stages[0]].out_channels * 2, self.neck[self.stages[0]].out_channels, 1)
            for _ in range(self.neck[self.stages[0]].num_outs)
        ])

        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            if self.stages[0] not in rpn_head_:
                rpn_head_.update(train_cfg = rpn_train_cfg, test_cfg = test_cfg.rpn)
                rpn_head_ = {stage: rpn_head_ for stage in self.stages}
            else:
                for stage in self.stages:
                    rpn_head_[stage].update(train_cfg = rpn_train_cfg, test_cfg = test_cfg.rpn)
            self.rpn_head = nn.ModuleDict({stage: build_head(rpn_head_[stage]) for stage in self.stages})

        if roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            if self.stages[0] not in roi_head:
                roi_head.update(train_cfg = rcnn_train_cfg)
                roi_head.update(test_cfg = test_cfg.rcnn)
                roi_head.pretrained = pretrained
                roi_head = {stage: roi_head for stage in self.stages}
            else:
                for stage in self.stages:
                    roi_head[stage].update(train_cfg = rcnn_train_cfg)
                    roi_head[stage].update(test_cfg = test_cfg.rcnn)
                    roi_head[stage].pretrained = pretrained
            self.roi_head = nn.ModuleDict({stage: build_head(roi_head[stage]) for stage in self.stages})

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def _extract_feat(self, img, stage):
        """Directly extract features from the backbone+neck."""
        x = self.backbone[stage](img)
        if self.with_neck:
            x = self.neck[stage](x)
        return x

    def extract_feat(self, acid_img, iodine_img):
        """Directly extract features from the backbone+neck."""
        return self._extract_feat(acid_img, 'acid'), self._extract_feat(iodine_img, 'iodine')

    def feature_fusion(self, acid_feats, iodine_feats):
        feats = []
        for i in range(len(self.fusion_module)):
            acid_feat = acid_feats[i]
            iodine_feat = iodine_feats[i]
            feat = torch.cat([acid_feat, iodine_feat], dim = 1)
            attn = self.max_pool(feat) + self.avg_pool(feat)
            attn = attn.reshape(attn.shape[0], -1)
            for m in self.fusion_module[i]:
                attn = m(attn)
            feat = feat * attn[..., None, None]
            feat = self.down_channel_module[i](feat)
            feats.append(feat)
        return feats

    def forward_train(self,
                      acid_img, iodine_img,
                      img_metas,
                      acid_gt_bboxes, iodine_gt_bboxes,
                      acid_gt_labels, iodine_gt_labels,
                      acid_gt_bboxes_ignore = None, iodine_gt_bboxes_ignore = None,
                      acid_proposals = None, iodine_proposals = None,
                      **kwargs):
        acid_feats, iodine_feats = self.extract_feat(acid_img, iodine_img)
        feats = self.feature_fusion(acid_feats, iodine_feats)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal', self.test_cfg.rpn)
            acid_rpn_losses, acid_proposal_list = self.rpn_head['acid'].forward_train(
                feats, img_metas, acid_gt_bboxes, gt_labels = None, gt_bboxes_ignore = acid_gt_bboxes_ignore,
                proposal_cfg = proposal_cfg)
            if 'acid' in self.prim:
                for k, v in acid_rpn_losses.items():
                    losses['acid_' + k] = v
            iodine_rpn_losses, iodine_proposal_list = self.rpn_head['iodine'].forward_train(
                feats, img_metas, iodine_gt_bboxes, gt_labels = None, gt_bboxes_ignore = iodine_gt_bboxes_ignore,
                proposal_cfg = proposal_cfg)
            if 'iodine' in self.prim:
                for k, v in iodine_rpn_losses.items():
                    losses['iodine_' + k] = v
        else:
            acid_proposal_list = acid_proposals
            iodine_proposal_list = iodine_proposals

        acid_roi_losses = self.roi_head['acid'].forward_train(feats, img_metas, acid_proposal_list, acid_gt_bboxes, acid_gt_labels,
                                                              acid_gt_bboxes_ignore, **kwargs)
        if 'acid' in self.prim:
            for k, v in acid_roi_losses.items():
                losses['acid_' + k] = v

        iodine_roi_losses = self.roi_head['iodine'].forward_train(feats, img_metas, iodine_proposal_list, iodine_gt_bboxes,
                                                                  iodine_gt_labels,
                                                                  iodine_gt_bboxes_ignore, **kwargs)
        if 'iodine' in self.prim:
            for k, v in iodine_roi_losses.items():
                losses['iodine_' + k] = v

        return losses

    def simple_test(self, acid_img, iodine_img, img_metas, acid_proposals = None, iodine_proposals = None, rescale = False):
        """Test without augmentation."""
        acid_feats, iodine_feats = self.extract_feat(acid_img, iodine_img)
        feats = self.feature_fusion(acid_feats, iodine_feats)

        if acid_proposals is None:
            acid_proposal_list = self.rpn_head['acid'].simple_test_rpn(feats, img_metas)
        else:
            acid_proposal_list = acid_proposals
        if iodine_proposals is None:
            iodine_proposal_list = self.rpn_head['iodine'].simple_test_rpn(feats, img_metas)
        else:
            iodine_proposal_list = iodine_proposals

        acid_results = self.roi_head['acid'].simple_test(feats, acid_proposal_list, img_metas, rescale = rescale)
        iodine_results = self.roi_head['iodine'].simple_test(feats, iodine_proposal_list, img_metas, rescale = rescale)

        return acid_results, iodine_results
