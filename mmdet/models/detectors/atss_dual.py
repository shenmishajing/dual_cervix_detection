import warnings

from torch import nn
from mmdet.core import bbox2result
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .single_stage import SingleStageDetector
from mmcv.runner import auto_fp16


@DETECTORS.register_module()
class ATSSDual(SingleStageDetector):
    """Implementation of `ATSS <https://arxiv.org/abs/1912.02424>`_."""

    def __init__(self,
                 backbone,
                 neck = None,
                 bbox_head = None,
                 train_cfg = None,
                 test_cfg = None,
                 pretrained = None,
                 init_cfg = None):
        super(SingleStageDetector, self).__init__(init_cfg)
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
        bbox_head.update(train_cfg = train_cfg)
        bbox_head.update(test_cfg = test_cfg)
        self.bbox_head = build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def extract_feat(self, acid_img, iodine_img):
        """Directly extract features from the backbone+neck."""
        acid_feats = self.backbone['acid'](acid_img)
        if self.with_neck:
            acid_feats = self.neck['acid'](acid_feats)
        iodine_feats = self.backbone['iodine'](iodine_img)
        if self.with_neck:
            iodine_feats = self.neck['iodine'](iodine_feats)
        return acid_feats, iodine_feats

    def forward_dummy(self, acid_img, iodine_img):
        raise NotImplementedError

    def forward_train(self,
                      acid_img, iodine_img,
                      img_metas,
                      acid_gt_bboxes, iodine_gt_bboxes,
                      acid_gt_labels, iodine_gt_labels,
                      acid_gt_bboxes_ignore = None, iodine_gt_bboxes_ignore = None,
                      acid_proposals = None, iodine_proposals = None,
                      **kwargs):
        acid_feats, iodine_feats = self.extract_feat(acid_img, iodine_img)

        losses = self.bbox_head.forward_train(acid_feats, iodine_feats,
                                              img_metas,
                                              acid_gt_bboxes, iodine_gt_bboxes,
                                              acid_gt_labels, iodine_gt_labels,
                                              acid_gt_bboxes_ignore = None, iodine_gt_bboxes_ignore = None,
                                              acid_proposals = None, iodine_proposals = None,
                                              **kwargs)

        return losses

    def forward_test(self, acid_imgs, iodine_imgs, img_metas, **kwargs):
        for var, name in [(acid_imgs, 'acid_imgs'), (iodine_imgs, 'iodine_imgs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got {type(var)}')

        num_augs = len(acid_imgs)
        if num_augs != len(img_metas):
            raise ValueError(f'num of augmentations ({len(acid_imgs)}) '
                             f'!= num of image meta ({len(img_metas)})')
        if num_augs == 1:
            if 'proposals' in kwargs:
                kwargs['proposals'] = kwargs['proposals'][0]
            return self.simple_test(acid_imgs[0], iodine_imgs[0], img_metas[0], **kwargs)
        else:
            raise NotImplementedError

    @auto_fp16(apply_to = ('acid_img', 'iodine_img'))
    def forward(self, acid_img, iodine_img, img_metas, return_loss = True, **kwargs):
        if return_loss:
            return self.forward_train(acid_img, iodine_img, img_metas, **kwargs)
        else:
            return self.forward_test(acid_img, iodine_img, img_metas, **kwargs)

    def simple_test(self, acid_img, iodine_img, img_metas, rescale = False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        acid_feats, iodine_feats = self.extract_feat(acid_img, iodine_img)

        bbox_results = []
        results_lists = self.bbox_head.simple_test(acid_feats, iodine_feats, img_metas, rescale = rescale)

        for stage in self.stages:
            bbox_results.append(
                [bbox2result(det_bboxes, det_labels, self.bbox_head[stage].num_classes) for det_bboxes, det_labels in results_lists[stage]])

        return bbox_results
