from .single_stage import SingleStageDetector
import warnings
import torch
from mmdet.core import bbox2result
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base import BaseDetector
from mmcv.runner import auto_fp16

@DETECTORS.register_module()
class ATSSFusion(SingleStageDetector):
    """Implementation of `ATSS <https://arxiv.org/abs/1912.02424>`_."""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(SingleStageDetector, self).__init__(init_cfg)
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def extract_feat(self, acid_img, iodine_img):
        """Directly extract features from the backbone+neck."""
        return super(ATSSFusion, self).extract_feat(acid_img), super(ATSSFusion, self).extract_feat(iodine_img)

    def forward_train(self,
                      acid_img, iodine_img,
                      img_metas,
                      acid_gt_bboxes=None,iodine_gt_bboxes=None,
                      acid_gt_labels=None,iodine_gt_labels=None,
                      acid_gt_bboxes_ignore=None,iodine_gt_bboxes_ignore=None,
                      **kwargs):


        acid_feats, iodine_feats = self.extract_feat(acid_img, iodine_img)
        losses = self.bbox_head.forward_train(acid_feats, iodine_feats, img_metas, iodine_gt_bboxes,
                                              iodine_gt_labels, iodine_gt_bboxes_ignore)
        return losses






    @auto_fp16(apply_to=('acid_img', 'iodine_img'))
    def forward(self, acid_img, iodine_img, img_metas, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(acid_img, iodine_img, img_metas, **kwargs)
        else:
            return self.forward_test(acid_img[0], iodine_img[0], img_metas[0], **kwargs)


    def forward_test(self, acid_img, iodine_img, img_metas, rescale=False, **kwargs):
        """Test function without test-time augmentation.

        Args:
            img (torch.Tensor): Images with shape (N, C, H, W).
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        acid_feats, iodine_feats = self.extract_feat(acid_img, iodine_img)
        results_list = self.bbox_head.simple_test(
            acid_feats, iodine_feats, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results

