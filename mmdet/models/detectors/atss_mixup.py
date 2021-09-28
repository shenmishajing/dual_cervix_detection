from .single_stage import SingleStageDetector
import torch
from mmdet.core import bbox2result
from ..builder import DETECTORS, build_backbone, build_head, build_neck
import numpy as np

@DETECTORS.register_module()
class ATSSMixup(SingleStageDetector):
    """Implementation of `ATSS <https://arxiv.org/abs/1912.02424>`_."""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(ATSSMixup, self).__init__(backbone, neck, bbox_head, train_cfg,
                                   test_cfg, pretrained, init_cfg)

    def mixup_data(self,img,gt_bboxes,gt_labels, img_metas):
        mix_ratio = 0.5
        # mixup two images
        img1, img2 = img[0].detach().cpu().numpy().transpose(1,2,0), img[1].detach().cpu().numpy().transpose(1,2,0)
        mix_bbox = torch.tensor(list(gt_bboxes[0].cpu().numpy()) + list(gt_bboxes[1].cpu().numpy())).cuda()
        mix_label = torch.tensor(list(gt_labels[0].cpu().numpy()) + list(gt_labels[1].cpu().numpy())).cuda()
        height = max(img1.shape[0], img2.shape[0])
        width = max(img1.shape[1], img2.shape[1])
        mix_img = np.zeros(shape=(height, width, 3), dtype='float32')
        mix_img[:img1.shape[0], :img1.shape[1], :] = img1.astype('float32') * mix_ratio
        mix_img[:img2.shape[0], :img2.shape[1], :] += img2.astype('float32') * (1. - mix_ratio)
        mix_img = torch.tensor(np.expand_dims(mix_img.transpose(2,0,1), axis=0), requires_grad=True).cuda() #mix_img= mix_img.astype('uint8')

        mix_img_metas=img_metas[-1]
        mix_img_metas['pad_shape']=(height, width, 3)
        mix_img_metas['batch_input_shape'] = (height, width)

        return torch.cat([img, mix_img], dim=0), gt_bboxes+[mix_bbox], gt_labels+[mix_label], img_metas+[mix_img_metas]

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        super(SingleStageDetector, self).forward_train(img, img_metas)
        img, gt_bboxes, gt_labels, img_metas = self.mixup_data(img, gt_bboxes, gt_labels, img_metas)
        x = self.extract_feat(img)


        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore)
        return losses




