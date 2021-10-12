import torch
from torch import nn
from ..builder import HEADS, build_head, build_roi_extractor
from .atss_head import ATSSHead
import mmcv
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize
import numpy as np

@HEADS.register_module()
class ATSSMiddleFusionChannelAttentionHead(ATSSHead):
    """Simplest base roi head including one bbox head and one mask head."""

    def __init__(self,
                 num_classes,
                 in_channels,
                 pool_size=1,
                 stacked_convs=4,
                 conv_cfg=None,
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 loss_centerness=dict(type='CrossEntropyLoss',use_sigmoid=True,loss_weight=1.0),
                 init_cfg=dict(type='Normal',layer='Conv2d',std=0.01,override=dict(type='Normal',name='atss_cls',std=0.01,bias_prob=0.01)),
                 enlarge=False,
                 **kwargs):
        super(ATSSMiddleFusionChannelAttentionHead, self).__init__(num_classes=num_classes,
                 in_channels=in_channels,
                 stacked_convs=stacked_convs,
                 conv_cfg=conv_cfg,
                 norm_cfg=norm_cfg,
                 loss_centerness=loss_centerness,
                 init_cfg=init_cfg,
                 **kwargs)
        self.enlarge = enlarge
        # self.offset_modules = nn.ModuleList([
        #     nn.Linear(2 * self.bbox_head.conv_out_channels * self.bbox_head.roi_feat_area, self.bbox_head.conv_out_channels),
        #     nn.ReLU(),
        #     nn.Linear(self.bbox_head.conv_out_channels, self.bbox_head.conv_out_channels),
        #     nn.ReLU(),
        #     nn.Linear(self.bbox_head.conv_out_channels, 2), #rescale
        # ])
        # self.offset_modules = nn.ModuleList([
        #     nn.Linear(2*256*7*7,256),
        #     nn.ReLU(),
        #     nn.Linear(256, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 2),  # rescale
        # ])

        # # fusion b
        # self.fusion_modules_b = nn.ModuleList([
        #     nn.Conv2d(2 * self.bbox_head.conv_out_channels, self.bbox_head.conv_out_channels,3,stride=1, padding=1),
        #     nn.ReLU()
        # ])
        # fusion b
        # self.fusion_modules_b = nn.ModuleList([
        #     nn.Conv2d(2 * 256, 256, 3, stride=1, padding=1),
        #     nn.ReLU()
        # ])

        pool_area = pool_size ** 2
        middle_channels = 256
        out_chan = [256, 256, 256, 256, 256]
        self.max_pool = nn.AdaptiveMaxPool2d((pool_size, pool_size))
        self.avg_pool = nn.AdaptiveAvgPool2d((pool_size, pool_size))
        self.fusion_module = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(out_chan[i] * 2 * pool_area, middle_channels),
                nn.ReLU(),
                nn.Linear(middle_channels, middle_channels),
                nn.ReLU(),
                nn.Linear(middle_channels, out_chan[i] * 2 * pool_area),
                nn.Sigmoid()
            ])
            for i in range(len(out_chan))
        ])
        self.down_channel_module = nn.ModuleList([
            nn.Conv2d(out_chan[i] * 2, out_chan[i], 1)
            for i in range(len(out_chan))
        ])

    def feature_fusion(self, acid_feats, iodine_feats):
        feats = []
        for i in range(len(self.fusion_module)):
            acid_feat = acid_feats[i]
            iodine_feat = iodine_feats[i]
            feat = torch.cat([acid_feat, iodine_feat], dim=1)
            attn = self.max_pool(feat) + self.avg_pool(feat)
            attn = attn.reshape(attn.shape[0], -1)
            for m in self.fusion_module[i]:
                attn = m(attn)
            feat = feat * attn[..., None, None]
            feat = self.down_channel_module[i](feat)
            feats.append(feat)
        return feats

    def _offset_forward(self, feats):

        feats = nn.AdaptiveAvgPool2d(7)(feats)  #torch.nn.AdaptiveMaxPool2d(output_size, return_indices=False)
        feats = feats.flatten(1)
        for m in self.offset_modules:
            feats = m(feats)
        return feats



    def forward_train(self,
                      acid_feats, iodine_feats,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): Features from FPN.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used

        Returns:
            tuple:
                losses: (dict[str, Tensor]): A dictionary of loss components.
                proposal_list (list[Tensor]): Proposals of each image.
        """
        x = self.feature_fusion(acid_feats, iodine_feats)

        outs = self(x)
        if gt_labels is None:
            loss_inputs = outs + (gt_bboxes, img_metas)
        else:
            loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        if proposal_cfg is None:
            return losses
        else:
            proposal_list = self.get_bboxes(*outs, img_metas, cfg=proposal_cfg)
            return losses, proposal_list


    def simple_test(self, acid_feats, iodine_feats, img_metas, rescale=False):
        """Test function without test-time augmentation.

        Args:
            feats (tuple[torch.Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is ``bboxes`` with shape (n, 5),
                where 5 represent (tl_x, tl_y, br_x, br_y, score).
                The shape of the second tensor in the tuple is ``labels``
                with shape (n,)
        """
        return self.simple_test_bboxes(acid_feats, iodine_feats, img_metas, rescale=rescale)


    def simple_test_bboxes(self, acid_feats, iodine_feats, img_metas, rescale=False):
        """Test det bboxes without test-time augmentation, can be applied in
        DenseHead except for ``RPNHead`` and its variants, e.g., ``GARPNHead``,
        etc.

        Args:
            feats (tuple[torch.Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is ``bboxes`` with shape (n, 5),
                where 5 represent (tl_x, tl_y, br_x, br_y, score).
                The shape of the second tensor in the tuple is ``labels``
                with shape (n,)
        """

        feats = self.feature_fusion(acid_feats, iodine_feats)

        outs = self(feats)
        results_list = self.get_bboxes(*outs, img_metas, rescale=rescale)
        return results_list