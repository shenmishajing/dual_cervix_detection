import torch
from torch import nn
from ..builder import HEADS, build_head, build_roi_extractor
from .atss_head import ATSSHead
import mmcv
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize
import numpy as np

@HEADS.register_module()
class ATSSFusionSelfmixHead(ATSSHead):
    """Simplest base roi head including one bbox head and one mask head."""

    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs=4,
                 conv_cfg=None,
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 loss_centerness=dict(type='CrossEntropyLoss',use_sigmoid=True,loss_weight=1.0),
                 init_cfg=dict(type='Normal',layer='Conv2d',std=0.01,override=dict(type='Normal',name='atss_cls',std=0.01,bias_prob=0.01)),
                 enlarge=False,
                 **kwargs):
        super(ATSSFusionSelfmixHead, self).__init__(num_classes=num_classes,
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
        self.offset_modules = nn.ModuleList([
            nn.Linear(2*256*7*7,256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2),  # rescale
        ])

        # # fusion b
        # self.fusion_modules_b = nn.ModuleList([
        #     nn.Conv2d(2 * self.bbox_head.conv_out_channels, self.bbox_head.conv_out_channels,3,stride=1, padding=1),
        #     nn.ReLU()
        # ])



        ch_in=256
        reduction=16
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局自适应池化

        self.fc = nn.Sequential(
            nn.Linear(ch_in* 2, ch_in* 2 // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in* 2 // reduction, ch_in* 2, bias=False),
            nn.Sigmoid()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )

        self.down_channel_module = nn.Conv2d(ch_in * 2, ch_in, 1)

        self.channel_module = nn.Conv2d(ch_in, ch_in, 1)


    def fusion_feature(self, feats1, feats2):
        # # fusion B +SE attention
        b, c, _, _ = feats1.size()
        y = self.avg_pool(feats1).view(b, c)
        y = self.fc2(y).view(b, c, 1, 1)
        feats1 = feats1 * y.expand_as(feats1)

        b, c, _, _ = feats2.size()
        y = self.avg_pool(feats2).view(b, c)
        y = self.fc2(y).view(b, c, 1, 1)
        feats2 = feats2 * y.expand_as(feats2)


        feats = self.channel_module(feats1 + feats2)

        return feats


    def fusion_later(self, feats1, feats2, feats3):
        # # fusion B +SE attention

        b, c, _, _ = feats1.size()
        y = self.avg_pool(feats1).view(b, c)
        y = self.fc2(y).view(b, c, 1, 1)
        feats1 = feats1 * y.expand_as(feats1)

        b, c, _, _ = feats2.size()
        y = self.avg_pool(feats2).view(b, c)
        y = self.fc2(y).view(b, c, 1, 1)
        feats2 = feats2 * y.expand_as(feats2)

        b, c, _, _ = feats3.size()
        y = self.avg_pool(feats3).view(b, c)
        y = self.fc2(y).view(b, c, 1, 1)
        feats3 = feats3 * y.expand_as(feats3)


        feats = self.channel_module(feats1 + feats2 + feats3)

        return feats


    def _offset_forward(self, feats):

        feats = nn.AdaptiveAvgPool2d(7)(feats)  #torch.nn.AdaptiveMaxPool2d(output_size, return_indices=False)
        feats = feats.flatten(1)
        for m in self.offset_modules:
            feats = m(feats)
        return feats

    def fusion_befor(self,acid_feats, iodine_feats,img_metas):
        x = []
        for lev in range(len(acid_feats)):
            acid_offsets = self._offset_forward(torch.cat([acid_feats[lev], iodine_feats[lev]], dim=1))  ### for fusion 1 and fusion 2
            acid_offsets = acid_offsets * acid_offsets.new_tensor(acid_feats[lev].size()[2:])  ### only for fusion 2
            if acid_feats[0].size()[0] ==2:
                acid_feats_lev_0 = acid_feats[lev][0][:,
                                   int(acid_offsets[0][0]):int(acid_feats[lev].size()[2] + acid_offsets[0][0]),
                                   int(acid_offsets[0][1]):int(acid_feats[lev].size()[3] + acid_offsets[0][1])]
                acid_feats_lev_1 = acid_feats[lev][1][:,
                                   int(acid_offsets[1][0]):int(acid_feats[lev].size()[2] + acid_offsets[1][0]),
                                   int(acid_offsets[1][1]):int(acid_feats[lev].size()[3] + acid_offsets[1][1])]
                # 裁出的图size 为0 时用原图替代
                if acid_feats_lev_0.size()[1]==0 or acid_feats_lev_0.size()[2]==0:
                    acid_feats_lev_0 = acid_feats[lev][0]
                if acid_feats_lev_1.size()[1]==0 or acid_feats_lev_1.size()[2]==0:
                    acid_feats_lev_1 = acid_feats[lev][1]

                acid_feats_lev_0 = np.resize(acid_feats_lev_0.clone().detach().cpu().numpy(),
                                             tuple(iodine_feats[lev][0].size()))
                acid_feats_lev_1 = np.resize(acid_feats_lev_1.clone().detach().cpu().numpy(),
                                             tuple(iodine_feats[lev][0].size()))

                acid_feats_lev_0 = torch.tensor(np.expand_dims(acid_feats_lev_0, axis=0), requires_grad=True).cuda()
                acid_feats_lev_1 = torch.tensor(np.expand_dims(acid_feats_lev_1, axis=0), requires_grad=True).cuda()
                acid_feats_lev = torch.cat([acid_feats_lev_0, acid_feats_lev_1], dim=0)
            elif acid_feats[0].size()[0] ==1:
                acid_feats_lev_0 = acid_feats[lev][0][:,
                                   int(acid_offsets[0][0]):int(acid_feats[lev].size()[2] + acid_offsets[0][0]),
                                   int(acid_offsets[0][1]):int(acid_feats[lev].size()[3] + acid_offsets[0][1])]
                # 裁出的图size 为0 时用原图替代
                if acid_feats_lev_0.size()[1]==0 or acid_feats_lev_0.size()[2]==0:
                    acid_feats_lev_0 = acid_feats[lev][0]

                acid_feats_lev_0 = np.resize(acid_feats_lev_0.clone().detach().cpu().numpy(),
                                             tuple(iodine_feats[lev][0].size()))
                acid_feats_lev_0 = torch.tensor(np.expand_dims(acid_feats_lev_0, axis=0), requires_grad=True).cuda()
                acid_feats_lev = acid_feats_lev_0
            else:
                print('error dimention')


            x.append(self.fusion_feature(acid_feats_lev, iodine_feats[lev]))

        x = tuple(x)
        return x



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
        x = self.fusion_befor(acid_feats, iodine_feats, img_metas)

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

        feats = self.fusion_befor(acid_feats, iodine_feats, img_metas)

        outs = self(feats)
        results_list = self.get_bboxes(*outs, img_metas, rescale=rescale)
        return results_list