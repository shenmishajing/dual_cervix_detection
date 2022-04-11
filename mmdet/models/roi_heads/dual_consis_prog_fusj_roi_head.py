import math
import torch
from torch import nn

from mmdet.core import bbox2result, bbox2roi
from ..builder import HEADS, build_head, build_roi_extractor
from .standard_roi_head import StandardRoIHead


@HEADS.register_module()
class DualConsisProgFusjRoIHead(StandardRoIHead):
    """Simplest base roi head including one bbox head and one mask head."""

    def __init__(self,
                 enlarge = False,
                 **kwargs):
        super(DualConsisProgFusjRoIHead, self).__init__(**kwargs)
        self.enlarge = enlarge
        self.offset_modules = nn.ModuleList([
            nn.Linear(2 * self.bbox_head.conv_out_channels * self.bbox_head.roi_feat_area, self.bbox_head.conv_out_channels),
            nn.ReLU(),
            nn.Linear(self.bbox_head.conv_out_channels, self.bbox_head.conv_out_channels),
            nn.ReLU(),
            nn.Linear(self.bbox_head.conv_out_channels, 2),
        ])
        self.fusion_modules = nn.ModuleList([
            nn.Linear(2 * self.bbox_head.conv_out_channels * self.bbox_head.roi_feat_area, 1),
            nn.Sigmoid()
        ])

        # self.fusion_modules_b = nn.ModuleList([
        #     nn.Conv2d(2 * self.bbox_head.conv_out_channels, self.bbox_head.conv_out_channels,3,stride=1, padding=1),
        #     nn.ReLU()
        # ])

        # fusion c
        # self.fusion_modules_c = nn.ModuleList([
        #     nn.Conv2d(2, 2, 1, stride=1),
        #     nn.ReLU()
        # ])

        # #fusion cb
        # self.fusion_modules_cb1 = nn.Conv2d(2, 2, 1, stride=1)
        # self.fusion_modules_cb2 = nn.Conv2d(2 * self.bbox_head.conv_out_channels, self.bbox_head.conv_out_channels, 3, stride=1, padding=1)


        # #fusion f
        # self.fusion_modules_f = nn.ModuleList([
        #     nn.Linear(2 * self.bbox_head.conv_out_channels, self.bbox_head.conv_out_channels),
        #     nn.ReLU(),
        #     nn.Linear(self.bbox_head.conv_out_channels, self.bbox_head.conv_out_channels),
        #     nn.ReLU(),
        #     nn.Linear(self.bbox_head.conv_out_channels, 2 *self.bbox_head.conv_out_channels),
        #     nn.Sigmoid()
        # ])
        # self.fusion_conv_f = nn.Conv2d(2 * self.bbox_head.conv_out_channels, self.bbox_head.conv_out_channels, 3, stride=1, padding=1)

        # fusion g or g_b
        self.fusion_conv_g1 = nn.Conv2d(2, 16, 3, stride=1, padding=1)
        self.fusion_relu = nn.ReLU()
        self.fusion_conv_g2 = nn.Conv2d(16, 2, 3, stride=1, padding=1)
        self.fusion_conv_g4 = nn.Conv2d(2 * self.bbox_head.conv_out_channels, self.bbox_head.conv_out_channels, 3,
                                        stride=1, padding=1)

        # # fusion g_c
        # self.fusion_conv_g1 = nn.Conv2d(2, 64, 3, stride=1, padding=1)
        # self.fusion_relu = nn.ReLU()
        # self.fusion_conv_g2 = nn.Conv2d(64, 2, 3, stride=1, padding=1)
        # self.fusion_conv_g4 = nn.Conv2d(2 * self.bbox_head.conv_out_channels, self.bbox_head.conv_out_channels, 3,
        #                                 stride=1, padding=1)

        # # fusion g_d
        # self.fusion_conv_g1 = nn.Conv2d(2, 16, 3, stride=1, padding=1)
        # self.fusion_relu1 = nn.ReLU()
        # self.fusion_conv_g11 = nn.Conv2d(16, 16, 3, stride=1, padding=1)
        # self.fusion_relu11 = nn.ReLU()
        # self.fusion_conv_g2 = nn.Conv2d(16, 2, 3, stride=1, padding=1)
        # self.fusion_conv_g4 = nn.Conv2d(2 * self.bbox_head.conv_out_channels, self.bbox_head.conv_out_channels, 3,
        #                                 stride=1, padding=1)


        # # fusion h or h_b
        # self.fusion_conv_h1 = nn.Conv2d(2 * self.bbox_head.conv_out_channels, self.bbox_head.conv_out_channels, 3, stride=1, padding=1)
        # self.fusion_relu_h1 = nn.ReLU()
        # self.fusion_conv_h2 = nn.Conv2d(98, 49, 3, stride=1, padding=1)
        # self.fusion_relu_h2 = nn.ReLU()
        # self.fusion_conv_h3 = nn.Conv2d(2 * self.bbox_head.conv_out_channels, self.bbox_head.conv_out_channels, 3, stride=1, padding=1)


        # # fusion i
        # self.fusion_conv_i1 = nn.Conv2d(98, 49, 3, stride=1, padding=1)
        # self.fusion_irelu1 = nn.ReLU()
        # self.fusion_conv_i2 = nn.Conv2d(49, 2, 3, stride=1, padding=1)
        # self.fusion_conv_i3 = nn.Conv2d(2 * self.bbox_head.conv_out_channels, self.bbox_head.conv_out_channels, 3, stride=1, padding=1)


        ##fusion j
        self.hidden_dim = self.bbox_head.conv_out_channels #256
        self.dim_dynamic = self.bbox_head.roi_feat_area #49
        self.num_dynamic = 2 # 2
        self.num_params = self.hidden_dim * self.dim_dynamic #256*49
        self.dynamic_layer = nn.Linear(self.hidden_dim, self.num_dynamic * self.hidden_dim) #256 to (2*256)   ,49 can be obtain by cat

        self.norm1 = nn.LayerNorm(self.dim_dynamic) #49
        self.norm2 = nn.LayerNorm(self.hidden_dim) #256

        self.activation = nn.ReLU(inplace=True)

        self.norm3 = nn.LayerNorm(self.hidden_dim)


    def fusion_feature(self, prim_feats, aux_feats):
        # #fusion a
        # prim_rate_feats = prim_feats.flatten(1)
        # aux_rate_feats = aux_feats.flatten(1)
        # rate = torch.cat([prim_rate_feats, aux_rate_feats], dim = 1)
        # for m in self.fusion_modules:
        #     rate = m(rate)
        # feats = prim_feats * rate[..., None, None] + aux_feats * (1 - rate[..., None, None])
        #

        # # # fusion B
        # feats = torch.cat([prim_feats, aux_feats], dim=1)
        # for m in self.fusion_modules_b:
        #     feats = m(feats)

        # # # fusion C
        #
        # prim_maxp = nn.MaxPool2d(prim_feats.size()[-1])(prim_feats)
        # prim_avgp = nn.AvgPool2d(prim_feats.size()[-1])(prim_feats)
        # prim_pool = (prim_maxp+prim_avgp)
        # aux_maxp = nn.MaxPool2d(aux_feats.size()[-1])(aux_feats)
        # aux_avgp = nn.AvgPool2d(aux_feats.size()[-1])(aux_feats)
        # aux_pool = (aux_maxp + aux_avgp)
        # pool = torch.cat([prim_pool, aux_pool], dim=-1).permute([0, 3, 2, 1]).reshape((prim_pool.size()[0],-1,16,16))
        # for m in self.fusion_modules_c:
        #     pool = m(pool)
        # weights=nn.Softmax(dim=1)(pool)
        # feats = prim_feats*(weights[:,0,:,:].reshape((weights.size()[0],-1,1,1))) + aux_feats*(weights[:,1,:,:].reshape((weights.size()[0],-1,1,1)))

        # # # fusion cb
        # prim_maxp = nn.MaxPool2d(prim_feats.size()[-1])(prim_feats)
        # prim_avgp = nn.AvgPool2d(prim_feats.size()[-1])(prim_feats)
        # prim_pool = (prim_maxp+prim_avgp)
        # aux_maxp = nn.MaxPool2d(aux_feats.size()[-1])(aux_feats)
        # aux_avgp = nn.AvgPool2d(aux_feats.size()[-1])(aux_feats)
        # aux_pool = (aux_maxp + aux_avgp)
        # pool = torch.cat([prim_pool, aux_pool], dim=-1).permute([0, 3, 2, 1]).reshape((prim_pool.size()[0],-1,16,16))
        # pool = self.fusion_modules_cb1(pool)
        # weights=nn.Softmax(dim=1)(pool)
        # weights = torch.cat((weights[:,0,:,:].reshape((weights.size()[0],-1,1,1)),weights[:,1,:,:].reshape((weights.size()[0],-1,1,1))),dim=1)
        # feats = torch.cat([prim_feats, aux_feats], dim=1)*weights
        # feats = self.fusion_modules_cb2(feats)



        # #fusion F
        # feats = torch.cat([prim_feats, aux_feats], dim=1)
        # maxp = nn.MaxPool2d(feats.size()[-1])(feats)
        # avgp = nn.AvgPool2d(feats.size()[-1])(feats)
        # pool = (maxp + avgp).squeeze(-1).squeeze(-1)
        # for m in self.fusion_modules_f:
        #   pool = m(pool)
        # feats = feats*(pool.unsqueeze(-1).unsqueeze(-1))
        # feats = self.fusion_conv_f(feats)

        # # fusion G
        # prim_avgp = nn.AvgPool2d(prim_feats.size()[-1])(prim_feats)
        # aux_avgp = nn.AvgPool2d(prim_feats.size()[-1])(aux_feats)
        # feats = torch.cat([prim_avgp, aux_avgp], dim=-1).permute([0, 3, 2, 1])
        # f1 = self.fusion_conv_g1(feats)
        # f1 = self.fusion_relu(f1)
        # f1 = self.fusion_conv_g2(f1)
        # weight = nn.Softmax(dim=1)(f1)
        # weight = torch.cat([weight[:, 0], weight[:, 1]], dim=-1)
        # feats = torch.cat([prim_feats, aux_feats], dim=1) * weight.permute([0, 2, 1])[..., None]
        # feats = self.fusion_conv_g4(feats)

        # # fusion G_b
        # prim_avgp = nn.AvgPool2d(prim_feats.size()[-1])(prim_feats) + nn.MaxPool2d(prim_feats.size()[-1])(prim_feats)
        # aux_avgp = nn.AvgPool2d(prim_feats.size()[-1])(aux_feats) + nn.MaxPool2d(prim_feats.size()[-1])(aux_feats)
        # feats = torch.cat([prim_avgp, aux_avgp], dim=-1).permute([0, 3, 2, 1])
        # f1 = self.fusion_conv_g1(feats)
        # f1 = self.fusion_relu(f1)
        # f1 = self.fusion_conv_g2(f1)
        # weight = nn.Softmax(dim=1)(f1)
        # weight = torch.cat([weight[:, 0], weight[:, 1]], dim=-1)
        # feats = torch.cat([prim_feats, aux_feats], dim=1) * weight.permute([0, 2, 1])[..., None]
        # feats = self.fusion_conv_g4(feats)


        # # fusion G_c
        # prim_avgp = nn.AvgPool2d(prim_feats.size()[-1])(prim_feats)
        # aux_avgp = nn.AvgPool2d(prim_feats.size()[-1])(aux_feats)
        # feats = torch.cat([prim_avgp, aux_avgp], dim=-1).permute([0, 3, 2, 1])
        # f1 = self.fusion_conv_g1(feats)
        # f1 = self.fusion_relu(f1)
        # f1 = self.fusion_conv_g2(f1)
        # weight = nn.Softmax(dim=1)(f1)
        # weight = torch.cat([weight[:, 0], weight[:, 1]], dim=-1)
        # feats = torch.cat([prim_feats, aux_feats], dim=1) * weight.permute([0, 2, 1])[..., None]
        # feats = self.fusion_conv_g4(feats)



        # # fusion g_d
        # prim_avgp = nn.AvgPool2d(prim_feats.size()[-1])(prim_feats)
        # aux_avgp = nn.AvgPool2d(prim_feats.size()[-1])(aux_feats)
        # feats = torch.cat([prim_avgp, aux_avgp], dim=-1).permute([0, 3, 2, 1])
        # f1 = self.fusion_conv_g1(feats)
        # f1 = self.fusion_relu1(f1)
        # f1 = self.fusion_conv_g11(f1)
        # f1 = self.fusion_relu11(f1)
        # f1 = self.fusion_conv_g2(f1)
        # weight = nn.Softmax(dim=1)(f1)
        # weight = torch.cat([weight[:, 0], weight[:, 1]], dim=-1)
        # feats = torch.cat([prim_feats, aux_feats], dim=1) * weight.permute([0, 2, 1])[..., None]
        # feats = self.fusion_conv_g4(feats)




        # # fusion h
        # N = prim_feats.size()[0]
        # feats_c = torch.cat([prim_feats, aux_feats], dim=1)
        # feats_c = self.fusion_conv_h1(feats_c)
        #
        # feats_s = torch.cat([prim_feats.reshape((N,256,1,-1)), aux_feats.reshape((N,256,1,-1))], dim=-1).permute([0, 3, 2, 1])
        # feats_s = self.fusion_conv_h2(feats_s)
        # feats_s = feats_s.reshape((N,7,7,256)).permute([0, 3, 2, 1])
        # feats = feats_c +feats_s


        # # fusion h_b
        # N = prim_feats.size()[0]
        # feats_c = torch.cat([prim_feats, aux_feats], dim=1)
        # feats_c = self.fusion_conv_h1(feats_c)
        # feats_c = self.fusion_relu_h1(feats_c)
        #
        # feats_s = torch.cat([prim_feats.reshape((N,256,1,-1)), aux_feats.reshape((N,256,1,-1))], dim=-1).permute([0, 3, 2, 1])
        # feats_s = self.fusion_conv_h2(feats_s)
        # feats_s = self.fusion_relu_h2(feats_s)
        # feats_s = feats_s.reshape((N,7,7,256)).permute([0, 3, 2, 1])
        # feats = torch.cat([feats_c, feats_s], dim=1)
        # feats = self.fusion_conv_h3(feats)





        # # #fusion i
        # N = prim_feats.size()[0]
        # feats_i_ori = torch.cat([prim_feats.reshape((N, 256,1,-1)), aux_feats.reshape((N, 256,1,-1))], dim=-1).permute([0, 3, 2, 1])
        # feats_i = self.fusion_conv_i1(feats_i_ori)
        # feats_i = self.fusion_irelu1(feats_i)
        # feats_i = self.fusion_conv_i2(feats_i)
        # weight = nn.Softmax(dim=1)(feats_i)
        # feats = torch.cat([(feats_i_ori[:,:49].permute([0, 3, 2, 1])*weight[:, 0].permute([0, 2, 1])[..., None]).reshape((N, 256 , 7, 7)),
        #                    (feats_i_ori[:,49:].permute([0, 3, 2, 1])*weight[:, 1].permute([0, 2, 1])[..., None]).reshape((N, 256, 7, 7))], dim=1)
        # feats = self.fusion_conv_i3(feats)


        ##### fusion j
        '''
        from https://github.com/henbucuoshanghai/sparsercnn/blob/master/projects/SparseRCNN/sparsercnn/head.py   in  class DynamicConv(nn.Module)
           prim_feats:(N*7*7*256)
           aux_feats:(N*7*7*256)
           pro_features: (1,  N * nr_boxes, self.d_model)
           roi_features: (49, N * nr_boxes, self.d_model)
        '''
        N = prim_feats.size()[0]
        roi_features = prim_feats.reshape((N, -1, 256))  # obtain N*49*256
        aux_feats = aux_feats.reshape((N, -1, 256))  # obtain N*49*256
        features = roi_features  # obtain N*49*256

        for i in range(aux_feats.size()[1]):
            pro_features = aux_feats[:, i, :].unsqueeze(1).permute(1, 0, 2)
            if i == 0:
                parameters = self.dynamic_layer(pro_features).permute(1, 0, 2)
            else:
                parameters = torch.cat((parameters, self.dynamic_layer(pro_features).permute(1, 0, 2)),1)  # obtain N*(i+1) *(2*256)

        param1 = parameters[:, :, :self.hidden_dim].reshape(-1, self.hidden_dim, self.dim_dynamic)  # obtain N*256*49
        param2 = parameters[:, :, self.hidden_dim:].reshape(-1, self.dim_dynamic, self.hidden_dim)  # obtain N*49*256

        features = torch.bmm(features, param1)  # obtain N*49*49
        features = self.norm1(features)
        features = self.activation(features)  # obtain N*49*49

        features = torch.bmm(features, param2)  # obtain N*49*256
        features = self.norm2(features)
        features = self.activation(features)  # obtain N*49*256

        features = features + roi_features  # obtain N*(49*256)
        feats = self.norm3(features)
        feats = feats.view(prim_feats.size())
        return feats


    def _offset_forward(self, feats):
        feats = feats.flatten(1)
        for m in self.offset_modules:
            feats = m(feats)
        return feats

    def forward_train(self,
                      acid_feats=None, iodine_feats=None,
                      img_metas=None,
                      acid_proposal_list=None,
                      acid_gt_bboxes=None,
                      acid_gt_labels=None,
                      acid_gt_bboxes_ignore = None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            proposals (list[Tensors]): list of region proposals.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            if acid_gt_bboxes_ignore is None:
                acid_gt_bboxes_ignore = [None for _ in range(num_imgs)]

            acid_sampling_results = []
            for i in range(num_imgs):
                acid_assign_result = self.bbox_assigner.assign(
                    acid_proposal_list[i], acid_gt_bboxes[i], acid_gt_bboxes_ignore[i],
                    acid_gt_labels[i])
                acid_sampling_result = self.bbox_sampler.sample(
                    acid_assign_result,
                    acid_proposal_list[i],
                    acid_gt_bboxes[i],
                    acid_gt_labels[i],
                    feats = [lvl_feat[i][None] for lvl_feat in acid_feats])
                acid_sampling_results.append(acid_sampling_result)


        losses = dict()
        # bbox head forward and loss
        if self.with_bbox:
            bbox_results = self._bbox_forward_train(acid_feats, iodine_feats,
                                                    acid_sampling_results,
                                                    acid_gt_bboxes,
                                                    acid_gt_labels,
                                                    img_metas)
            losses.update(bbox_results['loss_bbox'])

        return losses

    def _bbox_forward(self, acid_feats, iodine_feats, acid_rois, img_metas):
        """Box head forward function used in both training and testing."""
        # acid
        acid_bbox_feats = self.bbox_roi_extractor(acid_feats[:self.bbox_roi_extractor.num_inputs], acid_rois)
        acid_iodine_rois = acid_rois.clone()
        if self.enlarge:
            acid_iodine_rois = torch.cat(
                [acid_iodine_rois[:, 0, None], acid_iodine_rois[:, 1:3] - self.enlarge, acid_iodine_rois[:, 3:] + self.enlarge], dim = 1)
        acid_iodine_bbox_feats = self.bbox_roi_extractor(iodine_feats[:self.bbox_roi_extractor.num_inputs], acid_iodine_rois)
        acid_proposal_offsets = self._offset_forward(torch.cat([acid_bbox_feats, acid_iodine_bbox_feats], dim = 1))
        acid_proposal_offsets_scaled = []
        for i in range(len(img_metas)):
            cur_offsets = acid_proposal_offsets[acid_rois[:, 0] == i]
            cur_offsets = cur_offsets * cur_offsets.new_tensor(img_metas[i]['pad_shape'][:2])
            acid_proposal_offsets_scaled.append(cur_offsets)
        acid_proposal_offsets_scaled = torch.cat(acid_proposal_offsets_scaled)
        acid_proposal_offsets_added = torch.cat(
            [acid_proposal_offsets_scaled.new_zeros(acid_proposal_offsets_scaled.shape[0], 1), acid_proposal_offsets_scaled,
             acid_proposal_offsets_scaled], dim = 1)
        acid_iodine_rois = acid_rois + acid_proposal_offsets_added
        acid_iodine_bbox_feats = self.bbox_roi_extractor(iodine_feats[:self.bbox_roi_extractor.num_inputs], acid_iodine_rois)
        acid_fusion_bbox_feats = self.fusion_feature(acid_bbox_feats, acid_iodine_bbox_feats)


        if self.with_shared_head:
            acid_bbox_feats = self.shared_head(acid_bbox_feats)
            acid_fusion_bbox_feats = self.shared_head(acid_fusion_bbox_feats)
            acid_iodine_bbox_feats = self.shared_head(acid_iodine_bbox_feats)
        acid_cls_score, acid_bbox_pred = self.bbox_head(acid_bbox_feats)
        acid_fusion_cls_score, acid_fusion_bbox_pred = self.bbox_head(acid_fusion_bbox_feats)
        acid_iodine_cls_score, acid_iodine_bbox_pred = self.bbox_head(acid_iodine_bbox_feats)

        bbox_results = dict(acid_cls_score = acid_cls_score, acid_bbox_pred = acid_bbox_pred, acid_bbox_feats = acid_bbox_feats,
                            acid_proposal_offsets = acid_proposal_offsets,iodine_bbox_feats=acid_iodine_bbox_feats, iodine_cls_score = acid_iodine_cls_score,
                            acid_fusion_cls_score = acid_fusion_cls_score, acid_fusion_bbox_pred = acid_fusion_bbox_pred)
        return bbox_results

    def _bbox_forward_train(self, acid_feats, iodine_feats, acid_sampling_results, acid_gt_bboxes,
                            acid_gt_labels, img_metas):
        """Run forward function and calculate loss for box head in training."""
        acid_rois = bbox2roi([res.bboxes for res in acid_sampling_results])
        bbox_results = self._bbox_forward(acid_feats, iodine_feats, acid_rois, img_metas)

        bbox_targets = self.bbox_head.get_targets(acid_sampling_results,
                                                  acid_gt_bboxes, acid_gt_labels,
                                                  img_metas, self.train_cfg)
        loss_bbox = self.bbox_head.loss(bbox_results['acid_cls_score'], bbox_results['iodine_cls_score'],
                                        bbox_results['acid_bbox_pred'],
                                        bbox_results['acid_bbox_feats'], bbox_results['iodine_bbox_feats'],
                                        bbox_results['acid_fusion_cls_score'],
                                        bbox_results['acid_fusion_bbox_pred'],
                                        acid_rois,
                                        *bbox_targets)

        bbox_results.update(loss_bbox = loss_bbox)
        return bbox_results

    def simple_test(self,
                    acid_feats=None, iodine_feats=None,
                    acid_proposal_list=None,
                    img_metas=None,
                    acid_proposals = None, iodine_proposals = None,
                    rescale = False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        acid_det_bboxes, acid_det_labels = \
            self.simple_test_bboxes(acid_feats, iodine_feats, img_metas, acid_proposal_list, self.test_cfg,
                                    rescale = rescale)

        return [bbox2result(acid_det_bboxes[i], acid_det_labels[i], self.bbox_head.num_classes) for i in range(len(acid_det_bboxes))]


    def simple_test_bboxes(self,
                           acid_feats, iodine_feats,
                           img_metas,
                           acid_proposals,
                           rcnn_test_cfg,
                           rescale = False):
        """Test only det bboxes without augmentation.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            img_metas (list[dict]): Image meta info.
            proposals (List[Tensor]): Region proposals.
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of R-CNN.
            rescale (bool): If True, return boxes in original image space.
                Default: False.

        Returns:
            tuple[list[Tensor], list[Tensor]]: The first list contains
                the boxes of the corresponding image in a batch, each
                tensor has the shape (num_boxes, 5) and last dimension
                5 represent (tl_x, tl_y, br_x, br_y, score). Each Tensor
                in the second list is the labels with shape (num_boxes, ).
                The length of both lists should be equal to batch_size.
        """

        acid_rois = bbox2roi(acid_proposals)


        if acid_rois.shape[0] == 0:
            # There is no proposal in the whole batch
            return [acid_rois.new_zeros(0, 5)] * len(acid_proposals), [acid_rois.new_zeros((0,), dtype = torch.long)] * len(
                acid_proposals)

        bbox_results = self._bbox_forward(acid_feats, iodine_feats, acid_rois, img_metas)
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        # split batch bbox prediction back to each image
        acid_cls_score = bbox_results['acid_cls_score']
        acid_bbox_pred = bbox_results['acid_bbox_pred']
        acid_num_proposals_per_img = tuple(len(p) for p in acid_proposals)
        acid_rois = acid_rois.split(acid_num_proposals_per_img, 0)
        acid_cls_score = acid_cls_score.split(acid_num_proposals_per_img, 0)

        # some detector with_reg is False, bbox_pred will be None
        if acid_bbox_pred is not None:
            # TODO move this to a sabl_roi_head
            # the bbox prediction of some detectors like SABL is not Tensor
            if isinstance(acid_bbox_pred, torch.Tensor):
                acid_bbox_pred = acid_bbox_pred.split(acid_num_proposals_per_img, 0)
            else:
                acid_bbox_pred = self.bbox_head.bbox_pred_split(
                    acid_bbox_pred, acid_num_proposals_per_img)
        else:
            acid_bbox_pred = (None,) * len(acid_proposals)

        # apply bbox post-processing to each image individually
        acid_det_bboxes = []
        acid_det_labels = []
        for i in range(len(acid_proposals)):
            if acid_rois[i].shape[0] == 0:
                # There is no proposal in the single image
                acid_det_bbox = acid_rois[i].new_zeros(0, 5)
                acid_det_label = acid_rois[i].new_zeros((0,), dtype = torch.long)
            else:
                acid_det_bbox, acid_det_label = self.bbox_head.get_bboxes(
                    acid_rois[i],
                    acid_cls_score[i],
                    acid_bbox_pred[i],
                    img_shapes[i],
                    scale_factors[i],
                    rescale = rescale,
                    cfg = rcnn_test_cfg)
            acid_det_bboxes.append(acid_det_bbox)
            acid_det_labels.append(acid_det_label)

        return acid_det_bboxes,acid_det_labels