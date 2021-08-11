_base_ = [
    '../_base_/models/faster_rcnn_dual_r50_fpn.py',
    '../_base_/datasets/dual_cervix/hsil/dual/dual_base.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
]

# model settings
model = dict(prim = None,  # 'acid', 'iodine' or None
    roi_head = dict(enlarge = 0,bbox_head = dict(num_classes = 1)))

# log_config = dict(
#     interval = 50,
#     hooks = [
#         dict(type = 'TextLoggerHook'),
#         dict(type = 'WandbLoggerHook',
#              with_step = False,
#              init_kwargs = dict(project = 'dual_cervix_detection',
#                                 name = 'faster_rcnn_dual_r50_fpn_1x_dual',
#                                 tags = ['mmdetection', 'faster_rcnn_dual', 'r50', 'fpn', '1x', 'dual']))])


# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[16, 22])
runner = dict(type='EpochBasedRunner', max_epochs=24)
#!
optimizer = dict(lr=0.0025)

# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])


data_root = 'data/cervix/'
data = dict(
    samples_per_gpu = 2,
    workers_per_gpu = 2,
    train = dict(  # ! 具体查看dataset部分的cervix.py
        prim = 'acid',
        acid_ann_file = data_root + 'hsil_rereannos/train_acid.json',
        iodine_ann_file = data_root + 'hsil_rereannos/train_iodine.json',
 ),
    val = dict(
        prim = 'acid',
        acid_ann_file = data_root + 'hsil_rereannos/val_acid.json',
        iodine_ann_file = data_root + 'hsil_rereannos/val_iodine.json'),
    test = dict(
        prim = 'acid',
        acid_ann_file = data_root + 'hsil_rereannos/test_acid.json',
        iodine_ann_file = data_root + 'hsil_rereannos/test_iodine.json',
))









# 用于具有缩放的offset学习-dual_roi_head.py
# def _bbox_forward(self, acid_feats, iodine_feats, acid_rois, iodine_rois, img_metas):
#     """Box head forward function used in both training and testing."""
#     # acid
#     acid_bbox_feats = self.bbox_roi_extractor(acid_feats[:self.bbox_roi_extractor.num_inputs], acid_rois)
#     acid_iodine_rois = acid_rois.clone()
#     if self.enlarge:
#         acid_iodine_rois = torch.cat(
#             [acid_iodine_rois[:, 0, None], acid_iodine_rois[:, 1:3] - self.enlarge,
#              acid_iodine_rois[:, 3:] + self.enlarge], dim=1)
#     acid_iodine_bbox_feats = self.bbox_roi_extractor(iodine_feats[:self.bbox_roi_extractor.num_inputs],
#                                                      acid_iodine_rois)
#     acid_proposal_offsets = self._offset_forward(torch.cat([acid_bbox_feats, acid_iodine_bbox_feats], dim=1))
#     acid_proposal_offsets_scaled = []
#     acid_proposal_offsets_resc = []
#     for i in range(len(img_metas)):
#         cur_offsets_ori = acid_proposal_offsets[acid_rois[:, 0] == i]
#         cur_offsets = cur_offsets_ori[:, :2] * cur_offsets_ori[:, :2].new_tensor(img_metas[i]['pad_shape'][:2])
#         acid_proposal_offsets_scaled.append(cur_offsets)
#
#         acid_proposal_offsets_resc.append(torch.exp(cur_offsets_ori[:, 2:], out=None) * torch.cat(
#             [acid_rois[acid_rois[:, 0] == i][:, 3, None] - acid_rois[acid_rois[:, 0] == i][:, 1, None],
#              acid_rois[acid_rois[:, 0] == i][:, 4, None] - acid_rois[acid_rois[:, 0] == i][:, 2, None]], dim=1))
#
#     acid_proposal_offsets_scaled = torch.cat(acid_proposal_offsets_scaled)
#     acid_proposal_offsets_added = torch.cat(
#         [acid_proposal_offsets_scaled.new_zeros(acid_proposal_offsets_scaled.shape[0], 1), acid_proposal_offsets_scaled,
#          acid_proposal_offsets_scaled], dim=1)
#     acid_iodine_rois = acid_rois + acid_proposal_offsets_added
#
#     acid_iodine_rois = self.bbox_rescaless(acid_iodine_rois, scale_factor=torch.cat(acid_proposal_offsets_resc))
#     acid_iodine_bbox_feats = self.bbox_roi_extractor(iodine_feats[:self.bbox_roi_extractor.num_inputs],
#                                                      acid_iodine_rois)
#     acid_bbox_feats = self.fusion_feature(acid_bbox_feats, acid_iodine_bbox_feats)
#
#     # iodine
#     iodine_bbox_feats = self.bbox_roi_extractor(iodine_feats[:self.bbox_roi_extractor.num_inputs], iodine_rois)
#     iodine_acid_rois = acid_rois.clone()
#     if self.enlarge:
#         iodine_acid_rois = torch.cat(
#             [iodine_acid_rois[:, 0, None], iodine_acid_rois[:, 1:3] - self.enlarge,
#              iodine_acid_rois[:, 3:] + self.enlarge], dim=1)
#     iodine_acid_bbox_feats = self.bbox_roi_extractor(acid_feats[:self.bbox_roi_extractor.num_inputs], iodine_acid_rois)
#     iodine_proposal_offsets = self._offset_forward(torch.cat([iodine_bbox_feats, iodine_acid_bbox_feats], dim=1))
#     iodine_proposal_offsets_scaled = []
#     iodine_proposal_offsets_resc = []
#     for i in range(len(img_metas)):
#         cur_offsets_ori = iodine_proposal_offsets[iodine_rois[:, 0] == i]
#         cur_offsets = cur_offsets_ori[:, :2] * cur_offsets_ori[:, :2].new_tensor(img_metas[i]['pad_shape'][:2])
#         iodine_proposal_offsets_scaled.append(cur_offsets)
#         iodine_proposal_offsets_resc.append(torch.exp(cur_offsets_ori[:, 2:], out=None) * torch.cat(
#             [iodine_rois[iodine_rois[:, 0] == i][:, 3, None] - iodine_rois[iodine_rois[:, 0] == i][:, 1, None],
#              iodine_rois[iodine_rois[:, 0] == i][:, 4, None] - iodine_rois[iodine_rois[:, 0] == i][:, 2, None]], dim=1))
#     iodine_proposal_offsets_scaled = torch.cat(iodine_proposal_offsets_scaled)
#     iodine_proposal_offsets_added = torch.cat(
#         [iodine_proposal_offsets_scaled.new_zeros((iodine_proposal_offsets_scaled.shape[0], 1)),
#          iodine_proposal_offsets_scaled,
#          iodine_proposal_offsets_scaled], dim=1)
#     iodine_acid_rois = iodine_rois + iodine_proposal_offsets_added
#
#     iodine_acid_rois = self.bbox_rescaless(iodine_acid_rois, scale_factor=torch.cat(iodine_proposal_offsets_resc))
#     iodine_acid_bbox_feats = self.bbox_roi_extractor(iodine_feats[:self.bbox_roi_extractor.num_inputs],
#                                                      iodine_acid_rois)
#     iodine_bbox_feats = self.fusion_feature(iodine_bbox_feats, iodine_acid_bbox_feats)
#
#     if self.with_shared_head:
#         acid_bbox_feats = self.shared_head(acid_bbox_feats)
#         iodine_bbox_feats = self.shared_head(iodine_bbox_feats)
#     acid_cls_score, iodine_cls_score, acid_bbox_pred, iodine_bbox_pred = self.bbox_head(acid_bbox_feats,
#                                                                                         iodine_bbox_feats)
#
#     bbox_results = dict(acid_cls_score=acid_cls_score, acid_bbox_pred=acid_bbox_pred, acid_bbox_feats=acid_bbox_feats,
#                         acid_proposal_offsets=acid_proposal_offsets,
#                         iodine_cls_score=iodine_cls_score, iodine_bbox_pred=iodine_bbox_pred,
#                         iodine_bbox_feats=iodine_bbox_feats,
#                         iodine_proposal_offsets=iodine_proposal_offsets)
#     return bbox_results

