_base_ = [
    '../att_offset/dual_faster_rcnn_r50_fpn_droi_2x_acid_hsil.py'
]

model = dict(
    roi_head=dict(
        fpn_fuser_cfg=dict(
            roi_feat_size=7,
            num_levels=4,
            naive_fuse=False
        ),
     ))

# data_root = 'data/cervix/'
# data = dict(
#     samples_per_gpu=1,
#     workers_per_gpu=2,
#     train=dict(
#         acid_ann_file=data_root + 'hsil_annos/debug_acid.json',
#         iodine_ann_file=data_root + 'hsil_annos/debug_iodine.json',
#         ),
#     val=dict(
#         acid_ann_file=data_root + 'hsil_annos/debug_acid.json',
#         iodine_ann_file=data_root + 'hsil_annos/debug_iodine.json',),
#     test=dict(
#         acid_ann_file=data_root + 'hsil_annos/debug_acid.json',
#         iodine_ann_file=data_root + 'hsil_annos/debug_iodine.json',)
# )