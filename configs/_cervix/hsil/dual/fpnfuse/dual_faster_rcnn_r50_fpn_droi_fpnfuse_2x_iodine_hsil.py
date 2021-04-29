_base_ = [
    '../dual_faster_rcnn_r50_fpn_droi_2x_iodine_hsil.py'
]

model = dict(
    roi_head=dict(
        fpn_fuser_cfg=dict(
            roi_feat_size=7,
            num_levels=4,
        ),
     ))