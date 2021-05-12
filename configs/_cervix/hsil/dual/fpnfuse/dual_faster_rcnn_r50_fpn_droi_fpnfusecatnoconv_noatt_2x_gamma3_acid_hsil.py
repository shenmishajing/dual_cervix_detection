_base_ = [
    './dual_faster_rcnn_r50_fpn_droi_fpnfusecat_noatt_2x_gamma3_acid_hsil.py'
]

model = dict(
    roi_head=dict(
        fpn_fuser_cfg=dict(
            in_channels=512,
            out_channels=512,
            with_conv=False),
        offset_cfg=dict(
            in_channels=512),
     )
)