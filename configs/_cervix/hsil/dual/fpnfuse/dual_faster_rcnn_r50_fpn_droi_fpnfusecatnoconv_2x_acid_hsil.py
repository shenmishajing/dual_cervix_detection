_base_ = [
    './dual_faster_rcnn_r50_fpn_droi_fpnfusecat_2x_acid_hsil.py'
]

model = dict(
    roi_head=dict(
        fpn_fuser_cfg=dict(
            #! 不加conv之后，输出的通道数为512
            in_channels=512, #! prim + aux = 256 + 256
            out_channels=512,
            with_conv=False,
        ),
        offset_cfg=dict(
            in_channels=512),
     ))