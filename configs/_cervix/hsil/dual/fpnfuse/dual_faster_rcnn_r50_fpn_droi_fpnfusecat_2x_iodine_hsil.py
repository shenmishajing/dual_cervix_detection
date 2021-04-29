_base_ = [
    './dual_faster_rcnn_r50_fpn_droi_fpnfuse_2x_iodine_hsil.py'
]

model = dict(
    roi_head=dict(
        fpn_fuser_cfg=dict(
            in_channels=512, #! prim + aux = 256 + 256
            out_channels=256,
            fuse_type="cat"
        ),
     ))