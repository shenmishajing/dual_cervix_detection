_base_ = [
    './dual_dualdet_faster_rcnn_r50_fpn_droi_2x_iodine_hsil.py'
]

model = dict(
    roi_head=dict(
        offset_cfg=None,
        bridge_bbox_droi_extractor=None,
        prim_bbox_head=dict(
            in_channels=256 
            )))

data = dict(
    samples_per_gpu=1
)