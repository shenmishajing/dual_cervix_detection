_base_ = [
    './dual_dualdet_faster_rcnn_r50_fpn_droi_2x_acid_hsil.py'
]


model = dict(
    roi_head=dict(
        attention_cfg=None,
        offset_cfg=None,
        bridge_bbox_droi_extractor=None,
        prim_bbox_head=dict(
            in_channels=256 
            )
        ))