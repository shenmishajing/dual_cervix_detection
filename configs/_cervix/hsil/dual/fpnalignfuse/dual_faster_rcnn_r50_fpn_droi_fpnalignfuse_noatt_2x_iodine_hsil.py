_base_ = [
    './dual_faster_rcnn_r50_fpn_droi_fpnalignfuse_2x_iodine_hsil.py'
]

model = dict(
    roi_head=dict(
        attention_cfg=None
     ))