_base_ = [
    './dual_dualdet_faster_rcnn_r50_fpn_droi_2x_acid_hsil.py'
]


model = dict(
    roi_head=dict(
        attention_cfg=None))

data = dict(
    samples_per_gpu=1
)