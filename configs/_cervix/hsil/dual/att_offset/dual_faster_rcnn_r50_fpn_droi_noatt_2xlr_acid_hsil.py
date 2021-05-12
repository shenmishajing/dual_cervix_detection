_base_ = [
    './dual_faster_rcnn_r50_fpn_droi_2x_acid_hsil.py'
]


model = dict(
    roi_head=dict(
        attention_cfg=None))
        
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[10, 16])