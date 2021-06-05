_base_ = [
    './retinanet_r50_fpn_2x_acid_hsil.py',
]
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[10, 16])