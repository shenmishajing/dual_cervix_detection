_base_ = [
    '../../_base_/models/faster_rcnn_r50_fpn.py',
    '../../_base_/schedules/schedule_2x.py',
    '../../_base_/default_runtime.py',
    './iodine_base.py'
]

model = dict(roi_head=dict(bbox_head=dict(num_classes=2)))

optimizer = dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001)

img_norm_cfg = dict(
    mean=[134.4615, 89.148, 63.189], std=[60.996, 60.333, 55.8705], to_rgb=True) # for iodine
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(900, 600), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(900, 600),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        pipeline=train_pipeline),
    val=dict(
        pipeline=test_pipeline),
    test=dict(
        pipeline=test_pipeline))








