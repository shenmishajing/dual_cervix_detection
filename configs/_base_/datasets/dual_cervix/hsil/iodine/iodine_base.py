_base_ = [
    '../../../../schedules/schedule_2x.py',
    '../../../../default_runtime.py',
]

log_config = dict(
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

dataset_type = 'SingleCervixDataset'
data_root = 'data/cervix/'
classes = ("hsil", )
img_type = "iodine"
img_norm_cfg = dict(
    mean=[133.2375,  87.8985,  60.9195], std=[61.149, 59.568, 54.876], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
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
        img_scale=(1333, 800),
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
        type=dataset_type,
        img_type=img_type,
        classes=classes,
        ann_file=data_root + 'hsil_rereannos/train_{}.json'.format(img_type),
        img_prefix=data_root + 'img/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        img_type=img_type,
        classes=classes,
        ann_file=data_root + 'hsil_rereannos/val_{}.json'.format(img_type),
        img_prefix=data_root + 'img/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        img_type=img_type,
        classes=classes,
        ann_file=data_root + 'hsil_rereannos/test_{}.json'.format(img_type),
        img_prefix=data_root + 'img/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')
optimizer = dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[16, 22])