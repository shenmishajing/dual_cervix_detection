dataset_type = 'SingleCervixDataset'
data_root = 'data/cervix/'
classes = ("hsil", )
img_type = "acid"
img_norm_cfg = dict(
    mean=[143.463 , 102.0255,  97.5375], std=[42.636 , 39.9075, 40.7235], to_rgb=True)
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