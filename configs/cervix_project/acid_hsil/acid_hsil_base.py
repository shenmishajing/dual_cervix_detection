#dataset_type = 'CocoDataset'
dataset_type = 'SingleCervixDataset'
data_root = 'data/cervix_project/detection/'
classes = ("hsil", )
img_type = "acid"
# img_norm_cfg = dict(
#     mean=[134.4105, 89.1735, 63.24], std=[60.945, 60.3585, 55.9215], to_rgb=True)   #原数据统计而得，需要重新计算
# img_norm_cfg = dict(
#     mean=[134.4615, 89.148, 63.189], std=[60.996, 60.333, 55.8705], to_rgb=True) # for iodine
img_norm_cfg = dict(
    mean=[144.228, 102.2805, 98.124], std=[42.024, 39.2445, 40.086], to_rgb=True) # for acid

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
        ann_file=data_root + 'annos/train_acid.json',
        img_prefix=data_root + 'img/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        img_type=img_type,
        classes=classes,
        ann_file=data_root + 'annos/val_acid.json',
        img_prefix=data_root + 'img/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        img_type=img_type,
        classes=classes,
        ann_file=data_root + 'annos/test_acid.json',
        img_prefix=data_root + 'img/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')
