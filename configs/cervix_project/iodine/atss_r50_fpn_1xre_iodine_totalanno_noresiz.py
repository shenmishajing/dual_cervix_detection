_base_ = [
    '../../_base_/models/atss_r50_fpn.py',
    '../../_base_/schedules/schedule_1x.py',
    '../../_base_/default_runtime.py',
    './iodine_base.py'
]

model = dict(bbox_head=dict(num_classes=2))

optimizer = dict(type='SGD', lr=0.00125, momentum=0.9, weight_decay=0.0001)


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


data_root = 'data/cervix_project/detection/'
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        pipeline=train_pipeline,
        ann_file=data_root + 'annos_frompaper/train_iodine.json'),
    val=dict(
        pipeline=test_pipeline,
        ann_file=data_root + 'annos_frompaper/val_iodine.json'
        ),
    test=dict(
        pipeline=test_pipeline,
        ann_file=data_root + 'annos_frompaper/test_iodine.json'
        ))

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11, 15])
runner = dict(type='EpochBasedRunner', max_epochs=24)






