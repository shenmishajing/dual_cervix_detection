from mmdet.datasets import build_dataset
dataset_type = 'CocoDataset'
data_root = 'data/cervix/'
classes = ("lsil", "hsil")
img_norm_cfg = dict(
    mean=[134.4105, 89.1735, 63.24], std=[60.945, 60.3585, 55.9215], to_rgb=True)
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
acid_cfg = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'annos/single/train_acid.json',
        img_prefix=data_root + 'img/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'annos/single/valid_acid.json',
        img_prefix=data_root + 'img/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'annos/single/valid_acid.json',
        img_prefix=data_root + 'img/',
        pipeline=test_pipeline))


iodine_cfg = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'annos/single/train_iodine.json',
        img_prefix=data_root + 'img/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'annos/single/valid_iodine.json',
        img_prefix=data_root + 'img/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'annos/single/valid_iodine.json',
        img_prefix=data_root + 'img/',
        pipeline=test_pipeline))

if __name__ == "__main__":
    acid = build_dataset(acid_cfg["train"])
    # iodine = build_dataset(iodine_cfg["val"])
    
    print(acid.pipeline)
    # for i, j in zip(acid, iodine):

    #     print(i)
    #     print(j)

    #     exit(-1)
    # pass