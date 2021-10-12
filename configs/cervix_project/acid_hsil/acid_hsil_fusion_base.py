dataset_type = 'SingleDCervixDataset'
data_root = 'data/cervix_project/detection/'
classes = ("hsil", )
img_type = "acid"
fusion = True #表示输入数据中需要融合多种数据，test.py中需要用到
img_norm_cfg = dict(
    acid_mean = [144.228, 102.2805, 98.124], acid_std = [42.024, 39.2445, 40.086],
    iodine_mean = [134.4615, 89.148, 63.189], iodine_std = [60.996, 60.333, 55.8705],
    to_rgb = True)


train_pipeline = [
    dict(type = 'LoadDualCervixImageFromFile'),  # ! 同时加载醋酸和碘图像
    dict(type = 'LoadDualCervixAnnotations', with_bbox = True),  # ! 同时加载醋酸的框和碘的框
    dict(type = 'Resize', img_scale = (1333, 800), keep_ratio = True),
    dict(type = 'RandomFlip', flip_ratio = 0.5),
    dict(type = 'DualCervixNormalize', **img_norm_cfg),  # ! 分别标准化，根据各自的均值和标准差
    dict(type = 'Pad', size_divisor = 32),
    dict(type = 'DualCervixDefaultFormatBundle'),  # ! 格式标准化
    dict(type = 'Collect', keys = ['acid_img', 'iodine_img', 'acid_gt_bboxes', 'acid_gt_labels']),
    # ! 将需要的东西传到检测模型中
]
test_pipeline = [
    dict(type = 'LoadDualCervixImageFromFile'),
    dict(
        type = 'MultiScaleFlipAug',
        img_scale = (1333, 800),
        flip = False,
        transforms = [
            dict(type = 'Resize', keep_ratio = True),
            dict(type = 'RandomFlip'),
            dict(type = 'DualCervixNormalize', **img_norm_cfg),
            dict(type = 'Pad', size_divisor = 32),
            dict(type = 'ImageToTensor', keys = ['acid_img', 'iodine_img']),
            dict(type = 'Collect', keys = ['acid_img', 'iodine_img']),
        ])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        img_type=img_type,
        fusion=fusion,
        classes=classes,
        acid_ann_file=data_root + 'annos/train_acid.json',
        iodine_ann_file=data_root + 'annos/train_acid.json',
        img_prefix=data_root + 'img/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        img_type=img_type,
        fusion=fusion,
        classes=classes,
        acid_ann_file=data_root + 'annos/val_acid.json',
        iodine_ann_file=data_root + 'annos/val_acid.json',
        img_prefix=data_root + 'img/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        img_type=img_type,
        fusion=fusion,
        classes=classes,
        acid_ann_file=data_root + 'annos/test_acid.json',
        iodine_ann_file=data_root + 'annos/test_acid.json',
        img_prefix=data_root + 'img/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')
