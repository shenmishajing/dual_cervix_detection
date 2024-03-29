#! 针对宫颈专门写的数据模块
dataset_type = 'DualCervixDataset'
data_root = 'data/cervix/'
classes = ("hsil",)
img_norm_cfg = dict(
    acid_mean = [143.9475, 102.153, 97.971], acid_std = [42.075, 39.2445, 40.086],
    iodine_mean = [134.4105, 89.1735, 63.24], iodine_std = [60.945, 60.3585, 55.9215],
    to_rgb = True)
train_pipeline = [
    dict(type = 'LoadDualCervixImageFromFile'),  # ! 同时加载醋酸和碘图像
    dict(type = 'LoadDualCervixAnnotations', with_bbox = True),  # ! 同时加载醋酸的框和碘的框
    dict(type = 'Resize', img_scale = (1333, 800), keep_ratio = True),
    dict(type = 'RandomFlip', flip_ratio = 0.5),
    dict(type = 'DualCervixNormalize', **img_norm_cfg),  # ! 分别标准化，根据各自的均值和标准差
    dict(type = 'Pad', size_divisor = 32),
    dict(type = 'DualCervixDefaultFormatBundle'),  # ! 格式标准化
    dict(type = 'Collect', keys = ['acid_img', 'iodine_img', 'acid_gt_bboxes', 'iodine_gt_bboxes', 'acid_gt_labels', 'iodine_gt_labels']),
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
    samples_per_gpu = 2,
    workers_per_gpu = 2,
    train = dict(  # ! 具体查看dataset部分的cervix.py
        type = dataset_type,
        prim = 'acid',
        classes = classes,
        acid_ann_file = data_root + 'hsil_rereannos/train_acid.json',
        iodine_ann_file = data_root + 'hsil_rereannos/train_iodine.json',
        img_prefix = data_root + 'img/',
        pipeline = train_pipeline),
    val = dict(
        type = dataset_type,
        prim = 'acid',
        classes = classes,
        acid_ann_file = data_root + 'hsil_rereannos/val_acid.json',
        iodine_ann_file = data_root + 'hsil_rereannos/val_iodine.json',
        img_prefix = data_root + 'img/',
        pipeline = test_pipeline),
    test = dict(
        type = dataset_type,
        prim = 'acid',
        classes = classes,
        acid_ann_file = data_root + 'hsil_rereannos/test_acid.json',
        iodine_ann_file = data_root + 'hsil_rereannos/test_iodine.json',
        img_prefix = data_root + 'img/',
        pipeline = test_pipeline))
evaluation = dict(interval = 1, metric = 'bbox')
