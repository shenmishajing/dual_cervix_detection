_base_ = [
    '../_base_/datasets/dual_cervix/hsil/dual/dual_base_iodineonly.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
]

# model settings
model = dict(
    type = 'FasterRCNNIodine',
    prim = 'iodine',  # 'acid', 'iodine' or None
    backbone = dict(
        type = 'ResNet',
        depth = 50,
        num_stages = 4,
        out_indices = (0, 1, 2, 3),
        frozen_stages = 1,
        norm_cfg = dict(type = 'BN', requires_grad = True),
        norm_eval = True,
        style = 'pytorch',
        init_cfg = dict(type = 'Pretrained', checkpoint = 'torchvision://resnet50')),
    neck = dict(
        type = 'FPN',
        in_channels = [256, 512, 1024, 2048],
        out_channels = 256,
        num_outs = 5),
    rpn_head_iodine = dict(
        type = 'RPNHead',
        in_channels = 256,
        feat_channels = 256,
        anchor_generator = dict(
            type = 'AnchorGenerator',
            scales = [8],
            ratios = [0.5, 1.0, 2.0],
            strides = [4, 8, 16, 32, 64]),
        bbox_coder = dict(
            type = 'DeltaXYWHBBoxCoder',
            target_means = [.0, .0, .0, .0],
            target_stds = [1.0, 1.0, 1.0, 1.0]),
        loss_cls = dict(
            type = 'CrossEntropyLoss', use_sigmoid = True, loss_weight = 1.0),
        loss_bbox = dict(type = 'L1Loss', loss_weight = 1.0)),

    roi_head_iodine = dict(
        type = 'DualConsisProgFusjRoIHead',
        enlarge = 0,
        bbox_roi_extractor = dict(
            type = 'SingleRoIExtractor',
            roi_layer = dict(type = 'RoIAlign', output_size = 7, sampling_ratio = 0),
            out_channels = 256,
            featmap_strides = [4, 8, 16, 32]),
        bbox_head = dict(
            type = 'DualConsisLossbfcBBoxHead',
            in_channels = 256,
            fc_out_channels = 1024,
            roi_feat_size = 7,
            num_classes = 1,
            bbox_coder = dict(
                type = 'DeltaXYWHBBoxCoder',
                target_means = [0., 0., 0., 0.],
                target_stds = [0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic = False,
            loss_cls = dict(
                type = 'CrossEntropyLoss', use_sigmoid = False, loss_weight = 1.0),
            loss_bbox = dict(type = 'L1Loss', loss_weight = 1.0),
            loss_conscore = dict(type = 'L1Loss', loss_weight = 0.5),
            #loss_confeat = dict(type='CosinDis', loss_weight=1.0),
            loss_confeat_tot = dict(type = 'L2Loss', loss_weight = 0.5),
            #loss_progcls=dict(type='L1Loss', loss_weight=1.0),
            #loss_progbox=dict(type='L1Loss', loss_weight=1.0)
        )),
    # model training and testing settings
    train_cfg = dict(
        rpn = dict(
            assigner = dict(
                type = 'MaxIoUAssigner',
                pos_iou_thr = 0.7,
                neg_iou_thr = 0.3,
                min_pos_iou = 0.3,
                match_low_quality = True,
                ignore_iof_thr = -1),
            sampler = dict(
                type = 'RandomSampler',
                num = 256,
                pos_fraction = 0.5,
                neg_pos_ub = -1,
                add_gt_as_proposals = False),
            allowed_border = -1,
            pos_weight = -1,
            debug = False),
        rpn_proposal = dict(
            nms_pre = 2000,
            max_per_img = 1000,
            nms = dict(type = 'nms', iou_threshold = 0.7),
            min_bbox_size = 0),
        rcnn = dict(
            assigner = dict(
                type = 'MaxIoUAssigner',
                pos_iou_thr = 0.5,
                neg_iou_thr = 0.5,
                min_pos_iou = 0.5,
                match_low_quality = False,
                ignore_iof_thr = -1),
            sampler = dict(
                type = 'RandomSampler',
                num = 512,
                pos_fraction = 0.25,
                neg_pos_ub = -1,
                add_gt_as_proposals = True),
            pos_weight = -1,
            debug = False)),
    test_cfg = dict(
        rpn = dict(
            nms_pre = 1000,
            max_per_img = 1000,
            nms = dict(type = 'nms', iou_threshold = 0.7),
            min_bbox_size = 0),
        rcnn = dict(
            score_thr = 0.05,
            nms = dict(type = 'nms', iou_threshold = 0.5),
            max_per_img = 100)
        # soft-nms is also supported for rcnn testing
        # e.g., nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05)
    ))





# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[11, 16])
runner = dict(type='EpochBasedRunner', max_epochs=24)
#!
optimizer = dict(lr=0.0025)

# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])


