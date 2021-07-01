_base_ = [
    './dual_base_iodine.py'
]
model = dict(
    type='FasterPrimAuxDualDetector',
    pretrained='torchvision://resnet50',
    aug_acid=False,
    
    prim_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),

    aux_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    
    prim_neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),

    aux_neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),

    prim_rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    
    aux_rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),

    roi_head=dict(
        type='DualCervixDualDetPrimAuxRoiHead',
        attention_cfg=dict(
            in_channels=256, 
            out_channels=256,
            num_levels=5, 
            shared=False),

        offset_cfg=dict(
            in_channels=256, 
            out_channels=256, 
            roi_feat_area=7*7), #! squared roi_feat_size(in bbox_head)

        prim_bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),

        aux_bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),

        bridge_bbox_droi_extractor=dict(
            type='SingleDeformRoIExtractor',
            #! spatial_scale 会根据featmap_strides自动添加的，gamma不知道啥意思
            roi_layer=dict(type='DeformRoIPool', output_size=7, sampling_ratio=0, gamma=0.1),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32],
            finest_scale=56),

        prim_bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=512, #! 256 + 256
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=1,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0)),

        aux_bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256, 
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=1,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0))))


train_cfg = dict(
    rpn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.7,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            match_low_quality=True,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=256,
            pos_fraction=0.5,
            neg_pos_ub=-1,
            add_gt_as_proposals=False),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    rpn_proposal=dict(
        nms_across_levels=False,
        nms_pre=2000,
        nms_post=1000,
        max_num=1000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.5,
            match_low_quality=False,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=512,
            pos_fraction=0.25,
            neg_pos_ub=-1,
            add_gt_as_proposals=True),
        pos_weight=-1,
        debug=False))
test_cfg = dict(
    rpn=dict(
        nms_across_levels=False,
        nms_pre=1000,
        nms_post=1000,
        max_num=1000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100)
    # soft-nms is also supported for rcnn testing
    # e.g., nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05)
)
dataset_type = 'DualCervixDataset'
data_root = 'data/cervix/'
classes = ("hsil", )
img_norm_cfg = dict(
    acid_mean=[143.9475, 102.153, 97.971], acid_std=[42.075, 39.2445, 40.086],
    iodine_mean=[134.4105, 89.1735, 63.24], iodine_std=[60.945, 60.3585, 55.9215],
    to_rgb=True)
train_pipeline = [
    dict(type='LoadDualCervixImageFromFile'),
    dict(type='LoadDualCervixAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='DualCervixNormalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DualCervixDefaultFormatBundle'),
    dict(type='Collect', keys=['acid_img', 'iodine_img', 'acid_gt_bboxes', 'iodine_gt_bboxes', 'acid_gt_labels', 'iodine_gt_labels']),
]
test_pipeline = [
    dict(type='LoadDualCervixImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='DualCervixNormalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['acid_img', 'iodine_img']),
            dict(type='Collect', keys=['acid_img', 'iodine_img']),
        ])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        dual_det=True,
        classes=classes,
        acid_ann_file=data_root + 'hsil_annos/train_acid.json',
        iodine_ann_file=data_root + 'hsil_annos/train_iodine.json',
        img_prefix=data_root + 'img/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        dual_det=True,
        classes=classes,
        acid_ann_file=data_root + 'hsil_annos/debug_acid.json',
        iodine_ann_file=data_root + 'hsil_annos/debug_iodine.json',
        img_prefix=data_root + 'img/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        dual_det=True,
        classes=classes,
        acid_ann_file=data_root + 'hsil_annos/debug_acid.json',
        iodine_ann_file=data_root + 'hsil_annos/debug_iodine.json',
        img_prefix=data_root + 'img/',
        pipeline=test_pipeline))

log_config = dict(interval=2)
data = dict(
      samples_per_gpu=2,
)