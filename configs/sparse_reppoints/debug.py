_base_ = [
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
model = dict(
    type="SparseRepPointsDetector",
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5),
    bbox_head = dict(
        type="SparseRepPointsHead",
        num_classes=80,
        in_channels=256,
        feat_channels=256,
        point_feat_channels=256,
        num_points=9,
        top_k=40,
        stacked_linears=3,
        output_strides=[8, 16, 32, 64, 128],
        loss_obj=dict(type='CrossEntropyLoss', use_sigmoid=True),
        loss_bbox=dict(type='GIoULoss'),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
    )
)

# training and testing settings
train_cfg = dict(
    assigner = dict(
        type='HungarianAssigner',
        cls_cost=dict(type='ClassificationCost', weight=1.),
        reg_cost=dict(type='BBoxL1Cost', weight=1.0),
        iou_cost=dict(type='IoUCost', iou_mode='giou', weight=1.0)),
    allowed_border=-1,     
    pos_weight=-1,
)

test_cfg = dict(
    min_bbox_size=0, 
    score_thr=0.05,
    nms=dict(type='nms', iou_threshold=0.5),
    max_per_img=100
)

optimizer = dict(lr=0.01)