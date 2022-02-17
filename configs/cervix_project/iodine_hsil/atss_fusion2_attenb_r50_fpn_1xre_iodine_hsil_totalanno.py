_base_ = [
    '../../_base_/models/atss_r50_fpn.py',
    '../../_base_/schedules/schedule_1x.py',
    '../../_base_/default_runtime.py',
    './iodine_hsil_fusion_base.py'
]

model = dict(type='ATSSFusion',
             bbox_head=dict(
        type='ATSSFusionAttenbHead',
        num_classes=1,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)))

optimizer = dict(type='SGD', lr=0.00125, momentum=0.9, weight_decay=0.0001)




data_root = 'data/cervix_project/detection/'
data = dict(
    train=dict(
        acid_ann_file = data_root + 'annos_hsil_frompaper2/train_acid.json',
        iodine_ann_file=data_root + 'annos_hsil_frompaper2/train_iodine.json'),
    val=dict(
        acid_ann_file=data_root + 'annos_hsil_frompaper2/val_acid.json',
        iodine_ann_file=data_root + 'annos_hsil_frompaper2/val_iodine.json'
        ),
    test=dict(
        acid_ann_file=data_root + 'annos_hsil_frompaper2/test_acid.json',
        iodine_ann_file=data_root + 'annos_hsil_frompaper2/test_iodine.json'
        ))

##config
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11, 15])
runner = dict(type='EpochBasedRunner', max_epochs=24)


# offset size 7
#offset 乘以shape
#shift之后的图size为0时用原图替代。





