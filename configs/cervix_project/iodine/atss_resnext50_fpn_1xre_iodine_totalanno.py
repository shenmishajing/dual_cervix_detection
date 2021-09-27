_base_ = [
    '../../_base_/models/atss_r50_fpn.py',
    '../../_base_/schedules/schedule_1x.py',
    '../../_base_/default_runtime.py',
    './iodine_base.py'
]

model = dict(type='ATSS',
        backbone=dict(
        type='ResNeXt',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
             bbox_head=dict(num_classes=2))

optimizer = dict(type='SGD', lr=0.00125, momentum=0.9, weight_decay=0.0001)

data_root = 'data/cervix_project/detection/'
data = dict(
    train=dict(
        ann_file=data_root + 'annos_frompaper/train_iodine.json'),
    val=dict(
        ann_file=data_root + 'annos_frompaper/val_iodine.json'
        ),
    test=dict(
        ann_file=data_root + 'annos_frompaper/test_iodine.json'
        ))

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11, 15])
runner = dict(type='EpochBasedRunner', max_epochs=24)






