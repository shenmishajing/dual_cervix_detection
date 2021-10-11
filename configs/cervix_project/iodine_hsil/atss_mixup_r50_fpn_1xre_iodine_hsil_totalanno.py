_base_ = [
    '../../_base_/models/atss_r50_fpn.py',
    '../../_base_/schedules/schedule_1x.py',
    '../../_base_/default_runtime.py',
    './iodine_hsil_base.py'
]

model = dict(type='ATSSMixup',bbox_head=dict(num_classes=1))

optimizer = dict(type='SGD', lr=0.00125, momentum=0.9, weight_decay=0.0001)

data_root = 'data/cervix_project/detection/'
data = dict(
    train=dict(
        ann_file=data_root + 'annos_hsil_frompaper/train_iodine.json'),
    val=dict(
        ann_file=data_root + 'annos_hsil_frompaper/val_iodine.json'
        ),
    test=dict(
        ann_file=data_root + 'annos_hsil_frompaper/test_iodine.json'
        ))

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11, 15])
runner = dict(type='EpochBasedRunner', max_epochs=24)




# mixup mix_ratio = 0.5
