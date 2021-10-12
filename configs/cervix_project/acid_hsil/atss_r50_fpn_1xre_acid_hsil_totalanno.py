_base_ = [
    '../../_base_/models/atss_r50_fpn.py',
    '../../_base_/schedules/schedule_1x.py',
    '../../_base_/default_runtime.py',
    './acid_hsil_base.py'
]

model = dict(bbox_head=dict(num_classes=1))

optimizer = dict(type='SGD', lr=0.00125, momentum=0.9, weight_decay=0.0001)

data_root = 'data/cervix_project/detection/'
data = dict(
    train=dict(
        ann_file=data_root + 'annos_hsil_frompaper/train_acid.json'),
    val=dict(
        ann_file=data_root + 'annos_hsil_frompaper/val_acid.json'
        ),
    test=dict(
        ann_file=data_root + 'annos_hsil_frompaper/test_acid.json'
        ))

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11, 15])
runner = dict(type='EpochBasedRunner', max_epochs=24)






