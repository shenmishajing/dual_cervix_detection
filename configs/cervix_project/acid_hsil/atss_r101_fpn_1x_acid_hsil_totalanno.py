_base_ = [
    '../../_base_/models/atss_r50_fpn.py',
    '../../_base_/schedules/schedule_1x.py',
    '../../_base_/default_runtime.py',
    './acid_hsil_base.py'
]

model = dict(backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')),
        bbox_head=dict(num_classes=1))

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







