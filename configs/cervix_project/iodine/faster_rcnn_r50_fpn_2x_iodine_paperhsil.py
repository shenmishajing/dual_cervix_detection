_base_ = [
    '../../_base_/models/faster_rcnn_r50_fpn.py',
    '../../_base_/schedules/schedule_2x.py',
    '../../_base_/default_runtime.py',
    './iodine_base.py'
]

model = dict(roi_head=dict(bbox_head=dict(num_classes=1)))

optimizer = dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001)


classes = ("hsil",)

data_root = 'data/cervix/data_split/'
data = dict(
    train=dict(
        classes=classes,
        ann_file=data_root + 'annos/train_iodine.json'),
    val=dict(
        classes=classes,
        ann_file=data_root + 'annos/val_iodine.json'
        ),
    test=dict(
        classes=classes,
        ann_file=data_root + 'annos/test_iodine.json'
        ))









