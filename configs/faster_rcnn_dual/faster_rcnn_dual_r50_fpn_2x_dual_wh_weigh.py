_base_ = [
    '../_base_/models/faster_rcnn_dual_r50_fpn.py',
    '../_base_/datasets/dual_cervix/hsil/dual/dual_base.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
]

# model settings
model = dict(prim = None,  # 'acid', 'iodine' or None
    roi_head = dict(enlarge = 0,bbox_head = dict(num_classes = 1)))

# log_config = dict(
#     interval = 50,
#     hooks = [
#         dict(type = 'TextLoggerHook'),
#         dict(type = 'WandbLoggerHook',
#              with_step = False,
#              init_kwargs = dict(project = 'dual_cervix_detection',
#                                 name = 'faster_rcnn_dual_r50_fpn_1x_dual',
#                                 tags = ['mmdetection', 'faster_rcnn_dual', 'r50', 'fpn', '1x', 'dual']))])


# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[16, 22])
runner = dict(type='EpochBasedRunner', max_epochs=24)

optimizer = dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001)


# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])


data_root = 'data/cervix/'
data = dict(
    samples_per_gpu = 2,
    workers_per_gpu = 2,
    train = dict(  # ! 閸忚渹缍嬮弻銉ф箙dataset闁劌鍨庨惃鍒rvix.py
        prim = 'acid',
        acid_ann_file = data_root + 'hsil_rereannos/train_acid.json',
        iodine_ann_file = data_root + 'hsil_rereannos/train_iodine.json',
 ),
    val = dict(
        prim = 'acid',
        acid_ann_file = data_root + 'hsil_rereannos/val_acid.json',
        iodine_ann_file = data_root + 'hsil_rereannos/val_iodine.json'),
    test = dict(
        prim = 'acid',
        acid_ann_file = data_root + 'hsil_rereannos/test_acid.json',
        iodine_ann_file = data_root + 'hsil_rereannos/test_iodine.json',
))











