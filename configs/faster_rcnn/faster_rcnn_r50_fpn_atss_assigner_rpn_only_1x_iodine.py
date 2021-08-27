_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/dual_cervix/hsil/iodine/iodine_base.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

# model settings
model = dict(
    rpn_head = dict(type = 'RPNATSSAssignerHead'),
    roi_head = dict(bbox_head = dict(num_classes = 1)),
    train_cfg = dict(rpn = dict(assigner = dict(_delete_ = True, type = 'ATSSAssigner', topk = 9))))
