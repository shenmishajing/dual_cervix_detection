_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/dual_cervix/hsil/acid/acid_base.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

# model settings
model = dict(
    rpn_head = dict(type = 'RPNATSSAssignerHead'),
    roi_head = dict(
        type = 'ATSSAssignerRoIHead',
        bbox_head = dict(num_classes = 1)),
    train_cfg = dict(
        rpn = dict(assigner = dict(_delete_ = True, type = 'ATSSAssigner', topk = 9)),
        rcnn = dict(assigner = dict(_delete_ = True, type = 'ATSSAssigner', topk = 9))))

log_config = dict(
    interval = 50,
    hooks = [
        dict(type = 'TextLoggerHook'),
        dict(type = 'WandbLoggerHook',
             with_step = False,
             init_kwargs = dict(project = 'dual_cervix_detection',
                                name = 'faster_rcnn_r50_fpn_atss_assigner_1x_acid',
                                tags = ['mmdetection', 'faster_rcnn', 'r50', 'fpn', '1x', 'acid']))])
