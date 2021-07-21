_base_ = [
    '../_base_/datasets/dual_cervix/hsil/acid/acid_base.py', '../_base_/models/atss_dual_r50_fpn.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
model = dict(bbox_head = dict(num_classes = 1))
# optimizer
optimizer = dict(type = 'SGD', lr = 0.01, momentum = 0.9, weight_decay = 0.0001)

log_config = dict(
    interval = 50,
    hooks = [
        dict(type = 'TextLoggerHook'),
        dict(type = 'WandbLoggerHook',
             with_step = False,
             init_kwargs = dict(project = 'dual_cervix_detection',
                                name = 'atss_dual_r50_fpn_1x_acid',
                                tags = ['mmdetection', 'atss_dual', 'r50', '1x', 'acid']))])
