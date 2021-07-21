_base_ = [
    '../_base_/datasets/dual_cervix/hsil/dual/dual_base.py', '../_base_/models/atss_r50_fpn.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
model = dict(bbox_head = dict(num_classes = 1))
# optimizer
optimizer = dict(type = 'SGD', lr = 0.01, momentum = 0.9, weight_decay = 0.0001)
