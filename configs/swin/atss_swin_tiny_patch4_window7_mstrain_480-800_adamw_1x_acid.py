_base_ = [
    '../_base_/models/atss_swin_fpn.py',
    '../_base_/datasets/dual_cervix/hsil/acid/acid_base.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

model = dict(bbox_head = dict(num_classes = 1))

img_norm_cfg = dict(
    mean = [123.675, 116.28, 103.53], std = [58.395, 57.12, 57.375], to_rgb = True)

# augmentation strategy originates from DETR / Sparse RCNN
train_pipeline = [
    dict(type = 'LoadImageFromFile'),
    dict(type = 'LoadAnnotations', with_bbox = True),
    dict(type = 'RandomFlip', flip_ratio = 0.5),
    dict(type = 'AutoAugment',
         policies = [
             [
                 dict(type = 'Resize',
                      img_scale = [(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                                   (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                                   (736, 1333), (768, 1333), (800, 1333)],
                      multiscale_mode = 'value',
                      keep_ratio = True)
             ],
             [
                 dict(type = 'Resize',
                      img_scale = [(400, 1333), (500, 1333), (600, 1333)],
                      multiscale_mode = 'value',
                      keep_ratio = True),
                 dict(type = 'RandomCrop',
                      crop_type = 'absolute_range',
                      crop_size = (384, 600),
                      allow_negative_crop = True),
                 dict(type = 'Resize',
                      img_scale = [(480, 1333), (512, 1333), (544, 1333),
                                   (576, 1333), (608, 1333), (640, 1333),
                                   (672, 1333), (704, 1333), (736, 1333),
                                   (768, 1333), (800, 1333)],
                      multiscale_mode = 'value',
                      override = True,
                      keep_ratio = True)
             ]
         ]),
    dict(type = 'Normalize', **img_norm_cfg),
    dict(type = 'Pad', size_divisor = 32),
    dict(type = 'DefaultFormatBundle'),
    dict(type = 'Collect', keys = ['img', 'gt_bboxes', 'gt_labels']),
]
data = dict(
    samples_per_gpu = 1,
    workers_per_gpu = 1,
    train = dict(pipeline = train_pipeline))

optimizer = dict(_delete_ = True, type = 'AdamW', lr = 0.00005, betas = (0.9, 0.999), weight_decay = 0.05,
                 paramwise_cfg = dict(custom_keys = {'absolute_pos_embed': dict(decay_mult = 0.),
                                                     'relative_position_bias_table': dict(decay_mult = 0.),
                                                     'norm': dict(decay_mult = 0.)}))
lr_config = dict(step = [8, 11])
runner = dict(type = 'EpochBasedRunner', max_epochs = 12)
