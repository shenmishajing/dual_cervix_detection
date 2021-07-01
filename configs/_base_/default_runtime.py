checkpoint_config = dict(interval = 1)
# yapf:disable
log_config = dict(
    interval = 50,
    hooks = [
        dict(type = 'TextLoggerHook'),
        dict(type = 'WandbLoggerHook',
             with_step = False,
             init_kwargs = dict(project = 'your-project-name',
                                name = 'default-project-name',
                                tags = ['mmdetection']))])
# yapf:enable
custom_hooks = [dict(type = 'NumClassCheckHook')]

dist_params = dict(backend = 'nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
