_base_ = './retinanet_r50_fpn_1x_iodine.py'
# learning policy
lr_config = dict(step=[16, 22])
total_epochs = 24
data = dict(samples_per_gpu=4)

