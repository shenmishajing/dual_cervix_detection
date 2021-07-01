_base_ = [
    '../../../_base_/models/faster_rcnn_r50_caffe_c4.py',
    './acid_base.py',
]

model = dict(
    type='TridentFasterRCNN',
    pretrained='open-mmlab://detectron2/resnet50_caffe',
    backbone=dict(
        type='TridentResNet',
        trident_dilations=(1, 2, 3),
        num_branch=3,
        test_branch_idx=1),
    roi_head=dict(type='TridentRoIHead', num_branch=3, test_branch_idx=1,
        bbox_head=dict(
                type='BBoxHead',
                with_avg_pool=True,
                roi_feat_size=7,
                in_channels=2048,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]))))

train_cfg = dict(
    rpn_proposal=dict(nms_post=500, max_num=500),
    rcnn=dict(
        sampler=dict(num=128, pos_fraction=0.5, add_gt_as_proposals=False)))

