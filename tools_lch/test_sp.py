from mmdet.models import build_head
import torch
import os 

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "8"
    bbox_head=dict(
        type='SparseRepPointsHead',
        num_classes=80,
        in_channels=256,
        feat_channels=256,
        point_feat_channels=256,
        num_points=9,
        top_k=10,
        stacked_linears=3,
        output_strides=[8, 16, 32, 64, 128],
        loss_obj=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=True),
        loss_bbox=dict(type='SmoothL1Loss'),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        train_cfg=dict(
            pos_weight=-1,
            assigner=dict(type='HungarianAssigner'))
        )
    head = build_head(bbox_head)

    x = [
        torch.normal(2, 3, size=[2, 256, 100, 136]).cuda(),
        torch.normal(2, 3, size=[2, 256, 50, 68]).cuda(),
        torch.normal(2, 3, size=[2, 256, 25, 34]).cuda(),
        torch.normal(2, 3, size=[2, 256, 13, 17]).cuda(),
        torch.normal(2, 3, size=[2, 256, 7, 9]).cuda()
    ]
  
    gt_bboxes = [
        torch.tensor([[300., 400, 500, 600],[450, 200, 500, 500]]).cuda(),
        torch.tensor([[100., 20, 400, 230]]).cuda()
        ]
    gt_labels = [
        torch.tensor([3,4]).cuda(),
        torch.tensor([1]).cuda()
    ]
    img_metas = [
        dict(pad_shape=(800, 1344, 3),batch_input_shape=(800, 1344), img_shape= (770, 1333, 3)),
        dict(pad_shape=(800, 1344, 3),batch_input_shape=(800, 1344), img_shape= (770, 1333, 3))
    ]
    
    head.init_weights()
    obj_pred, topn_bbox, topn_cls, topn_idx = head.forward(x)
    # print(obj_pred[0].device)
    # print(topn_bbox[0].device)
    # print(topn_cls[0].device)
    # print(topn_idx[0].device)
    # print("obj(forward)",[obj.shape for obj in obj_pred])
    loss_dict = head.loss(obj_pred, topn_bbox, topn_cls, topn_idx, gt_bboxes, gt_labels, img_metas)
    print(loss_dict)