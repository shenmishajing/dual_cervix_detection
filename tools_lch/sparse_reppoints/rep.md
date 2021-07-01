# sparce_reppoints_head

## forward
    - input
        - feats (list): [feat_level1, feat_level2, ...] 
    - output
        - objectness (list[tensor]): [obj_level1, obj_level2, ...], obj_level shape = [num_img, 1, H, W]
        - topn_bbox (list[tensor]): [bbox_level1, bbox_level2, ...], bbox_level shape = [num_img, topn, 4]
        - topn_cls (list[tensor]): [cls_level1, cls_level2, ...], cls_level shape = [num_img, topn, num_cls]
        - topn_idx (list[tensor]): [idx_level1, idx_level2, ...], idx_level shape = [num_img, topn], idx in [0, feat_h * feat_w - 1]

## loss
    - input 
        - objectness_pred (list[tensor]): [obj_level1, obj_level2, ...], obj_level shape = [num_img, 1, H, W]
        - bbox_pred (list[tensor]): [bbox_level1, bbox_level2, ...], bbox_level shape = [num_img, 4, topn]
        - cls_pred (list[tensor]): [cls_level1, cls_level2, ...], cls_level shape = [num_img, num_cls, topn]
        - topn_idx (list[tensor]): [idx_level1, idx_level2, ...], idx_level shape = [num_img, topn], idx in [0, feat_h * feat_w - 1]
        - gt_bboxes (list[tensor]): [img1_gt_bbox_tensor, img2_gt_bbox_tensor, ...]
        - gt_labels (list[tensor]): [img1_gt_labels_tensor, img2_gt_labels_tensor, ...]
        - img_metas (list[dict]): [img1_meta, img2_meta, ...]
        - gt_bboxes_ignore (None): None
    - output
        - loss_dict_all (dict): {objectness_loss, bbox_loss, cls_loss}


## _get_objectness_single
    - input
        - gt_bboxes (Tensor): shape = (num_gt, 4), 像素值坐标（变换之后的）
        - img_meta (dict):
            {   
                'filename': 'data/coco/train2017/000000251577.jpg',
                'ori_filename': '000000251577.jpg',
                'ori_shape': (289, 500, 3),
                'img_shape': (770, 1333, 3),
                'pad_shape': (800, 1344, 3),
                'scale_factor': array([2.666    , 2.6643598, 2.666    , 2.6643598], dtype=float32),
                'flip': False,
                'flip_direction': None, 
                'img_norm_cfg': 
                    {
                        'mean': array([123.675, 116.28 , 103.53 ], dtype=float32),
                        'std': array([58.395, 57.12 , 57.375], dtype=float32),
                        'to_rgb': True
                    },
                'batch_input_shape': (800, 1344)
            }
        - output_strides (list): [8, 16, 32, 64, 128]
        - objectness_shape (list): [(h1 ,w1), ..., (hn, wn)]
    - output
        - objectness_list (list): [objectness_level1, objectness_level2, ..., objectness_level5],
                    !level1_obj shape = (num_imgs, 1, h_level1, w_level1)


## _get_objectness
    - input
        - gt_bboxes_list (list): [img1_gt_bboxes_tensor, img2_gt_bboxes_tensor, ...] 
        - img_metas (list): [img1_metas, img2_metas, ...]
        - output_strides (list): [8, 16, 32, 64, 128]
    - output
        - objectness_list (list): [level1_obj, level2_obj, ...]

## _get_targets_single
    - input 

    - output


## get_targets
    - input

    - output

