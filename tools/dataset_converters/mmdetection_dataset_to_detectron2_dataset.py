import os
import json
import random
import shutil
import pickle
from collections import defaultdict

import numpy as np


def main():
    # path
    # output path
    output_root = '/home/zhengwenhao/Project/dual_cervix_detection/mmdetection/data/cervix/CLDNet_DataSet'
    output_image_path = os.path.join(output_root, 'img_with_norm')
    output_ann_path = os.path.join(output_root, 'anns.pkl')
    output_txt_path = os.path.join(output_root, 'DataSet')

    # image path and ann path
    original_img_path = '/data/lxc/Cervix/paper_data/imgs'
    norm_anns = pickle.load(open('/data/ctt/data/CervixData/paper_data/norm.pkl', 'rb'))
    ann_dir = '/home/zhengwenhao/Project/dual_cervix_detection/mmdetection/data/cervix/hsil_norm'

    # annotations
    categorie_name_to_label = {
        'lsil': 1,
        'hsil': 2
    }
    anns = norm_anns.copy()
    for ann_name in os.listdir(ann_dir):
        cur_ann_file = os.path.join(ann_dir, ann_name)
        if not os.path.isfile(cur_ann_file) or not ann_name.endswith('.json'):
            continue
        ann_dict = json.load(open(cur_ann_file))
        categorie_id_to_label = {c['id']: categorie_name_to_label[c['name']] for c in ann_dict['categories']}
        image_id_to_anns = defaultdict(list)
        for a in ann_dict['annotations']:
            image_id_to_anns[a['image_id']].append(a)
        for img in ann_dict['images']:
            cur_ann = {
                'shape': [img['height'], img['width']],
                'annos': [],
                'image_label': 0
            }
            for a in image_id_to_anns[img['id']]:
                bbox = a['bbox']
                cur_ann['annos'].append({
                    'bbox': [*bbox[:2], *[x + d for x, d in zip(bbox[:2], bbox[2:])]],
                    'segm': np.zeros((0, 2), dtype = np.int64),
                    'label': categorie_id_to_label[a['category_id']]
                })
            if len(cur_ann['annos']):
                cur_ann['image_label'] = 1
            anns[img['file_name'].removesuffix('.jpg')] = cur_ann
    pickle.dump(anns, open(output_ann_path, 'wb'))

    # texts
    img_names = [l.strip() for l in open(os.path.join(output_txt_path, 'train_original.txt')).readlines()]
    img_names = [n for n in img_names if n + '_2' in norm_anns]
    ann_dict = json.load(open(os.path.join(ann_dir, 'train_acid.json')))
    if len(img_names) > len(ann_dict['images']):
        img_names = img_names[:len(ann_dict['images'])]
    elif len(img_names) < len(ann_dict['images']):
        all_img_names = list(set([n[:-2] for n in norm_anns]))
        random.shuffle(all_img_names)
        i = 0
        while len(img_names) < len(ann_dict['images']):
            if all_img_names[i] not in img_names:
                img_names.append(all_img_names[i])
            i += 1
    for ann_name in os.listdir(ann_dir):
        cur_ann_file = os.path.join(ann_dir, ann_name)
        if not os.path.isfile(cur_ann_file) or not ann_name.endswith('.json') or 'acid' not in ann_name:
            continue
        ann_dict = json.load(open(cur_ann_file))
        cur_txt_list = [name['file_name'][:-6] for name in ann_dict['images']]
        if 'train' in ann_name:
            f = open(os.path.join(output_txt_path, ann_name.removesuffix('_acid.json') + '_norm.txt'), 'w')
            f.write('\n'.join(cur_txt_list + img_names))
            f.close()
        f = open(os.path.join(output_txt_path, ann_name.removesuffix('_acid.json') + '.txt'), 'w')
        f.write('\n'.join(cur_txt_list))
        f.close()

    # images
    for cur_name in [2, 3]:
        for img_name in img_names:
            img_name = img_name + '_' + str(cur_name)
            if not os.path.exists(os.path.join(output_image_path, img_name + '.jpg')):
                print(img_name)
                shutil.copy2(os.path.join(original_img_path, img_name + '.jpg'), os.path.join(output_image_path, img_name + '.jpg'))


if __name__ == '__main__':
    main()
