from mmdet.apis import init_detector, inference_detector
import cv2
import numpy as np
import os
import random
from panopticapi.utils import rgb2id
from mmdet.core.evaluation.panoptic_utils import INSTANCE_OFFSET
import mmcv
from tqdm import tqdm

from xsma.generel_utils.tool_json import load_json, write_json

from IPython import embed
import json


def get_model(cfg, ckp):
    cfg = mmcv.Config.fromfile(cfg)

    cfg['model']['roi_head']['type'] = 'CascadeLastMaskRefineRoIHeadForinfer'

    cfg['model']['roi_head']['semantic_head']['pretrain'] = None
    cfg['model']['roi_head']['relationship_head']['pretrained_transformers'] = '/share/wangqixun/workspace/bs/tx_mm/code/model_dl/hfl/chinese-roberta-wwm-ext'
    cfg['model']['roi_head']['relationship_head']['cache_dir'] = './'

    cfg['model']['test_cfg']['rcnn']['score_thr'] = 0.3
    

    model = init_detector(cfg, ckp)
    return model


def get_tra_val_test_list():
    psg_tra_data_file = '/share/data/psg/dataset/for_participants/psg_train_val.json'
    psg_val_data_file = '/share/data/psg/dataset/for_participants/psg_val_test.json'
    psg_tra_data = load_json(psg_tra_data_file)
    psg_val_data = load_json(psg_val_data_file)

    tra_id_list = []
    val_id_list = []
    test_id_list = []

    for d in psg_tra_data['data']:
        if d['image_id'] in psg_tra_data['test_image_ids']:
            val_id_list.append(d['image_id'])
        else:
            tra_id_list.append(d['image_id'])

    for d in psg_val_data['data']:
        test_id_list.append(d['image_id'])
    
    tra_id_list = np.array(tra_id_list)
    val_id_list = np.array(val_id_list)
    test_id_list = np.array(test_id_list)
    print('tra', len(tra_id_list))
    print('val', len(val_id_list))
    print('test', len(test_id_list))

    np.save('/share/wangqixun/workspace/bs/psg/psg/wqx/tra_id_list.npy', tra_id_list)
    np.save('/share/wangqixun/workspace/bs/psg/psg/wqx/val_id_list.npy', val_id_list)
    np.save('/share/wangqixun/workspace/bs/psg/psg/wqx/test_id_list.npy', test_id_list)
    
    return tra_id_list, val_id_list, test_id_list


def get_test_p():
    jpg_output_dir = '/share/wangqixun/workspace/bs/psg/psg/submit/v0/submission/panseg'
    json_output_dir = '/share/wangqixun/workspace/bs/psg/psg/submit/v0/submission'
    os.makedirs(jpg_output_dir, exist_ok=True)



    # tra_id_list, val_id_list, test_id_list = get_tra_val_test_list()
    psg_val_data_file = '/share/data/psg/dataset/for_participants/psg_val_test.json'
    psg_val_data = load_json(psg_val_data_file)

    img_dir = '/share/data/psg/dataset'
    model = get_model()

    cur_nb = -1
    nb_vis = None

    all_img_dicts = []
    for d in tqdm(psg_val_data['data']):
        cur_nb += 1
        if nb_vis is not None and cur_nb > nb_vis:
            continue

        image_id = d['image_id']
        # if image_id not in test_id_list:
        #     continue
        # res['image_id'] = image_id
        img_file = os.path.join(img_dir, d['file_name'])
        img = cv2.imread(img_file)
        img_res = inference_detector(model, img)

        pan_results = img_res['pan_results']
        ins_results = img_res['ins_results']
        rela_results = img_res['rela_results']
        entityid_list = rela_results['entityid_list']
        realtion = rela_results['realtion']

        img_output = np.zeros_like(img)
        segments_info = []
        for instance_id in entityid_list:
            # instance_id == 255 background
            mask = pan_results == instance_id
            if instance_id == 255:
                r, g, b = 0, 0, 0
            else:
                r, g, b = random.choices(range(0, 255), k=3)
            coloring_mask = np.concatenate([mask[..., None]*1]*3, axis=-1)
            for j, color in enumerate([b, g, r]):
                coloring_mask[:, :, j] = coloring_mask[:, :, j] * color
            img_output = img_output + coloring_mask
            idx_class = instance_id % INSTANCE_OFFSET
            segment = dict(category_id=int(idx_class), id=rgb2id((r, g, b)))
            segments_info.append(segment)

        img_output = img_output.astype(np.uint8)
        # mask = np.sum(img_output, axis=-1) > 0
        # img_output_2 = np.copy(img)
        # img_output_2[mask] = img_output_2[mask] * 0.5 + img_output[mask] * 0.5
        # img_output = np.concatenate([img_output_2, img_output], axis=1)
        cv2.imwrite(f'{jpg_output_dir}/{cur_nb}.png', img_output)

        single_result_dict = dict(
            # image_id=image_id,
            relations=realtion,
            segments_info=segments_info,
            pan_seg_file_name='%d.png' % cur_nb,
        )
        all_img_dicts.append(single_result_dict)

    # write_json(all_img_dicts, f'{json_output_dir}/relation.json')
    with open(f'{json_output_dir}/relation.json', 'w') as outfile:
        json.dump(all_img_dicts, outfile, default=str)



def get_val_p(mode, cfg, ckp):
    jpg_output_dir = f'/share/wangqixun/workspace/bs/psg/psg/submit/{mode}/submission/panseg'
    json_output_dir = f'/share/wangqixun/workspace/bs/psg/psg/submit/{mode}/submission'

    if mode=='val':
        jpg_output_dir = '/share/wangqixun/workspace/bs/psg/psg/submit/val/submission/panseg'
        json_output_dir = '/share/wangqixun/workspace/bs/psg/psg/submit/val/submission'

    os.makedirs(jpg_output_dir, exist_ok=True)



    tra_id_list, val_id_list, test_id_list = get_tra_val_test_list()
    psg_val_data_file = '/share/data/psg/dataset/for_participants/psg_val_test.json'
    psg_val_data = load_json(psg_val_data_file)

    img_dir = '/share/data/psg/dataset'
    model = get_model(cfg, ckp)

    cur_nb = -1
    nb_vis = None

    all_img_dicts = []
    for d in tqdm(psg_val_data['data']):
        cur_nb += 1
        if nb_vis is not None and cur_nb > nb_vis:
            continue

        image_id = d['image_id']

        if mode=='val' and image_id not in val_id_list:
            continue

        img_file = os.path.join(img_dir, d['file_name'])
        img = cv2.imread(img_file)
        img_res = inference_detector(model, img)

        pan_results = img_res['pan_results']
        ins_results = img_res['ins_results']
        rela_results = img_res['rela_results']
        entityid_list = rela_results['entityid_list']
        relation = rela_results['relation']

        img_output = np.zeros_like(img)
        segments_info = []
        for instance_id in entityid_list:
            # instance_id == 255 background
            mask = pan_results == instance_id
            if instance_id == 255:
                r, g, b = 0, 0, 0
            else:
                r, g, b = random.choices(range(0, 255), k=3)
            coloring_mask = np.concatenate([mask[..., None]*1]*3, axis=-1)
            for j, color in enumerate([b, g, r]):
                coloring_mask[:, :, j] = coloring_mask[:, :, j] * color
            img_output = img_output + coloring_mask
            idx_class = instance_id % INSTANCE_OFFSET + 1
            segment = dict(category_id=int(idx_class), id=rgb2id((r, g, b)))
            segments_info.append(segment)

        img_output = img_output.astype(np.uint8)
        # mask = np.sum(img_output, axis=-1) > 0
        # img_output_2 = np.copy(img)
        # img_output_2[mask] = img_output_2[mask] * 0.5 + img_output[mask] * 0.5
        # img_output = np.concatenate([img_output_2, img_output], axis=1)
        cv2.imwrite(f'{jpg_output_dir}/{cur_nb}.png', img_output)

        single_result_dict = dict(
            # image_id=image_id,
            relations=[[s, o, r+1] for s, o, r in relation],
            segments_info=segments_info,
            pan_seg_file_name='%d.png' % cur_nb,
        )
        all_img_dicts.append(single_result_dict)

    # write_json(all_img_dicts, f'{json_output_dir}/relation.json')
    with open(f'{json_output_dir}/relation.json', 'w') as outfile:
        json.dump(all_img_dicts, outfile, default=str)





if __name__ == '__main__':
    # get_tra_val_test_list()
    # get_test_p()
    get_val_p(
        mode='val',
        cfg='/share/wangqixun/workspace/bs/psg/psg/configs/psg/v3-slurm.py',
        ckp='/share/wangqixun/workspace/bs/psg/psg/output/v3/epoch_36.pth',
    )




































