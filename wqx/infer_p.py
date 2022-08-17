from mmdet.apis import init_detector, inference_detector
import cv2
import numpy as np
import os
import random
from panopticapi.utils import rgb2id
from mmdet.core.evaluation.panoptic_utils import INSTANCE_OFFSET
from tqdm import tqdm

from xsma.generel_utils.tool_json import load_json, write_json



def get_model():
    cfg = '/share/wangqixun/workspace/bs/psg/psg/output/v0/v0.py'
    ckp = '/share/wangqixun/workspace/bs/psg/psg/output/v0/epoch_36.pth'
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
        if d['image_id'] not in val_id_list:
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
    jpg_output_dir = '/share/wangqixun/workspace/bs/psg/psg/submit/v0/pansegm'
    json_output_dir = '/share/wangqixun/workspace/bs/psg/psg/submit/v0'
    os.makedirs(jpg_output_dir, exist_ok=True)



    # tra_id_list, val_id_list, test_id_list = get_tra_val_test_list()
    psg_val_data_file = '/share/data/psg/dataset/for_participants/psg_val_test.json'
    psg_val_data = load_json(psg_val_data_file)

    img_dir = '/share/data/psg/dataset'
    model = get_model()

    cur_nb = 0
    nb_vis = 20

    all_img_dicts = []
    for d in tqdm(psg_val_data['data']):
        image_id = d['image_id']
        # if image_id not in test_id_list:
        #     continue
        # res['image_id'] = image_id
        img_file = os.path.join(img_dir, d['file_name'])
        img = cv2.imread(img_file)
        img_res = inference_detector(model, img)
        pan_results = img_res['pan_results']
        bbox_results, segm_results = img_res['ins_results']

        # img_output = np.zeros_like(img)
        # for idx_class in range(len(bbox_results)):
        #     class_bbox_results = bbox_results[idx_class]
        #     class_segm_results = segm_results[idx_class]
        #     for idx_sample in range(len(class_bbox_results)):
        #         sample_class_bbox_result = class_bbox_results[idx_sample]
        #         sample_class_segm_result = class_segm_results[idx_sample]
        #         score = sample_class_bbox_result[4]
        #         segm = sample_class_segm_result
        #         if score < 0.4:
        #             continue
        #         r, g, b = random.choices(range(0, 255), k=3)
        #         print(r, g, b)
        #         coloring_mask = np.concatenate([segm[..., None]*1]*3, axis=-1)
        #         for j, color in enumerate([b, g, r]):
        #             coloring_mask[:, :, j] = coloring_mask[:, :, j] * color
        #         img_output = img_output + coloring_mask
        #         segment = dict(category_id=int(idx_class), id=rgb2id((r, g, b)))
        # img_output = img_output.astype(np.uint8)
        # mask = np.sum(img_output, axis=-1) > 0
        # img_output_2 = np.copy(img)
        # img_output_2[mask] = img_output_2[mask] * 0.5 + img_output[mask] * 0.5
        # img_output = np.concatenate([img_output_2, img_output], axis=1)
        # cv2,imwrite(f'/share/wangqixun/workspace/bs/psg/psg/wqx/{idx}.jpg', img_output)

        instance_id_all = np.unique(pan_results)
        img_output = np.zeros_like(img)
        segments_info = []
        for instance_id in instance_id_all:
            if not (instance_id <= 133 or instance_id >= INSTANCE_OFFSET):
                continue
            mask = pan_results == instance_id
            r, g, b = random.choices(range(0, 255), k=3)
            coloring_mask = np.concatenate([mask[..., None]*1]*3, axis=-1)
            for j, color in enumerate([b, g, r]):
                coloring_mask[:, :, j] = coloring_mask[:, :, j] * color
            img_output = img_output + coloring_mask
            idx_class = instance_id % INSTANCE_OFFSET
            segment = dict(category_id=int(idx_class), id=rgb2id((r, g, b)))
            segments_info.append(segment)

        img_output = img_output.astype(np.uint8)
        mask = np.sum(img_output, axis=-1) > 0
        img_output_2 = np.copy(img)
        img_output_2[mask] = img_output_2[mask] * 0.5 + img_output[mask] * 0.5
        img_output = np.concatenate([img_output_2, img_output], axis=1)
        cv2.imwrite(f'{jpg_output_dir}/{cur_nb}.jpg', img_output)
        cur_nb += 1

        single_result_dict = dict(
            image_id=image_id,
            relations='锋哥 YYDS',
            segments_info=segments_info,
            pan_seg_file_name='%d.png' % cur_nb,
        )
        all_img_dicts.append(single_result_dict)

    write_json(all_img_dicts, f'{json_output_dir}/relation.json')


def merge_wqx_gxf():
    wqx_json_file_path = '/share/wangqixun/workspace/bs/psg/psg/wqx/data/relation.json'
    gxf_json_file_path = '/share/wangqixun/workspace/bs/psg/psg/wqx/data/relation.json'

    wqx_json = load_json(wqx_json_file_path)
    gxf_json = load_json(gxf_json_file_path)

    for idx in range(len(wqx_json)):
        relation = wqx_json[idx]
        






if __name__ == '__main__':
    # get_tra_val_test_list()
    get_test_p()




































