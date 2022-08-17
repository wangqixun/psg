from xsma.generel_utils.tool_json import load_json, write_json
import numpy as np
from infer_p import get_tra_val_test_list
from tqdm import tqdm


def f2(psg_data, id_list, output_json):

    out_json = {}
    out_json['images'] = []
    out_json['annotations'] = []
    out_json['categories'] = []
    out_json['relations_categories'] = []
    

    CLASSES = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
        ' truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
        'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
        'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
        'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
        'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
        'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'banner',
        'blanket', 'bridge', 'cardboard', 'counter', 'curtain', 'door-stuff',
        'floor-wood', 'flower', 'fruit', 'gravel', 'house', 'light',
        'mirror-stuff', 'net', 'pillow', 'platform', 'playingfield',
        'railroad', 'river', 'road', 'roof', 'sand', 'sea', 'shelf', 'snow',
        'stairs', 'tent', 'towel', 'wall-brick', 'wall-stone', 'wall-tile',
        'wall-wood', 'water-other', 'window-blind', 'window-other',
        'tree-merged', 'fence-merged', 'ceiling-merged', 'sky-other-merged',
        'cabinet-merged', 'table-merged', 'floor-other-merged',
        'pavement-merged', 'mountain-merged', 'grass-merged', 'dirt-merged',
        'paper-merged', 'food-other-merged', 'building-other-merged',
        'rock-merged', 'wall-other-merged', 'rug-merged'
    ]
    THING_CLASSES = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
        'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
        'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
        'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
        'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
        'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
        'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    STUFF_CLASSES = [
        'banner', 'blanket', 'bridge', 'cardboard', 'counter', 'curtain',
        'door-stuff', 'floor-wood', 'flower', 'fruit', 'gravel', 'house',
        'light', 'mirror-stuff', 'net', 'pillow', 'platform', 'playingfield',
        'railroad', 'river', 'road', 'roof', 'sand', 'sea', 'shelf', 'snow',
        'stairs', 'tent', 'towel', 'wall-brick', 'wall-stone', 'wall-tile',
        'wall-wood', 'water-other', 'window-blind', 'window-other',
        'tree-merged', 'fence-merged', 'ceiling-merged', 'sky-other-merged',
        'cabinet-merged', 'table-merged', 'floor-other-merged',
        'pavement-merged', 'mountain-merged', 'grass-merged', 'dirt-merged',
        'paper-merged', 'food-other-merged', 'building-other-merged',
        'rock-merged', 'wall-other-merged', 'rug-merged'
    ]

    # categories
    class2id = {}
    for idx in range(len(CLASSES)):
        supercategory = ''
        name = CLASSES[idx]
        isthing = 1 if name in THING_CLASSES else 0
        categorie = dict(
            supercategory=supercategory,
            isthing=isthing,
            id=idx+1,
            name=name,
        )
        out_json['categories'].append(categorie)
        class2id[name] = idx
    
    # relations_categories
    for idx in range(len(psg_data['predicate_classes'])):
        name = psg_data['predicate_classes'][idx]
        categorie = dict(
            id=idx+1,
            name=name,
        )
        out_json['relations_categories'].append(categorie)

    # images
    for idx in tqdm(range(len(psg_data['data']))):
        psg_data_info = psg_data['data'][idx]
        file_name = psg_data_info['file_name']
        height = psg_data_info['height']
        width = psg_data_info['width']
        img_id = int(psg_data_info['image_id'])
        if str(img_id) not in id_list:
            continue
        image = dict(
            file_name=file_name,
            height=height,
            width=width,
            id=img_id,
        )
        out_json['images'].append(image)

    # annotations
    for idx in tqdm(range(len(psg_data['data']))):
        psg_data_info = psg_data['data'][idx]
        file_name = psg_data_info['pan_seg_file_name']
        image_id = int(psg_data_info['image_id'])
        relations = psg_data_info['relations']
        if str(image_id) not in id_list:
            continue
        segments_info = []
        for idx_segment in range(len(psg_data_info['segments_info'])):
            id = psg_data_info['segments_info'][idx_segment]['id']
            category_id = psg_data_info['segments_info'][idx_segment]['category_id']
            iscrowd = psg_data_info['segments_info'][idx_segment]['iscrowd']
            bbox = psg_data_info['annotations'][idx_segment]['bbox']
            area = psg_data_info['segments_info'][idx_segment]['area']
            x1, y1, x2, y2 = bbox
            w, h = x2 - x1, y2 - y1
            bbox = [x1, y1, w, h]
            segment = dict(
                id=id,
                category_id=category_id+1,
                iscrowd=iscrowd,
                bbox=bbox,
                area=area,
            )
            segments_info.append(segment)
        annotation = dict(
            segments_info=segments_info,
            file_name=file_name,
            image_id=image_id,
            relations=relations,
        )
        out_json['annotations'].append(annotation)


    write_json(out_json, output_json)

    

def f3(psg_data, id_list, output_instance_json, coco_json_file):
    # coco_json_file = '/share/data/coco/annotations/instances_train2017_coco80.json'
    coco_json = load_json(coco_json_file)


    out_json = {}
    out_json['images'] = []
    out_json['annotations'] = []
    out_json['categories'] = coco_json['categories']
    
    # images
    use_coco_id_list = {}
    for idx in tqdm(range(len(psg_data['data']))):
        psg_data_info = psg_data['data'][idx]
        coco_image_id = psg_data_info['coco_image_id']
        img_id = int(psg_data_info['image_id'])
        file_name = psg_data_info['file_name']
        height = psg_data_info['height']
        width = psg_data_info['width']
        if str(img_id) not in id_list:
            continue
        image = dict(
            file_name=file_name,
            height=height,
            width=width,
            id=img_id,
        )
        out_json['images'].append(image)
        use_coco_id_list[coco_image_id] = img_id

    # annotations
    for idx in tqdm(range(len(coco_json['annotations']))):
        ann_info = coco_json['annotations'][idx]
        image_id = ann_info['image_id']
        if str(image_id) not in use_coco_id_list:
            continue
        ann_info['image_id'] = use_coco_id_list[str(image_id)]
        out_json['annotations'].append(ann_info)


    write_json(out_json, output_instance_json)

    

def f1():
    output_tra_json = '/share/wangqixun/workspace/bs/psg/psg/data/psg_tra.json'
    output_val_json = '/share/wangqixun/workspace/bs/psg/psg/data/psg_val.json'
    output_tra_instance_json = '/share/wangqixun/workspace/bs/psg/psg/data/psg_instance_tra.json'
    output_val_instance_json = '/share/wangqixun/workspace/bs/psg/psg/data/psg_instance_val.json'
    psg_data_file = '/share/data/psg/dataset/for_participants/psg_train_val.json'
    # psg_data_file = '/share/data/psg/dataset/for_participants/psg_val_test.json'

    tra_id_list, val_id_list, test_id_list = get_tra_val_test_list()

    psg_data = load_json(psg_data_file)
    # test_image_ids = psg_data['test_image_ids']

    # id_all = [data_info['image_id'] for data_info in psg_data['data']]
    # id_all = np.unique(id_all)
    # np.random.shuffle(id_all)
    # id_tra = id_all[:int(len(id_all)*0.9)]
    # id_val = id_all[int(len(id_all)*0.9):]
    # id_tra = np.load(tra_list_npy)
    # id_val = np.load(val_list_npy)
    
    f2(psg_data, tra_id_list, output_tra_json)
    f2(psg_data, val_id_list, output_val_json)
    f3(psg_data, tra_id_list, output_tra_instance_json, coco_json_file = '/share/data/coco/annotations/instances_train2017_coco80.json')
    f3(psg_data, val_id_list, output_val_instance_json, coco_json_file = '/share/data/coco/annotations/instances_val2017_coco80.json')











if __name__ == '__main__':
    f1()


