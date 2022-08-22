# PSG

结合了[refine-mask](https://github.com/zhanggang001/RefineMask)、[CBNetv2](https://github.com/VDIGPKU/CBNetV2),其中relation部分借鉴[transformers](https://github.com/huggingface/transformers)

整体框架还是基于[mmdet](https://github.com/open-mmlab/mmdetection)


<br>

## Install
环境参考 [get_started.md](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/get_started.md) 
```
pip install -e .
```

此外，还需要 [apex](https://github.com/NVIDIA/apex) 。建议通过下面代码安装
```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```


<br>

## 数据准备

下载 [coco instance val 2017](https://cocodataset.org/#download)，用于验证 psg val 的instance map

修改 `wqx/main.py` 中下列文件路径
```python
if __name__ == '__main__':
    # raw data file
    raw_psg_data='/share/data/psg/dataset/for_participants/psg_train_val.json'
    raw_coco_val_json_file='/share/data/coco/annotations/instances_val2017.json'

    # output file
    output_coco80_val_instance_json = '/share/wangqixun/workspace/bs/psg/psg/data/instances_val2017_coco80.json'
    output_tra_json='/share/wangqixun/workspace/bs/psg/psg/data/psg_tra.json'
    output_val_json='/share/wangqixun/workspace/bs/psg/psg/data/psg_val.json'
    output_val_instance_json='/share/wangqixun/workspace/bs/psg/psg/data/psg_instance_val.json'

```
执行
```
python wqx/main.py
```




<br>

## (可能)需要的一些预训练权重

### 实例分割：
[Refine Cascade-Last-Mask RCNN](https://github.com/wangqixun/refine_caslm)

|model |train|val| bbox map | segm map | checkpoint|
|:--- | :-----: |:-----: |:-----: | ----: | :----:|
|cbv2_swimtiny_mask_rcnn|coco train|coco minival|50.2|44.5|[链接](https://github.com/CBNetwork/storage/releases/download/v1.0.0/mask_rcnn_cbv2_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco.pth.zip)(from [repo](https://github.com/VDIGPKU/CBNetV2))|
|refine_cbv2_swimtiny_mask_rcnn|coco train|coco val|50.7|46.5|[链接](https://cloud.189.cn/t/iMbINfRRfER3)(访问码:fj4k)|
|refine_cbv2_swimtiny_cascade-last-mask_rcnn |coco train|coco val| 52.8 |46.8| [链接](https://cloud.189.cn/t/BJBZjanuERR3)(访问码:qr0n)

### 语义分割：
[U-2-Net](https://github.com/xuebinqin/U-2-Net)，[预训练权重](https://drive.google.com/file/d/1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ/view?usp=sharing)

### 关系分类：
[transformers](https://github.com/huggingface/transformers)，
[预训练权重](https://huggingface.co/hfl/chinese-roberta-wwm-ext)



<br>

## 训练
+ 1、调整 `configs/psg/xxxx.py` 中预训练路径、输出路径
+ 2、训练咯
```
# 8卡训练
bash tools/dist_train.sh configs/psg/v3-slurm.py 8 
```

<br>

## Submit
+ 1、调整 `wqx/infer_p.py` 中 `cfg` 和 `ckp`
+ 2、推理
```
python wqx/infer_p.py
```
+ 3、打包
```
cd submit/v3
zip -r submission.zip submission/
```









