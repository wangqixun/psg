# PSG

结合了[refine-mask](https://github.com/zhanggang001/RefineMask)、[CBNetv2](https://github.com/VDIGPKU/CBNetV2),其中relation部分借鉴[transformers](https://github.com/huggingface/transformers)

整体框架还是基于[mmdet](https://github.com/open-mmlab/mmdetection)





<br>


## 需要的coco预训练 checkpoint
### 锋哥下载最后一个就行
|model |train|val| bbox map | segm map | checkpoint|
|:--- | :-----: |:-----: |:-----: | ----: | :----:|
|cbv2_swimtiny_mask_rcnn|coco train|coco minival|50.2|44.5|[链接](https://github.com/CBNetwork/storage/releases/download/v1.0.0/mask_rcnn_cbv2_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco.pth.zip)(from [repo](https://github.com/VDIGPKU/CBNetV2))|
|refine_cbv2_swimtiny_mask_rcnn|coco train|coco val|50.7|46.5|[链接](https://cloud.189.cn/t/iMbINfRRfER3)(访问码:fj4k)|
|refine_cbv2_swimtiny_cascade-last-mask_rcnn |coco train|coco val| 52.8 |46.8| [链接](https://cloud.189.cn/t/BJBZjanuERR3)(访问码:qr0n)

<br>

## install
安装参考[get_started.md](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/get_started.md)

此外，还需要
```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

<br>


## 训练

```
bash tools/dist_train.sh configs/psg/v3-slurm.py 8 
```

## submit
```
python wqx/infer_p.py
```









