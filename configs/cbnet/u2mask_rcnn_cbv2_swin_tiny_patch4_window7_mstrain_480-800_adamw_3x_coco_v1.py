_base_ = [
    '../swin/mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco.py'
]
find_unused_parameters=True

model = dict(
    backbone=dict(
        type='CBSwinTransformer',
    ),
    neck=dict(
        type='CBFPN',
        upsample_cfg=dict(mode='bilinear'),
    ),
    roi_head=dict(
        type='U2netRoIHead',
        mask_roi_extractor=dict(
            type='DownSingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=112, sampling_ratio=0),
        ),
        mask_head=dict(
            type='U2MaskHead',
        ),
    ),
    train_cfg=dict(
        rcnn=dict(
            mask_size=112,
            sampler=dict(
                num=128,
            ),
        ),
    ),
)



evaluation = dict(metric=['bbox', 'segm'], classwise=True)



# sit
# load_from = '/share/wangqixun/workspace/github_project/mmdetection/model_dl/mask_rcnn_cbv2_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco.pth'
# resume_from = '/share/wangqixun/workspace/github_project/CBNetV2_train/wqx/u2mask_rcnn_cbv2_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco/latest.pth'
# work_dir = '/share/wangqixun/workspace/github_project/CBNetV2_train/wqx/u2mask_rcnn_cbv2_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco'


# 平台
load_from = '/mnt/mmtech01/usr/guiwan/workspace/model_dl/mask_rcnn_cbv2_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco.pth'
resume_from = None

work_dir = '/mnt/mmtech01/usr/guiwan/workspace/instance_segm/wqx/u2mask_rcnn_cbv2_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco_v1'





