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
    ),
    roi_head=dict(
        type='U2netRoIHead',
        mask_roi_extractor=dict(
            roi_layer=dict(type='RoIAlign', output_size=40, sampling_ratio=0),
        ),
        mask_head=dict(
            type='U2MaskHead',
        ),
    ),
    train_cfg=dict(
        rcnn=dict(
            mask_size=40,
        ),
    ),
)


evaluation = dict(metric=['bbox', 'segm'], classwise=True)




load_from = '/share/wangqixun/workspace/github_project/mmdetection/model_dl/mask_rcnn_cbv2_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco.pth'
resume_from = '/share/wangqixun/workspace/github_project/CBNetV2_train/wqx/u2mask_rcnn_cbv2_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco/latest.pth'
# resume_from = None
work_dir = '/share/wangqixun/workspace/github_project/CBNetV2_train/wqx/u2mask_rcnn_cbv2_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco'

