_base_ = 'htc_cbv2_swin_large_patch4_window7_mstrain_400-1400_giou_4conv1f_adamw_1x_coco.py'

model = dict(
    test_cfg = dict(
        rcnn=dict(
            score_thr=0.001,
            nms=dict(type='soft_nms'),
            mask_thr_binary=-1,
            # mask_thr_binary=0.5,
        ),
    )
)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        # img_scale=[(1600, 800), (1600, 1000), (1800, 1200) ],
        # flip=True,
        img_scale=[(1600, 800), ],
        flip=True,
        flip_direction=['vertical'], # "horizontal", "vertical", "diagonal"
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
            val=dict(pipeline=test_pipeline),
            test=dict(pipeline=test_pipeline))