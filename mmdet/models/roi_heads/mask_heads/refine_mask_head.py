import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

import numpy as np
from mmcv.cnn import ConvModule, build_conv_layer, build_upsample_layer
from mmcv.ops.roi_align import roi_align, RoIAlign

from mmdet.core.mask.structures import polygon_to_bitmap, BitmapMasks
from mmdet.models.builder import HEADS, build_loss, build_roi_extractor
from .fcn_mask_head import _do_paste_mask
from .fcn_mask_head import BYTES_PER_FLOAT, GPU_MEM_LIMIT
from mmcv.ops import SimpleRoIAlign
from .u2net import U2NET, U2NETP


class MultiBranchFusion(nn.Module):

    def __init__(self, feat_dim, dilations=[1, 3, 5]):
        super(MultiBranchFusion, self).__init__()

        for idx, dilation in enumerate(dilations):
            self.add_module(f'dilation_conv_{idx + 1}', ConvModule(
                feat_dim, feat_dim, kernel_size=3, padding=dilation, dilation=dilation))

        self.merge_conv = ConvModule(feat_dim, feat_dim, kernel_size=1, act_cfg=None)

    def forward(self, x):
        feat_1 = self.dilation_conv_1(x)
        feat_2 = self.dilation_conv_2(x)
        feat_3 = self.dilation_conv_3(x)
        out_feat = self.merge_conv(feat_1 + feat_2 + feat_3)
        return out_feat


class MultiBranchFusionAvg(MultiBranchFusion):

    def forward(self, x):
        feat_1 = self.dilation_conv_1(x)
        feat_2 = self.dilation_conv_2(x)
        feat_3 = self.dilation_conv_3(x)
        feat_4 = F.avg_pool2d(x, x.shape[-1])
        out_feat = self.merge_conv(feat_1 + feat_2 + feat_3 + feat_4)
        return out_feat


class SFMStage(nn.Module):

    def __init__(self,
                 semantic_in_channel=256,
                 semantic_out_channel=256,
                 instance_in_channel=256,
                 instance_out_channel=256,
                 fusion_type='MultiBranchFusion',
                 dilations=[1, 3, 5],
                 out_size=14,
                 num_classes=80,
                 semantic_out_stride=4,
                 mask_use_sigmoid=False,
                 mask_predictor_cfg=None,
                 upsample_cfg=dict(type='bilinear', scale_factor=2)):
        super(SFMStage, self).__init__()

        self.semantic_out_stride = semantic_out_stride
        self.mask_use_sigmoid = mask_use_sigmoid
        self.num_classes = num_classes

        # for extracting instance-wise semantic feats
        self.semantic_transform_in = nn.Conv2d(semantic_in_channel, semantic_out_channel, 1)
        self.semantic_roi_extractor = build_roi_extractor(dict(
                type='SingleRoIExtractor',
                roi_layer=dict(type='RoIAlign', output_size=out_size, sampling_ratio=0),
                out_channels=semantic_out_channel,
                featmap_strides=[semantic_out_stride, ]))
        self.semantic_transform_out = nn.Conv2d(semantic_out_channel, semantic_out_channel, 1)

        self.mask_predictor_cfg = mask_predictor_cfg
        if self.mask_predictor_cfg is None:
            self.instance_logits = nn.Conv2d(instance_in_channel, num_classes, 1)
        else:
            self.instance_logits = build_conv_layer(self.mask_predictor_cfg, instance_in_channel, num_classes, 1)

        fuse_in_channel = instance_in_channel + semantic_out_channel + 2
        self.fuse_conv = nn.ModuleList([
            nn.Conv2d(fuse_in_channel, instance_in_channel, 1),
            globals()[fusion_type](instance_in_channel, dilations=dilations)])

        self.fuse_transform_out = nn.Conv2d(instance_in_channel, instance_out_channel - 2, 1)
        self.upsample_cfg = upsample_cfg
        if upsample_cfg['type'] == 'carafe':
            _upsample_cfg = upsample_cfg.copy()
            _upsample_cfg['channels'] = instance_out_channel - 2
            self.upsample = build_upsample_layer(_upsample_cfg)
            _upsample_cfg = upsample_cfg.copy()
            _upsample_cfg['channels'] = 1
            self.upsample_instance_masks = build_upsample_layer(_upsample_cfg)
            _upsample_cfg = upsample_cfg.copy()
            _upsample_cfg['channels'] = 1
            self.upsample_ins_semantic_masks = build_upsample_layer(_upsample_cfg)
        else:
            self.upsample = build_upsample_layer(upsample_cfg.copy())
        self.relu = nn.ReLU(inplace=True)

        self._init_weights()

    def _init_weights(self):
        for m in [self.semantic_transform_in, self.semantic_transform_out, self.instance_logits, self.fuse_transform_out]:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)

        for m in self.fuse_conv:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, instance_feats, semantic_feat, semantic_pred, rois, roi_labels):
        concat_tensors = [instance_feats]

        # instance-wise semantic feats
        semantic_feat = self.relu(self.semantic_transform_in(semantic_feat))
        ins_semantic_feats = self.semantic_roi_extractor([semantic_feat,], rois)
        ins_semantic_feats = self.relu(self.semantic_transform_out(ins_semantic_feats))
        concat_tensors.append(ins_semantic_feats)

        # instance masks
        # instance_preds = self.instance_logits(instance_feats)[torch.arange(len(rois)), roi_labels][:, None]
        roi_labels[roi_labels == -1] = 0
        instance_feats = self.instance_logits(instance_feats)
        b,c,h,w = instance_feats.shape
        trt_tensor = instance_feats.reshape([-1, h, w])
        indices = torch.arange(b, device=roi_labels.device, dtype=roi_labels.dtype) * c + roi_labels
        instance_preds = torch.index_select(trt_tensor, 0, indices.to(trt_tensor.device))
        instance_preds = instance_preds.unsqueeze(1)
        _instance_preds = instance_preds.sigmoid() if self.mask_use_sigmoid else instance_preds
        instance_masks = F.interpolate(_instance_preds, instance_feats.shape[-2:], mode='bilinear', align_corners=True)
        instance_masks = instance_masks.to(semantic_feat.dtype)
        concat_tensors.append(instance_masks)

        # instance-wise semantic masks
        _semantic_pred = semantic_pred.sigmoid() if self.mask_use_sigmoid else semantic_pred
        if rois.shape[0] == 0:
            ins_semantic_masks = torch.zeros([0, 1, instance_feats.shape[-2], instance_feats.shape[-1]], dtype=instance_feats.dtype, device=instance_feats.device)
        else:
            # ins_semantic_masks = roi_align(
            #     _semantic_pred, rois.to(_semantic_pred.dtype), instance_feats.shape[-2:], 1.0 / self.semantic_out_stride, 0, 'avg', True)
            ins_semantic_masks = RoIAlign(
                instance_feats.shape[-2:], 
                1.0 / self.semantic_out_stride, 
                0, 
                'avg', 
                True
            )(_semantic_pred, rois.to(_semantic_pred.dtype))
        ins_semantic_masks = F.interpolate(
            ins_semantic_masks, instance_feats.shape[-2:], mode='bilinear', align_corners=True)
        ins_semantic_masks = ins_semantic_masks.to(semantic_feat.dtype)
        concat_tensors.append(ins_semantic_masks)

        # fuse instance feats & instance masks & semantic feats & semantic masks
        fused_feats = torch.cat(concat_tensors, dim=1)
        for conv in self.fuse_conv:
            fused_feats = self.relu(conv(fused_feats))

        fused_feats = self.relu(self.fuse_transform_out(fused_feats))
        fused_feats = self.relu(self.upsample(fused_feats))

        # concat instance and semantic masks with fused feats again
        if self.upsample_cfg['type'] == 'carafe':
            instance_masks = self.upsample_instance_masks(_instance_preds)
            ins_semantic_masks = self.upsample_ins_semantic_masks(ins_semantic_masks)
        else:
            instance_masks = F.interpolate(_instance_preds, fused_feats.shape[-2:], mode='bilinear', align_corners=True)
            ins_semantic_masks = F.interpolate(ins_semantic_masks, fused_feats.shape[-2:], mode='bilinear', align_corners=True)
        instance_masks = instance_masks.to(semantic_feat.dtype)
        ins_semantic_masks = ins_semantic_masks.to(semantic_feat.dtype)
        fused_feats = torch.cat([fused_feats, instance_masks, ins_semantic_masks], dim=1)


        instance_preds = instance_preds.to(semantic_feat.dtype)
        fused_feats = fused_feats.to(semantic_feat.dtype)
        return instance_preds, fused_feats


@HEADS.register_module()
class RefineMaskHead(nn.Module):

    def __init__(self,
                 num_convs_instance=2,
                 num_convs_semantic=4,
                 conv_in_channels_instance=256,
                 conv_in_channels_semantic=256,
                 conv_kernel_size_instance=3,
                 conv_kernel_size_semantic=3,
                 conv_out_channels_instance=256,
                 conv_out_channels_semantic=256,
                 conv_cfg=None,
                 norm_cfg=None,
                 u2semantic=False,
                 fusion_type='MultiBranchFusion',
                 dilations=[1, 3, 5],
                 semantic_out_stride=4,
                 mask_use_sigmoid=False,
                 mask_predictor_cfg=None,
                 stage_num_classes=[80, 80, 80, 80],
                 stage_sup_size=[14, 28, 56, 112],
                 upsample_cfg=dict(type='bilinear', scale_factor=2),
                 loss_cfg=dict(
                    type='RefineCrossEntropyLoss',
                    stage_instance_loss_weight=[0.25, 0.5, 0.75, 1.0],
                    semantic_loss_weight=1.0,
                    boundary_width=2,
                    start_stage=1)):
        super(RefineMaskHead, self).__init__()

        self.num_convs_instance = num_convs_instance
        self.conv_kernel_size_instance = conv_kernel_size_instance
        self.conv_in_channels_instance = conv_in_channels_instance
        self.conv_out_channels_instance = conv_out_channels_instance

        self.num_convs_semantic = num_convs_semantic
        self.conv_kernel_size_semantic = conv_kernel_size_semantic
        self.conv_in_channels_semantic = conv_in_channels_semantic
        self.conv_out_channels_semantic = conv_out_channels_semantic

        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.u2semantic = u2semantic

        self.semantic_out_stride = semantic_out_stride
        self.stage_sup_size = stage_sup_size
        self.stage_num_classes = stage_num_classes

        self._build_conv_layer('instance')
        self._build_conv_layer('semantic')
        self.loss_func = build_loss(loss_cfg)

        assert len(self.stage_sup_size) > 1
        self.stages = nn.ModuleList()
        out_channel = conv_out_channels_instance
        for idx, out_size in enumerate(self.stage_sup_size[:-1]):
            in_channel = out_channel
            out_channel = in_channel // 2

            new_stage = SFMStage(
                semantic_in_channel=conv_out_channels_semantic,
                semantic_out_channel=in_channel,
                instance_in_channel=in_channel,
                instance_out_channel=out_channel,
                fusion_type=fusion_type,
                dilations=dilations,
                out_size=out_size,
                num_classes=self.stage_num_classes[idx],
                semantic_out_stride=semantic_out_stride,
                mask_use_sigmoid=mask_use_sigmoid,
                mask_predictor_cfg=mask_predictor_cfg,
                upsample_cfg=upsample_cfg)

            self.stages.append(new_stage)

        self.final_instance_logits = nn.Conv2d(out_channel, self.stage_num_classes[-1], 1)
        self.semantic_logits = nn.Conv2d(conv_out_channels_semantic, 1, 1)
        self.relu = nn.ReLU(inplace=True)

    def _build_conv_layer_raw(self, name):
        out_channels = getattr(self, f'conv_out_channels_{name}')
        conv_kernel_size = getattr(self, f'conv_kernel_size_{name}')

        convs = []
        for i in range(getattr(self, f'num_convs_{name}')):
            in_channels = getattr(self, f'conv_in_channels_{name}') if i == 0 else out_channels
            conv = ConvModule(in_channels, out_channels, conv_kernel_size, dilation=1, padding=1)
            convs.append(conv)

        self.add_module(f'{name}_convs', nn.ModuleList(convs))


    def _build_semantic_u2(self):
        convs = [
            U2NETP(self.conv_in_channels_semantic, self.conv_out_channels_semantic),
        ]
        self.add_module(f'semantic_convs', nn.ModuleList(convs))


    def _build_conv_layer(self, name):
        if name == 'semantic' and self.u2semantic:
            self._build_semantic_u2()
        else:
            self._build_conv_layer_raw(name)

    def init_weights(self):
        for m in [self.final_instance_logits, self.semantic_logits]:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)

    def forward(self, instance_feats, semantic_feat, rois, roi_labels):
        for conv in self.instance_convs:
            instance_feats = conv(instance_feats)

        for conv in self.semantic_convs:
            semantic_feat = conv(semantic_feat)

        semantic_pred = self.semantic_logits(semantic_feat)

        stage_instance_preds = []
        for idx_stage, stage in enumerate(self.stages):
            instance_preds, instance_feats = stage(instance_feats, semantic_feat, semantic_pred, rois, roi_labels)
            stage_instance_preds.append(instance_preds)

        # for LVIS, use class-agnostic classifier for the last stage
        if self.stage_num_classes[-1] == 1:
            roi_labels = roi_labels.clamp(max=0)

        # instance_preds = self.final_instance_logits(instance_feats)[torch.arange(len(rois)), roi_labels][:, None]
        roi_labels[roi_labels == -1] = 0
        instance_feats = self.final_instance_logits(instance_feats)
        b,c,h,w = instance_feats.shape
        trt_tensor = instance_feats.reshape([-1, h, w])
        indices = torch.arange(b, device=roi_labels.device, dtype=roi_labels.dtype) * c + roi_labels
        instance_preds = torch.index_select(trt_tensor, 0, indices.to(trt_tensor.device))
        instance_preds = instance_preds.unsqueeze(1)
        stage_instance_preds.append(instance_preds)

        return stage_instance_preds, semantic_pred

    def get_targets(self, pos_bboxes_list, pos_assigned_gt_inds_list, gt_masks_list):

        def _generate_instance_targets(pos_proposals, pos_assigned_gt_inds, gt_masks, mask_size=None):
            device = pos_proposals.device
            proposals_np = pos_proposals.cpu().numpy()
            maxh, maxw = gt_masks.height, gt_masks.width
            proposals_np[:, [0, 2]] = np.clip(proposals_np[:, [0, 2]], 0, maxw)
            proposals_np[:, [1, 3]] = np.clip(proposals_np[:, [1, 3]], 0, maxh)
            pos_assigned_gt_inds = pos_assigned_gt_inds.cpu().numpy()

            # crop and resize the instance mask
            resize_masks = gt_masks.crop_and_resize(
                proposals_np, _pair(mask_size), inds=pos_assigned_gt_inds, device=device,).to_ndarray()
            instance_targets = torch.from_numpy(resize_masks).float().to(device)  # Tensor(Bitmaps)

            return instance_targets

        semantic_target_list = []
        stage_instance_targets_list = [[] for _ in range(len(self.stage_sup_size))]
        for pos_bboxes, pos_assigned_gt_inds, gt_masks in zip(pos_bboxes_list, pos_assigned_gt_inds_list, gt_masks_list):
            # multi-stage instance mask targets
            stage_instance_targets = [_generate_instance_targets(
                pos_bboxes, pos_assigned_gt_inds, gt_masks, mask_size=mask_size) for mask_size in self.stage_sup_size]

            # binary image semantic target
            if isinstance(gt_masks, BitmapMasks):
                instance_masks = torch.from_numpy(gt_masks.to_ndarray()).to(device=pos_bboxes.device, dtype=pos_bboxes.dtype)
            else:
                im_height, im_width = gt_masks.height, gt_masks.width
                instance_masks = [polygon_to_bitmap(polygon, im_height, im_width) for polygon in gt_masks]
                instance_masks = torch.from_numpy(np.stack(instance_masks)).to(device=pos_bboxes.device, dtype=pos_bboxes.dtype)
            # print('instance_masks', instance_masks.shape)
            if instance_masks.shape[0] == 0:
                semantic_target = instance_masks
            else:
                semantic_target = instance_masks.max(dim=0, keepdim=True)[0]                
            semantic_target_list.append(semantic_target)
            # print('semantic_target', semantic_target.shape)

            for stage_idx in range(len(self.stage_sup_size)):
                stage_instance_targets_list[stage_idx].append(stage_instance_targets[stage_idx])

        stage_instance_targets = [torch.cat(targets) for targets in stage_instance_targets_list]

        max_h = max([target.shape[-2] for target in semantic_target_list])
        max_w = max([target.shape[-1] for target in semantic_target_list])
        semantic_target = torch.zeros(
            (len(semantic_target_list), max_h, max_w),
            dtype=semantic_target_list[0].dtype, device=semantic_target_list[0].device)
        for idx, target in enumerate(semantic_target_list):
            if target.shape[0] == 0:
                continue
            else:
                semantic_target[idx, :target.shape[-2], :target.shape[-1]] = target

        return stage_instance_targets, semantic_target

    def loss(self, stage_instance_preds, semantic_pred, stage_instance_targets, semantic_target):
        if isinstance(stage_instance_targets, list):
            stage_instance_targets = [sit.to(semantic_pred.dtype) for sit in stage_instance_targets]
        else:
            stage_instance_targets = stage_instance_targets.to(semantic_pred.dtype)
        if isinstance(semantic_target, list):
            semantic_target = [st.to(semantic_pred.dtype) for st in semantic_target]
        else:
            semantic_target = semantic_target.to(semantic_pred.dtype)
        
        # isnan = False
        # for idx in range(len(stage_instance_preds)):
        #     instance_pred = stage_instance_preds[idx]
        #     if instance_pred.size(0) == 0:
        #         isnan = True
        #         # loss_mask = instance_pred.sum()


        loss_instance, loss_semantic = self.loss_func(
            stage_instance_preds, semantic_pred, stage_instance_targets, semantic_target)

        return dict(loss_instance=loss_instance), dict(loss_semantic=loss_semantic)

    def get_seg_masks(self, mask_pred, det_bboxes, det_labels, rcnn_test_cfg, ori_shape, scale_factor, rescale):

        mask_pred = mask_pred.sigmoid()

        device = mask_pred[0].device
        bboxes = det_bboxes[:, :4]
        labels = det_labels

        if rescale:
            img_h, img_w = ori_shape[:2]
        else:
            img_h = np.round(ori_shape[0] * scale_factor).astype(np.int32)
            img_w = np.round(ori_shape[1] * scale_factor).astype(np.int32)
            scale_factor = 1.0

        if not isinstance(scale_factor, (float, torch.Tensor)):
            scale_factor = bboxes.new_tensor(scale_factor)
        bboxes = bboxes / scale_factor

        N = len(mask_pred)
        # The actual implementation split the input into chunks,
        # and paste them chunk by chunk.
        if device.type == 'cpu':
            # CPU is most efficient when they are pasted one by one with
            # skip_empty=True, so that it performs minimal number of
            # operations.
            num_chunks = N
        else:
            # GPU benefits from parallelism for larger chunks,
            # but may have memory issue
            num_chunks = int(
                np.ceil(N * img_h * img_w * BYTES_PER_FLOAT / GPU_MEM_LIMIT))
            assert (num_chunks <= N), 'Default GPU_MEM_LIMIT is too small; try increasing it'
        chunks = torch.chunk(torch.arange(N, device=device), num_chunks)

        threshold = rcnn_test_cfg.mask_thr_binary
        im_mask = torch.zeros(
            N,
            img_h,
            img_w,
            device=device,
            dtype=torch.bool if threshold >= 0 else torch.uint8)

        if mask_pred.shape[1] > 1:
            mask_pred = mask_pred[range(N), labels][:, None]

        for inds in chunks:
            masks_chunk, spatial_inds = _do_paste_mask(
                mask_pred[inds],
                bboxes[inds],
                img_h,
                img_w,
                skip_empty=device.type == 'cpu')

            if threshold >= 0:
                masks_chunk = (masks_chunk >= threshold).to(dtype=torch.bool)
            else:
                # for visualization and debugging
                masks_chunk = (masks_chunk * 255).to(dtype=torch.uint8)

            im_mask[(inds, ) + spatial_inds] = masks_chunk

        im_segms = [im_mask[i].cpu().numpy() for i in range(N)]
        return im_segms


class SimpleSFMStage(nn.Module):

    def __init__(self,
                 semantic_in_channel=256,
                 semantic_out_channel=256,
                 instance_in_channel=256,
                 instance_out_channel=256,
                 fusion_type='MultiBranchFusion',
                 dilations=[1, 3, 5],
                 out_size=14,
                 num_classes=80,
                 semantic_out_stride=4,
                 upsample_cfg=dict(type='bilinear', scale_factor=2)):
        super(SimpleSFMStage, self).__init__()

        self.semantic_out_stride = semantic_out_stride
        self.num_classes = num_classes

        # for extracting instance-wise semantic feats
        self.semantic_transform_in = nn.Conv2d(semantic_in_channel, semantic_out_channel, 1)
        self.semantic_roi_extractor = SimpleRoIAlign(output_size=out_size, spatial_scale=1.0/semantic_out_stride)

        fuse_in_channel = instance_in_channel + semantic_out_channel + 1
        self.fuse_conv = nn.ModuleList([
            nn.Conv2d(fuse_in_channel, instance_in_channel, 1),
            globals()[fusion_type](instance_in_channel, dilations=dilations)])

        self.fuse_transform_out = nn.Conv2d(instance_in_channel, instance_out_channel - 1, 1)
        self.upsample = build_upsample_layer(upsample_cfg.copy())
        self.relu = nn.ReLU(inplace=True)

        self._init_weights()

    def _init_weights(self):
        for m in [self.semantic_transform_in, self.fuse_transform_out]:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)

        for m in self.fuse_conv:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, instance_feats, instance_logits, semantic_feat, rois, upsample=True):
        # instance-wise semantic feats
        semantic_feat = self.relu(self.semantic_transform_in(semantic_feat))
        ins_semantic_feats = self.semantic_roi_extractor(semantic_feat, rois)

        # fuse instance feats & instance masks & semantic feats
        concat_tensors = [instance_feats, ins_semantic_feats, instance_logits.sigmoid()]
        fused_feats = torch.cat(concat_tensors, dim=1)
        for conv in self.fuse_conv:
            fused_feats = self.relu(conv(fused_feats))

        # concat instance masks with fused feats again
        fused_feats = self.relu(self.fuse_transform_out(fused_feats))
        fused_feats = torch.cat([fused_feats, instance_logits.sigmoid()], dim=1)   # concat before upsampling, same as paper
        fused_feats = self.upsample(fused_feats) if upsample else fused_feats
        return fused_feats


@HEADS.register_module()
class SimpleRefineMaskHead(nn.Module):

    def __init__(self,
                 num_convs_instance=2,
                 num_convs_semantic=4,
                 conv_in_channels_instance=256,
                 conv_in_channels_semantic=256,
                 conv_kernel_size_instance=3,
                 conv_kernel_size_semantic=3,
                 conv_out_channels_instance=256,
                 conv_out_channels_semantic=256,
                 conv_cfg=None,
                 norm_cfg=None,
                 fusion_type='MultiBranchFusionAvg',
                 dilations=[1, 3, 5],
                 semantic_out_stride=4,
                 stage_num_classes=[80, 80, 80, 80],
                 stage_sup_size=[14, 28, 56, 112],
                 pre_upsample_last_stage=False,  # if True, upsample features and then compute logits
                 upsample_cfg=dict(type='bilinear', scale_factor=2),
                 loss_cfg=dict(
                    type='BARCrossEntropyLoss',
                    stage_instance_loss_weight=[0.25, 0.5, 0.75, 1.0],
                    boundary_width=2,
                    start_stage=1)):
        super(SimpleRefineMaskHead, self).__init__()

        self.num_convs_instance = num_convs_instance
        self.conv_kernel_size_instance = conv_kernel_size_instance
        self.conv_in_channels_instance = conv_in_channels_instance
        self.conv_out_channels_instance = conv_out_channels_instance

        self.num_convs_semantic = num_convs_semantic
        self.conv_kernel_size_semantic = conv_kernel_size_semantic
        self.conv_in_channels_semantic = conv_in_channels_semantic
        self.conv_out_channels_semantic = conv_out_channels_semantic

        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.semantic_out_stride = semantic_out_stride
        self.stage_sup_size = stage_sup_size
        self.stage_num_classes = stage_num_classes
        self.pre_upsample_last_stage = pre_upsample_last_stage

        self._build_conv_layer('instance')
        self._build_conv_layer('semantic')
        self.loss_func = build_loss(loss_cfg)

        assert len(self.stage_sup_size) > 1
        self.stages = nn.ModuleList()
        out_channel = conv_out_channels_instance
        stage_out_channels = [conv_out_channels_instance]
        for idx, out_size in enumerate(self.stage_sup_size[:-1]):
            in_channel = out_channel
            out_channel = in_channel // 2

            new_stage = SimpleSFMStage(
                semantic_in_channel=conv_out_channels_semantic,
                semantic_out_channel=in_channel,
                instance_in_channel=in_channel,
                instance_out_channel=out_channel,
                fusion_type=fusion_type,
                dilations=dilations,
                out_size=out_size,
                num_classes=self.stage_num_classes[idx],
                semantic_out_stride=semantic_out_stride,
                upsample_cfg=upsample_cfg)

            self.stages.append(new_stage)
            stage_out_channels.append(out_channel)

        self.stage_instance_logits = nn.ModuleList([
            nn.Conv2d(stage_out_channels[idx], num_classes, 1) for idx, num_classes in enumerate(self.stage_num_classes)])
        self.relu = nn.ReLU(inplace=True)

    def _build_conv_layer(self, name):
        out_channels = getattr(self, f'conv_out_channels_{name}')
        conv_kernel_size = getattr(self, f'conv_kernel_size_{name}')

        convs = []
        for i in range(getattr(self, f'num_convs_{name}')):
            in_channels = getattr(self, f'conv_in_channels_{name}') if i == 0 else out_channels
            conv = ConvModule(in_channels, out_channels, conv_kernel_size, dilation=1, padding=1)
            convs.append(conv)

        self.add_module(f'{name}_convs', nn.ModuleList(convs))

    def init_weights(self):
        for m in self.stage_instance_logits:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)

    def forward(self, instance_feats, semantic_feat, rois, roi_labels):
        for conv in self.instance_convs:
            instance_feats = conv(instance_feats)

        for conv in self.semantic_convs:
            semantic_feat = conv(semantic_feat)

        stage_instance_preds = []
        for idx, stage in enumerate(self.stages):
            instance_logits = self.stage_instance_logits[idx](instance_feats)[torch.arange(len(rois)), roi_labels][:, None]
            upsample_flag = self.pre_upsample_last_stage or idx < len(self.stages) - 1
            instance_feats = stage(instance_feats, instance_logits, semantic_feat, rois, upsample_flag)
            stage_instance_preds.append(instance_logits)

        # if use class-agnostic classifier for the last stage
        if self.stage_num_classes[-1] == 1:
            roi_labels = roi_labels.clamp(max=0)

        instance_preds = self.stage_instance_logits[-1](instance_feats)[torch.arange(len(rois)), roi_labels][:, None]
        if not self.pre_upsample_last_stage:
            instance_preds = F.interpolate(instance_preds, scale_factor=2, mode='bilinear', align_corners=True)
            instance_preds = instance_preds.to(semantic_feat.dtype)
        stage_instance_preds.append(instance_preds)

        return stage_instance_preds

    def get_targets(self, pos_bboxes_list, pos_assigned_gt_inds_list, gt_masks_list):

        def _generate_instance_targets(pos_proposals, pos_assigned_gt_inds, gt_masks, mask_size=None):
            device = pos_proposals.device
            proposals_np = pos_proposals.cpu().numpy()
            maxh, maxw = gt_masks.height, gt_masks.width
            proposals_np[:, [0, 2]] = np.clip(proposals_np[:, [0, 2]], 0, maxw)
            proposals_np[:, [1, 3]] = np.clip(proposals_np[:, [1, 3]], 0, maxh)
            pos_assigned_gt_inds = pos_assigned_gt_inds.cpu().numpy()

            # crop and resize the instance mask
            resize_masks = gt_masks.crop_and_resize(
                proposals_np, _pair(mask_size), inds=pos_assigned_gt_inds, device=device,).to_ndarray()
            instance_targets = torch.from_numpy(resize_masks).float().to(device)  # Tensor(Bitmaps)

            return instance_targets

        stage_instance_targets_list = [[] for _ in range(len(self.stage_sup_size))]
        for pos_bboxes, pos_assigned_gt_inds, gt_masks in zip(pos_bboxes_list, pos_assigned_gt_inds_list, gt_masks_list):
            stage_instance_targets = [_generate_instance_targets(
                pos_bboxes, pos_assigned_gt_inds, gt_masks, mask_size=mask_size) for mask_size in self.stage_sup_size]
            for stage_idx in range(len(self.stage_sup_size)):
                stage_instance_targets_list[stage_idx].append(stage_instance_targets[stage_idx])
        stage_instance_targets = [torch.cat(targets) for targets in stage_instance_targets_list]

        return stage_instance_targets

    def loss(self, stage_instance_preds, stage_instance_targets):
        loss_instance = self.loss_func(stage_instance_preds, stage_instance_targets)
        return dict(loss_instance=loss_instance)

    def get_seg_masks(self, mask_pred, det_bboxes, det_labels, rcnn_test_cfg, ori_shape, scale_factor, rescale):
        mask_pred = mask_pred.sigmoid()
        device = mask_pred[0].device
        bboxes = det_bboxes[:, :4]
        labels = det_labels

        if rescale:
            img_h, img_w = ori_shape[:2]
        else:
            img_h = np.round(ori_shape[0] * scale_factor).astype(np.int32)
            img_w = np.round(ori_shape[1] * scale_factor).astype(np.int32)
            scale_factor = 1.0

        if not isinstance(scale_factor, (float, torch.Tensor)):
            scale_factor = bboxes.new_tensor(scale_factor)
        bboxes = bboxes / scale_factor

        N = len(mask_pred)
        # The actual implementation split the input into chunks,
        # and paste them chunk by chunk.
        if device.type == 'cpu':
            # CPU is most efficient when they are pasted one by one with
            # skip_empty=True, so that it performs minimal number of
            # operations.
            num_chunks = N
        else:
            # GPU benefits from parallelism for larger chunks,
            # but may have memory issue
            num_chunks = int(
                np.ceil(N * img_h * img_w * BYTES_PER_FLOAT / GPU_MEM_LIMIT))
            assert (num_chunks <= N), 'Default GPU_MEM_LIMIT is too small; try increasing it'
        chunks = torch.chunk(torch.arange(N, device=device), num_chunks)

        threshold = rcnn_test_cfg.mask_thr_binary
        im_mask = torch.zeros(
            N,
            img_h,
            img_w,
            device=device,
            dtype=torch.bool if threshold >= 0 else torch.uint8)

        if mask_pred.shape[1] > 1:
            mask_pred = mask_pred[range(N), labels][:, None]

        for inds in chunks:
            masks_chunk, spatial_inds = _do_paste_mask(
                mask_pred[inds],
                bboxes[inds],
                img_h,
                img_w,
                skip_empty=device.type == 'cpu')

            if threshold >= 0:
                masks_chunk = (masks_chunk >= threshold).to(dtype=torch.bool)
            else:
                # for visualization and debugging
                masks_chunk = (masks_chunk * 255).to(dtype=torch.uint8)

            im_mask[(inds, ) + spatial_inds] = masks_chunk

        im_segms = [im_mask[i].cpu().numpy() for i in range(N)]
        return im_segms
