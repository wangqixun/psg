from mmcv.cnn import ConvModule

from mmdet.models.builder import HEADS
from .fcn_mask_head import FCNMaskHead
from .u2_mask_head import U2MaskSimpleLossHead


@HEADS.register_module()
class HTCMaskHead(FCNMaskHead):

    def __init__(self, with_conv_res=True, *args, **kwargs):
        super(HTCMaskHead, self).__init__(*args, **kwargs)
        self.with_conv_res = with_conv_res
        if self.with_conv_res:
            self.conv_res = ConvModule(
                self.conv_out_channels,
                self.conv_out_channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg)

    def forward(self, x, res_feat=None, return_logits=True, return_feat=True):
        if res_feat is not None:
            assert self.with_conv_res
            res_feat = self.conv_res(res_feat)
            x = x + res_feat
        for conv in self.convs:
            x = conv(x)
        res_feat = x
        outs = []
        if return_logits:
            x = self.upsample(x)
            if self.upsample_method == 'deconv':
                x = self.relu(x)
            mask_pred = self.conv_logits(x)
            outs.append(mask_pred)
        if return_feat:
            outs.append(res_feat)
        return outs if len(outs) > 1 else outs[0]

@HEADS.register_module()
class HTCU2MaskHead(U2MaskSimpleLossHead):

    def __init__(self, with_conv_res=True, *args, **kwargs):
        super(HTCU2MaskHead, self).__init__(*args, **kwargs)
        self.with_conv_res = with_conv_res
        if self.with_conv_res:
            self.conv_res = ConvModule(
                1 if self.class_agnostic else self.num_classes,
                self.in_channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg)

    def forward(self, x, res_feat=None, return_logits=True, return_feat=True):
        if res_feat is not None:
            assert self.with_conv_res
            if 0 not in res_feat.shape:
                res_feat = self.conv_res(res_feat)
                x = x + res_feat
        # for conv in self.convs:
        #     x = conv(x)
        # res_feat = x
        mask_pred, mask_pred_all = self.u2net(x)
        res_feat = mask_pred
        
        outs = []
        if return_logits:
            # x = self.upsample(x)
            # if self.upsample_method == 'deconv':
            #     x = self.relu(x)
            # mask_pred = self.conv_logits(x)
            outs.append(mask_pred)
        if return_feat:
            outs.append(res_feat)
        return outs if len(outs) > 1 else outs[0]
