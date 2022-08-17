from turtle import forward
from mmdet.models.builder import HEADS, build_loss
from mmcv.runner import BaseModule, ModuleList, auto_fp16, force_fp32
from mmcv.runner import load_checkpoint

import torch
import torch.nn as nn

from .u2net import U2NET, U2TinyNET
from IPython import embed


@HEADS.register_module()
class FakeTestHead(BaseModule):
    def __init__(
        self,
        num_in_channels,
        num_classes,
        pretrain=None,
        ignore_index=255

    ):
        super(FakeTestHead,self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(num_in_channels, 256, 3, padding=1),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.Conv2d(256, 256, 3, padding=1),
        )
        self.out_list = nn.ModuleList([
            nn.Conv2d(256, num_classes+1, 1),
            nn.Conv2d(256, num_classes+1, 1),
            nn.Conv2d(256, num_classes+1, 1),
            nn.Conv2d(256, num_classes+1, 1),
            nn.Conv2d(256, num_classes+1, 1),
            nn.Conv2d(256, num_classes+1, 1),
            nn.Conv2d(256, num_classes+1, 1),
        ])
        self.ce_loss = nn.CrossEntropyLoss(size_average=True, ignore_index=ignore_index)

    @auto_fp16()
    def forward(self, x, mode='val'):
        x = self.conv(x)
        mask_pred_all = [self.out_list[idx](x) for idx in range(len(self.out_list))]

        if mode == 'tra':
            res = mask_pred_all
        else:
            res = mask_pred_all[0]
        return res

    @force_fp32(apply_to=('mask_pred', ))
    def loss(self, mask_pred_all, mask_targets):
        """
        """
        # mask_pred = mask_pred_all
        loss = self.muti_ce_loss_fusion(mask_pred_all, mask_targets)
        return loss

    def muti_ce_loss_fusion(self, mask_pred_all, labels_v):
        bs, n, h, w = mask_pred_all[0].shape
        d0, d1, d2, d3, d4, d5, d6 = mask_pred_all
        labels_v = labels_v.reshape([-1, ]).to(torch.long)
        loss0 = self.ce_loss(d0.permute([0,2,3,1]).reshape([-1, n]),labels_v)
        loss1 = self.ce_loss(d1.permute([0,2,3,1]).reshape([-1, n]),labels_v)
        loss2 = self.ce_loss(d2.permute([0,2,3,1]).reshape([-1, n]),labels_v)
        loss3 = self.ce_loss(d3.permute([0,2,3,1]).reshape([-1, n]),labels_v)
        loss4 = self.ce_loss(d4.permute([0,2,3,1]).reshape([-1, n]),labels_v)
        loss5 = self.ce_loss(d5.permute([0,2,3,1]).reshape([-1, n]),labels_v)
        loss6 = self.ce_loss(d6.permute([0,2,3,1]).reshape([-1, n]),labels_v)
        print(loss0, loss1, loss2, loss3, loss4, loss5, loss6)
        loss = (loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6) / 7
        return loss        



@HEADS.register_module()
class SemanticU2Head(BaseModule):
    def __init__(
        self,
        num_in_channels,
        num_classes,
        pretrain=None,
        ignore_index=255,
        net='U2NET',
    ):
        super(SemanticU2Head,self).__init__()
        if net == 'U2NET':
            self.u2net = U2NET(
                in_ch=num_in_channels,
                out_ch=num_classes + 1,
            )
        elif net == 'U2TinyNET':
            self.u2net = U2TinyNET(
                in_ch=num_in_channels,
                out_ch=num_classes + 1,
            )

        self.num_classes = num_classes
        if pretrain is not None:
            load_checkpoint(self.u2net, pretrain, map_location='cpu')
        self.ce_loss = nn.CrossEntropyLoss(size_average=True, ignore_index=ignore_index)

    def init_weights(self):
        pass

    @auto_fp16()
    def forward(self, x, mode='val'):
        mask_pred_all = self.u2net(x)
        if mode == 'tra':
            res = mask_pred_all
        else:
            res = mask_pred_all[0]
        return res

    @force_fp32(apply_to=('mask_pred', ))
    def loss(self, mask_pred_all, mask_targets):
        """
        """
        # mask_pred = mask_pred_all
        loss = self.muti_ce_loss_fusion(mask_pred_all, mask_targets)
        return loss

    def muti_ce_loss_fusion(self, mask_pred_all, labels_v):
        bs, n, h, w = mask_pred_all[0].shape
        labels_v = labels_v.reshape([-1, ]).to(torch.long)
        labels_v[labels_v==255] = self.num_classes
        loss = 0
        for d in mask_pred_all:
            l = self.ce_loss(d.permute([0,2,3,1]).reshape([-1, n]),labels_v)
            loss += l
        loss = loss / (len(mask_pred_all) + 1e-8)
        return loss

