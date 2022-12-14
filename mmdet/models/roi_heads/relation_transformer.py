from turtle import forward
from mmdet.models.builder import HEADS, build_loss
from mmcv.runner import BaseModule, ModuleList, auto_fp16, force_fp32
from mmcv.runner import load_checkpoint

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import BertModel, BertTokenizer

from mmdet.models.losses import accuracy


from IPython import embed


@HEADS.register_module()
class BertTransformer(BaseModule):
    def __init__(
        self, 
        pretrained_transformers='/share/wangqixun/workspace/bs/tx_mm/code/model_dl/hfl/chinese-roberta-wwm-ext', 
        cache_dir='/share/wangqixun/workspace/bs/psg/psg/tmp',
        input_feature_size=256,
        layers_transformers=6,
        feature_size=768,
        num_cls=56,
        cls_qk_size=512,
        loss_weight=1.,
        num_entity_max=30,
    ):
        super().__init__()
        self.num_cls = num_cls
        self.cls_qk_size = cls_qk_size
        self.fc_input = nn.Sequential(
            nn.Linear(input_feature_size, feature_size),
            nn.LayerNorm(feature_size),
        )
        self.fc_output = nn.Sequential(
            nn.Linear(feature_size, feature_size),
            nn.LayerNorm(feature_size),
        )
        self.model = AutoModel.from_pretrained(pretrained_transformers, cache_dir=cache_dir)
        self.cls_q = nn.Linear(feature_size, cls_qk_size * num_cls)
        self.cls_k = nn.Linear(feature_size, cls_qk_size * num_cls)
        self.model.encoder.layer = self.model.encoder.layer[:layers_transformers]
        self.loss_weight = loss_weight
        self.feature_size = feature_size
        self.input_feature_size = input_feature_size
        self.num_entity_max = num_entity_max


    def forward(self,inputs_embeds, attention_mask=None):
        position_ids = torch.ones([1, inputs_embeds.shape[1]]).to(inputs_embeds.device).to(torch.long)
        encode_inputs_embeds = self.fc_input(inputs_embeds)
        encode_res = self.model(inputs_embeds=encode_inputs_embeds, attention_mask=attention_mask, position_ids=position_ids)
        encode_embedding = encode_res['last_hidden_state']
        encode_embedding = self.fc_output(encode_embedding)
        bs, N, c = encode_embedding.shape
        q_embedding = self.cls_q(encode_embedding).reshape([bs, N, self.num_cls, self.cls_qk_size]).permute([0,2,1,3])
        k_embedding = self.cls_k(encode_embedding).reshape([bs, N, self.num_cls, self.cls_qk_size]).permute([0,2,1,3])
        cls_pred = q_embedding @ torch.transpose(k_embedding, 2, 3) / self.cls_qk_size ** 0.5
        return cls_pred

    def get_f1_p_r(self, y_pred, y_true, mask_attention, th=0):
        # y_pred     [bs, 56, N, N]
        # y_true     [bs, 56, N, N]
        # mask_attention   [bs, 56, N, N]
        res = []
        
        y_pred[y_pred > th] = 1
        y_pred[y_pred < th] = 0

        n1 = y_pred * y_true * mask_attention
        n2 = y_pred * mask_attention
        n3 = y_true * mask_attention

        p = 100 * n1.sum(dim=[1,2,3]) / (1e-8 + n2.sum(dim=[1,2,3]))
        r = 100 * n1.sum(dim=[1,2,3]) / (1e-8 + n3.sum(dim=[1,2,3]))
        f1 = 2 * p * r / (p + r + 1e-8)
        res.append([f1.mean(), p.mean(), r.mean()])

        mask_mean = y_true.sum(dim=[0, 2, 3]) > 0
        p = 100 * n1.sum(dim=[0,2,3]) / (1e-8 + n2.sum(dim=[0,2,3]))
        r = 100 * n1.sum(dim=[0,2,3]) / (1e-8 + n3.sum(dim=[0,2,3]))
        f1 = 2 * p * r / (p + r + 1e-8)
        res.append([
            torch.sum(f1 * mask_mean) / (torch.sum(mask_mean) + 1e-8),
            torch.sum(p * mask_mean) / (torch.sum(mask_mean) + 1e-8),
            torch.sum(r * mask_mean) / (torch.sum(mask_mean) + 1e-8),
        ])

        return res


    def loss(self, pred, target, mask_attention):
        # pred     [bs, 56, N, N]
        # target   [bs, 56, N, N]
        # mask_attention   [bs, N]
        losses = {}
        bs, nb_cls, N, N = pred.shape
        
        mask = torch.zeros_like(pred).to(pred.device)
        for idx in range(bs):
            n = torch.sum(mask_attention[idx]).to(torch.int)
            mask[idx, :, :n, :n] = 1
        pred = pred * mask - 9999 * (1 - mask)

        loss = self.multilabel_categorical_crossentropy(target.reshape([bs*nb_cls, -1]), pred.reshape([bs*nb_cls, -1]))
        loss = loss.mean()
        losses['loss_relationship'] = loss * self.loss_weight

        # f1, p, r
        [f1, precise, recall], [f1_mean, precise_mean, recall_mean] = self.get_f1_p_r(pred, target, mask)
        losses['rela.F1'] = f1
        losses['rela.precise'] = precise
        losses['rela.recall'] = recall
        losses['rela.F1_mean'] = f1_mean
        losses['rela.precise_mean'] = precise_mean
        losses['rela.recall_mean'] = recall_mean

        return losses


    def multilabel_categorical_crossentropy(self, y_true, y_pred):
        """???????????????????????????
        ?????????y_true???y_pred???shape?????????y_true????????????0???1???
            1?????????????????????????????????0????????????????????????????????????
        ??????????????????y_pred???????????????????????????????????????????????????y_pred
            ??????????????????????????????????????????sigmoid??????softmax?????????
            ???????????????y_pred??????0?????????
        """
        y_pred = (1 - 2 * y_true) * y_pred
        y_pred_neg = y_pred - y_true * 9999
        y_pred_pos = y_pred - (1 - y_true) * 9999
        zeros = torch.zeros_like(y_pred[..., :1])
        y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
        y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
        return neg_loss + pos_loss



@HEADS.register_module()
class MultiHeadCls(BaseModule):
    def __init__(
        self, 
        pretrained_transformers=None, 
        cache_dir=None,
        input_feature_size=256,
        layers_transformers=6,
        feature_size=768,
        num_cls=56,
        cls_qk_size=512,
        loss_weight=1.,
        num_entity_max=30,
    ):
        super().__init__()
        self.num_cls = num_cls
        self.cls_qk_size = cls_qk_size
        self.fc_input = nn.Sequential(
            nn.Linear(input_feature_size, feature_size),
            nn.LayerNorm(feature_size),
        )
        self.cls_q = nn.Linear(feature_size, cls_qk_size * num_cls)
        self.cls_k = nn.Linear(feature_size, cls_qk_size * num_cls)
        self.loss_weight = loss_weight
        self.feature_size = feature_size
        self.input_feature_size = input_feature_size
        self.num_entity_max = num_entity_max


    def forward(self,inputs_embeds, attention_mask=None):
        encode_embedding = self.fc_input(inputs_embeds)
        bs, N, c = encode_embedding.shape
        q_embedding = self.cls_q(encode_embedding).reshape([bs, N, self.num_cls, self.cls_qk_size]).permute([0,2,1,3])
        k_embedding = self.cls_k(encode_embedding).reshape([bs, N, self.num_cls, self.cls_qk_size]).permute([0,2,1,3])
        cls_pred = q_embedding @ torch.transpose(k_embedding, 2, 3) / self.cls_qk_size ** 0.5
        return cls_pred


    def get_f1_p_r(self, y_pred, y_true, mask_attention, th=0):
        # y_pred     [bs, 56, N, N]
        # y_true     [bs, 56, N, N]
        # mask_attention   [bs, 56, N, N]
        res = []
        
        y_pred[y_pred > th] = 1
        y_pred[y_pred < th] = 0

        n1 = y_pred * y_true * mask_attention
        n2 = y_pred * mask_attention
        n3 = y_true * mask_attention

        p = 100 * n1.sum(dim=[1,2,3]) / (1e-8 + n2.sum(dim=[1,2,3]))
        r = 100 * n1.sum(dim=[1,2,3]) / (1e-8 + n3.sum(dim=[1,2,3]))
        f1 = 2 * p * r / (p + r + 1e-8)
        res.append([f1.mean(), p.mean(), r.mean()])

        mask_mean = y_true.sum(dim=[0, 2, 3]) > 0
        p = 100 * n1.sum(dim=[0,2,3]) / (1e-8 + n2.sum(dim=[0,2,3]))
        r = 100 * n1.sum(dim=[0,2,3]) / (1e-8 + n3.sum(dim=[0,2,3]))
        f1 = 2 * p * r / (p + r + 1e-8)
        res.append([
            torch.sum(f1 * mask_mean) / (torch.sum(mask_mean) + 1e-8),
            torch.sum(p * mask_mean) / (torch.sum(mask_mean) + 1e-8),
            torch.sum(r * mask_mean) / (torch.sum(mask_mean) + 1e-8),
        ])

        return res


    def loss(self, pred, target, mask_attention):
        # pred     [bs, 56, N, N]
        # target   [bs, 56, N, N]
        # mask_attention   [bs, N]
        losses = {}
        bs, nb_cls, N, N = pred.shape
        
        mask = torch.zeros_like(pred).to(pred.device)
        for idx in range(bs):
            n = torch.sum(mask_attention[idx]).to(torch.int)
            mask[idx, :, :n, :n] = 1
        pred = pred * mask - 9999 * (1 - mask)

        loss = self.multilabel_categorical_crossentropy(target.reshape([bs*nb_cls, -1]), pred.reshape([bs*nb_cls, -1]))
        loss = loss.mean()
        losses['loss_relationship'] = loss * self.loss_weight

        # f1, p, r
        [f1, precise, recall], [f1_mean, precise_mean, recall_mean] = self.get_f1_p_r(pred, target, mask)
        losses['rela.F1'] = f1
        losses['rela.precise'] = precise
        losses['rela.recall'] = recall
        losses['rela.F1_mean'] = f1_mean
        losses['rela.precise_mean'] = precise_mean
        losses['rela.recall_mean'] = recall_mean

        return losses


    def multilabel_categorical_crossentropy(self, y_true, y_pred):
        """???????????????????????????
        ?????????y_true???y_pred???shape?????????y_true????????????0???1???
            1?????????????????????????????????0????????????????????????????????????
        ??????????????????y_pred???????????????????????????????????????????????????y_pred
            ??????????????????????????????????????????sigmoid??????softmax?????????
            ???????????????y_pred??????0?????????
        """
        y_pred = (1 - 2 * y_true) * y_pred
        y_pred_neg = y_pred - y_true * 9999
        y_pred_pos = y_pred - (1 - y_true) * 9999
        zeros = torch.zeros_like(y_pred[..., :1])
        y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
        y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
        return neg_loss + pos_loss





