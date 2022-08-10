import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init

from ..builder import NECKS
from .fpn import FPN
from .. import builder
from .fpg import FPG
from .fpn_carafe import FPN_CARAFE

@NECKS.register_module()
class CBFPN(FPN):
    '''
    FPN with weight sharing
    which support mutliple outputs from cbnet
    '''
    def forward(self, inputs):
        if not isinstance(inputs[0], (list, tuple)):
            inputs = [inputs]
            
        if self.training:
            outs = []
            for x in inputs:
                out = super().forward(x)
                outs.append(out)
            return outs
        else:
            out = super().forward(inputs[-1])
            return out


@NECKS.register_module()
class CBFPG(FPG):
    '''
    FPG with weight sharing
    which support mutliple outputs from cbnet
    '''
    def forward(self, inputs):
        if not isinstance(inputs[0], (list, tuple)):
            inputs = [inputs]
            
        if self.training:
            outs = []
            for x in inputs:
                out = super().forward(x)
                outs.append(out)
            return outs
        else:
            out = super().forward(inputs[-1])
            return out


@NECKS.register_module()
class CBFPN_CARAFE(FPN_CARAFE):
    '''
    FPN_CARAFE with weight sharing
    which support mutliple outputs from cbnet
    '''
    def forward(self, inputs):
        if not isinstance(inputs[0], (list, tuple)):
            inputs = [inputs]
            
        if self.training:
            outs = []
            for x in inputs:
                out = super().forward(x)
                outs.append(out)
            return outs
        else:
            out = super().forward(inputs[-1])
            return out

