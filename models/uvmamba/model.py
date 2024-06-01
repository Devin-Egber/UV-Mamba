import torch
import torch.nn as nn
import torch.nn.functional as F

from models.uvmamba.head import SegformerHead
from models.uvmamba.backbone import MixVisionMamba, DeformMixVisionMamba

from models.uvmamba.back2 import MixVisionSamba

class UVMamba(nn.Module):
    def __init__(self, config):
        super(UVMamba, self).__init__()

        self.backbone = MixVisionMamba(**config.MODEL.backbone)
        self.decode_head = SegformerHead(**config.MODEL.head)

    def forward(self, inputs):
        H, W = inputs.size(2), inputs.size(3)
        x = self.backbone.forward(inputs)
        x = self.decode_head.forward(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x


class DefromUVMamba(nn.Module):
    def __init__(self, config):
        super(DefromUVMamba, self).__init__()

        self.backbone = DeformMixVisionMamba(**config.MODEL.backbone)
        self.decode_head = SegformerHead(**config.MODEL.head)

    def forward(self, inputs):
        H, W = inputs.size(2), inputs.size(3)
        x = self.backbone.forward(inputs)
        x = self.decode_head.forward(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x


class Samba(nn.Module):
    def __init__(self, config):
        super(Samba, self).__init__()

        self.backbone = MixVisionSamba(**config.MODEL.backbone)
        self.decode_head = SegformerHead(**config.MODEL.head)

    def forward(self, inputs):
        H, W = inputs.size(2), inputs.size(3)
        x = self.backbone.forward(inputs)
        x = self.decode_head.forward(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x




