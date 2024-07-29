import torch.nn as nn
import torch.nn.functional as F

from models.head import SegformerHead, UnetHead
from models.backbone import DeformMixVisionMamba


class DefromMambaUnet(nn.Module):
    def __init__(self, config):
        super(DefromMambaUnet, self).__init__()

        self.backbone = DeformMixVisionMamba(**config.MODEL.backbone)
        self.decode_head = UnetHead(**config.MODEL.head)
        # self.decode_head = SegformerHead(**config.MODEL.head)

    def forward(self, inputs):
        H, W = inputs.size(2), inputs.size(3)
        x = self.backbone.forward(inputs)
        x = self.decode_head.forward(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x


class DefromMambaSegformer(nn.Module):
    def __init__(self, config):
        super(DefromMambaSegformer, self).__init__()

        self.backbone = DeformMixVisionMamba(**config.MODEL.backbone)
        self.decode_head = SegformerHead(**config.MODEL.head)

    def forward(self, inputs):
        H, W = inputs.size(2), inputs.size(3)
        x = self.backbone.forward(inputs)
        x = self.decode_head.forward(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x
