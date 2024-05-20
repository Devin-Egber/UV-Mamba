import torch.nn as nn
from .head import SegformerHead
from .backbone import MixVisionTransformer


class UVMamba(nn.Module):
    def __init__(self, config):
        super(UVMamba, self).__init__()

        self.backbone = MixVisionTransformer(**config.MODEL.backbone)
        self.decode_head = SegformerHead(**config.MODEL.head)

    def forward(self, inputs):
        H, W = inputs.size(2), inputs.size(3)

        x = self.backbone.forward(inputs)
        x = self.decode_head.forward(x)
        return x