import torch
import torch.nn as nn
import torch.nn.functional as F
from models.module import DoubleConv

from models.backbone import DeformMixVisionMamba

# Ablation Study
from models.backbone import MixVisionMamba, DeformMixVision, ParallelDeformMixVisionMamba, MambaMixVisionDeform


class Decoder(nn.Module):
    def __init__(self, inchannels, num_classes, channels=256, interpolate_mode='bilinear', dropout_ratio=0.1):
        super().__init__()

        self.in_channels = inchannels
        self.channels = channels
        self.interpolate_mode = interpolate_mode
        num_inputs = len(self.in_channels)

        self.up1 = nn.ConvTranspose2d(256, 160, 2, stride=2)
        self.conv1 = DoubleConv(320, 160)

        self.up2 = nn.ConvTranspose2d(160, 64, 2, stride=2)
        self.conv2 = DoubleConv(128, 64)

        self.up3 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv3 = DoubleConv(64, 32)

        self.up4 = nn.ConvTranspose2d(32, 64, 2, stride=2)
        self.conv4 = DoubleConv(64, 64)

        self.cls_seg = nn.Conv2d(64, num_classes, 1)

    def forward(self, inputs):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        inputs = inputs[::-1]
        x1, x2, x3, x4 = inputs

        up1 = self.up1(x1)
        merge1 = torch.concat([up1, x2], dim=1)
        conv1 = self.conv1(merge1)

        up2 = self.up2(conv1)
        merge2 = torch.concat([up2, x3], dim=1)
        conv2 = self.conv2(merge2)

        up3 = self.up3(conv2)
        merge3 = torch.concat([up3, x4], dim=1)
        conv3 = self.conv3(merge3)

        up4 = self.up4(conv3)
        conv4 = self.conv4(up4)

        out = self.cls_seg(conv4)
        return out


class UVMamba(nn.Module):
    def __init__(self, config):
        super(UVMamba, self).__init__()

        self.backbone = DeformMixVisionMamba(**config.MODEL.backbone)
        self.decode_head = Decoder(**config.MODEL.head)

    def forward(self, inputs):
        H, W = inputs.size(2), inputs.size(3)
        x = self.backbone.forward(inputs)
        x = self.decode_head.forward(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x



# ====================================
# Ablation 1: without SADE
# ====================================

class UVMambaNoDeform(nn.Module):
    def __init__(self, config):
        super(UVMambaNoDeform, self).__init__()

        self.backbone = MixVisionMamba(**config.MODEL.backbone)
        self.decode_head = Decoder(**config.MODEL.head)

    def forward(self, inputs):
        H, W = inputs.size(2), inputs.size(3)
        x = self.backbone.forward(inputs)
        x = self.decode_head.forward(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x


# ====================================
# Ablation 2: without MSSM
# ====================================

class UVMambaNoSSM(nn.Module):
    def __init__(self, config):
        super(UVMambaNoSSM, self).__init__()

        self.backbone = DeformMixVision(**config.MODEL.backbone)
        self.decode_head = Decoder(**config.MODEL.head)

    def forward(self, inputs):
        H, W = inputs.size(2), inputs.size(3)
        x = self.backbone.forward(inputs)
        x = self.decode_head.forward(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x

# ==================================================
# Ablation 3: SADE and MSSM are arranged parallel
# ==================================================


class UVMambaParallel(nn.Module):
    def __init__(self, config):
        super(UVMambaParallel, self).__init__()

        self.backbone = ParallelDeformMixVisionMamba(**config.MODEL.backbone)
        self.decode_head = Decoder(**config.MODEL.head)

    def forward(self, inputs):
        H, W = inputs.size(2), inputs.size(3)
        x = self.backbone.forward(inputs)
        x = self.decode_head.forward(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x

# ====================================
# Ablation 4: SSM --> DCN
# ====================================


class UVMambaReverse(nn.Module):
    def __init__(self, config):
        super(UVMambaReverse, self).__init__()

        self.backbone = MambaMixVisionDeform(**config.MODEL.backbone)
        self.decode_head = Decoder(**config.MODEL.head)

    def forward(self, inputs):
        H, W = inputs.size(2), inputs.size(3)
        x = self.backbone.forward(inputs)
        x = self.decode_head.forward(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x

