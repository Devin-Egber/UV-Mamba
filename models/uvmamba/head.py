import torch
import torch.nn as nn
from .utils import resize


class SegformerHead(nn.Module):

    def __init__(self, inchannels, num_classes, channels=256):
        super().__init__()

        self.in_channels = inchannels
        self.channels = channels
        num_inputs = len(self.in_channels)

        self.convs = nn.ModuleList()
        for i in range(num_inputs):
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=self.in_channels[i],
                        out_channels=self.channels,
                        kernel_size=1,
                        stride=1),
                    nn.BatchNorm2d(self.channels),
                    nn.ReLU()
                )
            )

        self.fusion_conv = nn.Sequential(
                nn.Conv2d(in_channels=self.channels * num_inputs, out_channels=self.channels, kernel_size=1),
                nn.BatchNorm2d(self.channels),
                nn.ReLU()
        )

        self.cls_seg = nn.Conv2d(self.channels * num_inputs, num_classes, kernel_size=1)

    def forward(self, inputs):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        inputs = self._transform_inputs(inputs)
        outs = []
        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs[idx]
            outs.append(
                resize(
                    input=conv(x),
                    size=inputs[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))

        out = self.fusion_conv(torch.cat(outs, dim=1))

        out = self.cls_seg(out)

        return out
