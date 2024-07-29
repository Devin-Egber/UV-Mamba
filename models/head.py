import torch
import torch.nn as nn
from models.utils import resize


class SegformerHead(nn.Module):

    def __init__(self, inchannels, num_classes, channels=256, interpolate_mode='bilinear', dropout_ratio=0.1):
        super().__init__()

        self.in_channels = inchannels
        self.channels = channels
        self.interpolate_mode = interpolate_mode
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
        self.dropout = nn.Dropout2d(dropout_ratio)

        self.cls_seg = nn.Conv2d(self.channels, num_classes, kernel_size=1)

    def forward(self, inputs):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        # inputs = self._transform_inputs(inputs)
        outs = []
        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs[idx]
            outs.append(
                resize(
                    input=conv(x),
                    size=inputs[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=False))

        out = self.fusion_conv(torch.cat(outs, dim=1))
        out = self.dropout(out)
        out = self.cls_seg(out)
        return out


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1),
                                  nn.BatchNorm2d(out_channels),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(out_channels, out_channels, 3, padding=1),
                                  nn.BatchNorm2d(out_channels),
                                  nn.ReLU(inplace=True))

    def forward(self, input):
        return self.conv(input)


class UnetHead(nn.Module):
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


