import torch
import torch.nn as nn
from osnet import ConvLayer, Conv1x1, Conv3x3, OSBlock
import numpy as np

class score_att(nn.Module):
    """1x1 convolution + bn + relu."""

    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super(score_att, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=stride, padding=0,
                              bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.Sigmoid()
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class Pose_Subnet(nn.Module):

    def __init__(self, blocks, in_channels, channels, att_num=1, IN=False, matching_score_reg=False):
        super(Pose_Subnet, self).__init__()
        num_blocks = len(blocks)
        self.conv1 = ConvLayer(in_channels, channels[0], 7, stride=1, padding=3, IN=IN)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = self._make_layer(blocks[0], 1, channels[0], channels[1], reduce_spatial_size=True)
        self.conv3 = self._make_layer(blocks[1], 1, channels[1], channels[2], reduce_spatial_size=False)
        self.conv4 = Conv3x3(channels[2], channels[2])
        self.conv_out = score_att(channels[2], att_num)
        self._init_params()

    def _make_layer(self, block, layer, in_channels, out_channels, reduce_spatial_size, IN=False):
        layers = []
        layers.append(block(in_channels, out_channels, IN=IN, gate_reduction=4))
        for i in range(1, layer):
            layers.append(block(out_channels, out_channels, IN=IN, gate_reduction=4))

        if reduce_spatial_size:
            layers.append(
                nn.Sequential(
                    Conv1x1(out_channels, out_channels),
                    nn.AvgPool2d(2, stride=2)
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)#64*56*72*36
        x_ = self.maxpool(x)#64*32*72*36
        x = self.conv2(x_)#64*32*36*18
        x = self.conv3(x)#64*32*18*9
        x1 = self.conv4(x)#64*32*18*9

        x = self.conv_out(x1)
        _, max_index = x.max(dim=1, keepdim=True)
        onehot_index = torch.zeros_like(x).scatter_(1, max_index, 1)
        return x_, x1, x, onehot_index


    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)