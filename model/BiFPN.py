import torch
import torch.nn as nn
import torch.nn.functional as F
from pyexpat import features




class BiFPNLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BiFPNLayer, self).__init__()
        self.out_channels = out_channels


        self.weights_up = nn.ParameterList([
            nn.Parameter(torch.ones(2, dtype=torch.float32)),
            nn.Parameter(torch.ones(2, dtype=torch.float32))
        ])

        self.weights_down = nn.ParameterList([
            nn.Parameter(torch.ones(3, dtype=torch.float32)),
            nn.Parameter(torch.ones(2, dtype=torch.float32))
        ])

    def forward(self, features):
        p5, p4, p3 = features


        p4_up = self._fuse_features(p4, F.interpolate(p5, scale_factor=2, mode='nearest'), weights=self.weights_up[0])
        p3_up = self._fuse_features(p3, F.interpolate(p4, scale_factor=2, mode='nearest'), weights=self.weights_up[1])

        p4_down = self._fuse_features(p4, p4_up, F.max_pool2d(p3_up, kernel_size=2, stride=2), weights=self.weights_down[0])
        p5_down = self._fuse_features(p5, F.max_pool2d(p4_down, kernel_size=2, stride=2), weights=self.weights_down[1])

        return p5_down, p4_down, p3_up


    def _fuse_features(self, *features, weights):
        # Weighted feature fusion
        weights = F.softmax(weights, dim=0)
        fused_feature = torch.zeros_like(features[0])
        for i, feature in enumerate(features):
            fused_feature += weights[i] * feature
        return fused_feature

class BiFPN(nn.Module):
    def __init__(self, out_channels, num_layers=3):
        super(BiFPN, self).__init__()
        self.num_layers = num_layers

        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(1024, out_channels, kernel_size=1),
            nn.Conv2d(512, out_channels, kernel_size=1),
            nn.Conv2d(256, out_channels, kernel_size=1)
        ])


        self.bifpn_layers = nn.ModuleList([
            BiFPNLayer(out_channels, out_channels) for _ in range(num_layers)
        ])


    def forward(self, features):
        p3, p4, p5 = [conv(feature) for conv, feature in zip(self.lateral_convs, features)]

        for bifpn_layer in self.bifpn_layers:
            p3, p4, p5 = bifpn_layer([p3, p4, p5])

        return p3, p4, p5
    