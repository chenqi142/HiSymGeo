# -*- coding:utf8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from .darknet import *
import torchvision.models as models

from .BiFPN import BiFPN
from .PALA import PALA
from .CrossViewMutilScaleFusionv import CrossViewMutilScaleFusion
from .osnet import ConvLayer, Conv1x1, Conv3x3, OSBlock

import pdb

class MyResnet(nn.Module):
    def __init__(self):
        super(MyResnet, self).__init__()
        self.base_model = models.resnet18(pretrained=True)
        self.base_model.avgpool = nn.Sequential()
        self.base_model.fc = nn.Sequential()

    def forward(self, x):
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)
        
        x = self.base_model.layer1(x) 
        # 打印原始模型的 layer1
        # print("Original layer1:")
        # print(self.base_model.layer1)
        # print(('x1', x.shape), flush=True) # torch.Size([B, 64, 64, 64])
        
        # self.base_model.layer2._modules['0'].conv1 = self._modify_conv_layer(
        #     self.base_model.layer2._modules['0'].conv1, 64 + 32, 128, kernel_size=3, stride=2, padding=1, bias=False
        # )
        # self.base_model.layer2._modules['0'].downsample[0] = self._modify_downsample_layer(
        #     self.base_model.layer2._modules['0'].downsample[0], 64 + 32, 128, kernel_size=1, stride=2, bias=False
        # )

        # x = torch.cat([x, x1], dim=1)


        x = self.base_model.layer2(x)
        # 打印原始模型的 layer2
        # print("Original layer2:")
        # print(self.base_model.layer2)
        # print(('x2', x.shape), flush=True) # torch.Size([B, 128, 32, 32])
        # 修改 layer3 的第一个卷积层
        
        x = self.base_model.layer3(x)
        # 打印原始模型的 layer3
        # print("Original layer3:")
        # print(self.base_model.layer3)
        # print(('x3', x.shape), flush=True) # torch.Size([B, 256, 16, 16])
        x = self.base_model.layer4(x)
        # 打印原始模型的 layer4
        # print("Original layer4:")
        # print(self.base_model.layer4)
        # print(('x4', x.shape), flush=True) # torch.Size([B, 512, 8, 8])
        return x
    
    def _modify_conv_layer(self, conv_layer, new_in_channels, out_channels, kernel_size, stride, padding, bias):
        # 获取原始权重
        w = conv_layer.weight
        # 替换卷积层
        new_conv = nn.Conv2d(new_in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        # 初始化新的权重
        w1 = torch.nn.Parameter(torch.empty(out_channels, new_in_channels - w.shape[1], kernel_size, kernel_size, device=w.device))
        nn.init.kaiming_normal_(w1, mode='fan_out', nonlinearity='relu')
        # 拼接权重
        new_conv.weight = torch.nn.Parameter(torch.cat((w, w1), dim=1))
        return new_conv
    
    # 修改 layer3 的下采样路径
    def _modify_downsample_layer(self, conv_layer, new_in_channels, out_channels, kernel_size, stride, bias):
        # 获取原始权重
        w = conv_layer.weight
        # 替换卷积层
        new_conv = nn.Conv2d(new_in_channels, out_channels, kernel_size=kernel_size, stride=stride, bias=bias)
        # 初始化新的权重
        w1 = torch.nn.Parameter(torch.empty(out_channels, new_in_channels - w.shape[1], kernel_size, kernel_size, device=w.device))
        nn.init.kaiming_normal_(w1, mode='fan_out', nonlinearity='relu')
        # 拼接权重
        new_conv.weight = torch.nn.Parameter(torch.cat((w, w1), dim=1))
        return new_conv


class CrossViewFusionModule(nn.Module):
    def __init__(self):
        super(CrossViewFusionModule, self).__init__()

    # normlized global_query:B, D
    # normlized value: B, D, H, W
    def forward(self, global_query, value):
        global_query = F.normalize(global_query, p=2, dim=-1)
        value = F.normalize(value, p=2, dim=1)

        B, D, W, H = value.shape
        new_value = value.permute(0, 2, 3, 1).view(B, W*H, D)
        score = torch.bmm(global_query.view(B, 1, D), new_value.transpose(1,2))
        score = score.view(B, W*H)
        with torch.no_grad():
            score_np = score.clone().detach().cpu().numpy()
            max_score, min_score = score_np.max(axis=1), score_np.min(axis=1)
        
        attn = torch.zeros(B, H*W).to(value.device)
        for ii in range(B):
            attn[ii, :] = (score[ii] - min_score[ii]) / (max_score[ii] - min_score[ii])
        
        attn = attn.view(B, 1, W, H)
        context = attn * value
        return context, attn
    
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
        self.conv1 = ConvLayer(in_channels, channels[0], 7, stride=2, padding=3, IN=IN)
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

class AttentionFusion(nn.Module):
    def __init__(self, in_channels, embed_dim):
        """
        in_channels: query_image 的通道数，例如 3。
        embed_dim: 位置编码的通道数，例如 1。
        """
        super(AttentionFusion, self).__init__()
        self.query_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.pos_proj = nn.Conv2d(embed_dim, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)  # 在通道维度上归一化

    def forward(self, query_image, pos_encoding):
        """
        query_image: [B, 3, H, W]
        pos_encoding: [B, 1, H, W]
        """
        # 特征投影
        query_features = self.query_proj(query_image)  # [B, 3, H, W]
        pos_features = self.pos_proj(pos_encoding)  # [B, 3, H, W]

        # 生成注意力权重
        attention_weights = self.softmax(pos_features)  # [B, 3, H, W]

        # 加权融合
        fused_features = query_features * attention_weights + query_image
        return fused_features
    
class ResidualConvFusion(nn.Module):
    def __init__(self, in_channels, out_channels, leaky=False, use_instnorm=False):
        """
        in_channels: 输入通道数（query_image 通道数 + 位置编码通道数）
        out_channels: 输出通道数（通常与 query_image 的通道数一致）
        """
        super(ResidualConvFusion, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels) if use_instnorm else nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True) if leaky else nn.ReLU(inplace=True)
        )

    def forward(self, query_image, pos_encoding):
        """
        query_image: [B, 3, H, W]
        pos_encoding: [B, 1, H, W]
        """
        fused_input = torch.cat((query_image, pos_encoding), dim=1)  # [B, 4, H, W]
        out = self.conv_block(fused_input)
        # 残差连接
        return out + query_image

class HisymGeo(nn.Module):
    def __init__(self, emb_size=512, leaky=False):
        super(HisymGeo, self).__init__()
        ## Visual model
        self.query_resnet = MyResnet()
        
        self.reference_darknet = Darknet(config_path='./model/yolov3_rs.cfg')
        self.reference_darknet.load_weights('./saved_models/yolov3.weights')

        #self.pose_subnet = Pose_Subnet([OSBlock, OSBlock], 1, [32, 32, 32], att_num=1, IN=False, matching_score_reg=False)



        self.bifpn = BiFPN(512)
        
        use_instnorm=False

        # self.combine_clickptns = ConvBatchNormReLU(4, 3, 1, 1, 0, 1, leaky=leaky, instance=use_instnorm)
        self.combine_clickptns = ResidualConvFusion(4, 3, leaky=leaky, use_instnorm=use_instnorm)
        self.crossview_fusionmodule = CrossViewMutilScaleFusion()
        # self.crossview_fusionmodule = CrossViewFusionModule()
		
        self.query_visudim = 512 
        self.reference_visudim = 512

        self.query_mapping_visu = ConvBatchNormReLU(self.query_visudim, emb_size, 1, 1, 0, 1, leaky=leaky, instance=use_instnorm)
        self.reference_mapping_visu1 = ConvBatchNormReLU(self.reference_visudim, emb_size, 1, 1, 0, 1, leaky=leaky, instance=use_instnorm)
        self.reference_mapping_visu2 = ConvBatchNormReLU(self.reference_visudim, emb_size, 1, 1, 0, 1, leaky=leaky, instance=use_instnorm)
        self.reference_mapping_visu3 = ConvBatchNormReLU(self.reference_visudim, emb_size, 1, 1, 0, 1, leaky=leaky, instance=use_instnorm)

        self.PALA = PALA(512)


        ## output head
        self.fcn_out = torch.nn.Sequential(
                ConvBatchNormReLU(emb_size, emb_size//2, 1, 1, 0, 1, leaky=leaky, instance=use_instnorm),
                nn.Conv2d(emb_size//2, 9*5, kernel_size=1))
        
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.lamda = torch.nn.Parameter(torch.tensor(1.0))
        self.alpha = nn.Parameter(torch.tensor([0.0, 0.0, 1.0]))  # 初始化为 0.0

    def forward(self, query_imgs, reference_imgs, mat_clickptns, training):
        mat_clickptns = mat_clickptns.unsqueeze(1) # torch.Size([B, 1, 256, 256])
        #x_, x1, x, onehot_index = self.pose_subnet(mat_clickptns)

        
        query_imgs = self.combine_clickptns(query_imgs, mat_clickptns)
        # query_imgs = self.combine_clickptns(torch.cat((query_imgs, mat_clickptns), dim=1))
        query_fvisu = self.query_resnet(query_imgs)
        query_fvisu = self.PALA(query_fvisu)

        
        reference_raw_fvisu = self.reference_darknet(reference_imgs)
        # BiFPN   FPN
        reference_raw_fvisu = self.bifpn(reference_raw_fvisu) 

        reference_fvisu1 = reference_raw_fvisu[0]
        reference_fvisu2 = reference_raw_fvisu[1]
        reference_fvisu3 = reference_raw_fvisu[2]

        query_fvisu = self.query_mapping_visu(query_fvisu)
        reference_fvisu1 = self.reference_mapping_visu1(reference_fvisu1)
        reference_fvisu2 = self.reference_mapping_visu2(reference_fvisu2)
        reference_fvisu3 = self.reference_mapping_visu3(reference_fvisu3)

        B, D, Hquery, Wquery = query_fvisu.shape
        B, D, Hreference1, Wreference1 = reference_fvisu1.shape
        B, D, Hreference2, Wreference2 = reference_fvisu2.shape
        B, D, Hreference3, Wreference3 = reference_fvisu3.shape

        # cross-view fusion
        query_gvisu = torch.mean(query_fvisu.view(B, D, Hquery*Wquery), dim=2, keepdims=False).view(B, D)
        reference_gvisu1 = torch.mean(reference_fvisu1.view(B, D, Hreference1*Wreference1), dim=2, keepdims=False).view(B, D)
        reference_gvisu2 = torch.mean(reference_fvisu2.view(B, D, Hreference2*Wreference2), dim=2, keepdims=False).view(B, D)
        reference_gvisu3 = torch.mean(reference_fvisu3.view(B, D, Hreference3*Wreference3), dim=2, keepdims=False).view(B, D)
        fused_features, attn_score = self.crossview_fusionmodule(query_gvisu, reference_raw_fvisu, training)
        # fused_features, attn_score = self.crossview_fusionmodule(query_gvisu, [reference_fvisu1, reference_fvisu2, reference_fvisu3], training)
        # fused_features, attn_score = self.crossview_fusionmodule(query_gvisu, reference_raw_fvisu[1])
        attn_score = attn_score.squeeze(1)

        outbox = self.fcn_out(fused_features)

        return outbox, attn_score, query_gvisu, reference_gvisu1, reference_gvisu2, reference_gvisu3
    