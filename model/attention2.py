import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb


class SoftPooling2D(torch.nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        # 调用父类构造函数初始化网络层
        super(SoftPooling2D, self).__init__()
        # 定义平均池化操作，不计入填充部分的计算
        self.avgpool = torch.nn.AvgPool2d(kernel_size, stride, padding, count_include_pad=False)
        self.eps = 1e-6  # 防止除零错误

    def forward(self, x):
        # 对输入张量应用指数运算，减去最大值以稳定梯度
        x_max = x.max(dim=1, keepdim=True).values  # 沿通道维度取最大值
        x_exp = torch.exp(x - x_max)
        # 对指数处理后的张量进行平均池化
        x_exp_pool = self.avgpool(x_exp)
        # 对输入与指数结果相乘后进行平均池化
        x = self.avgpool(x_exp * x)
        # 返回最终的加权平均结果
        return x / (x_exp_pool + self.eps)  # 防止除零错误


class LocalAttention(nn.Module):
    def __init__(self, channels, f=128):
        # 初始化模块，设置通道数和中间层维度
        super().__init__()
        # 定义注意力机制的主要处理流程
        self.body = nn.Sequential(
            # 使用1x1卷积调整通道数
            nn.Conv2d(channels, f, 1),
            # 应用Soft Pooling来捕捉重要性信息
            SoftPooling2D(5, stride=2),
            # 通过3x3卷积进一步处理，步长为2
            nn.Conv2d(f, f, kernel_size=3, stride=2, padding=1),
            # 恢复到原始通道数
            nn.Conv2d(f, channels, 3, padding=1),
            # 使用Sigmoid激活函数生成权重
            nn.Sigmoid(),
        )

        """
            定义门控机制：
            为避免步长卷积和双线性插值带来的伪影，重新校准局部重要性，采用门控机制对局部重要性进行特征优化。
        """
        # self.gate = nn.Sequential(
        #     nn.Sigmoid(),  # 使用Sigmoid作为门控激活函数
        # )
        # 改进后的门控机制
        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),  # 使用1x1卷积学习门控权重
            nn.Sigmoid(),  # 使用Sigmoid作为门控激活函数
        )

    def forward(self, x):
        # 对输入的第一通道应用门控机制
        g = self.gate(x)

        # 将body处理的结果调整大小以匹配输入尺寸
        w = F.interpolate(self.body(x), (x.size(2), x.size(3)), mode='bilinear', align_corners=False)

        # 返回输入与生成的权重和门控值的乘积
        return x * w * g + x


class Global_Spatial_Attention(nn.Module):
    def __init__(self, in_dim):
        super(Global_Spatial_Attention, self).__init__()
        self.chanel_in = in_dim  # 输入通道数

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)  # 查询卷积
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)  # 键卷积
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)  # 值卷积

        self.gamma = nn.Parameter(torch.zeros(1))  # 可学习的缩放参数
        self.softmax = nn.Softmax(dim=-1)  # 对注意力权重进行归一化

    def forward(self, x):
        m_batchsize, C, height, width = x.size()  # 获取输入张量的批量大小、通道数、高度和宽度

        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # 计算查询向量并调整形状
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # 计算键向量并调整形状

        energy = torch.bmm(proj_query, proj_key)  # 计算查询和键的点积，得到注意力能量
        attention = self.softmax(energy)  # 对注意力能量进行归一化，得到注意力权重

        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # 计算值向量并调整形状
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # 根据注意力权重加权值向量
        out = out.view(m_batchsize, C, height, width)  # 恢复输出张量的形状

        out = self.gamma * out + x  # 将注意力输出与输入残差连接
        return out  # 返回最终的输出


class GALA(nn.Module):
    def __init__(self, in_dim):
        super(GALA, self).__init__()
        self.LA = LocalAttention(in_dim)
        self.GSA = Global_Spatial_Attention(in_dim)
        # self.gate = nn.Sequential(
        #     nn.Conv2d(in_dim * 2, in_dim, kernel_size=1),  # 使用1x1卷积
        #     nn.ReLU(),  # 添加ReLU激活函数
        #     nn.Conv2d(in_dim, 1, kernel_size=1),  # 再次使用1x1卷积
        #     nn.Sigmoid()  # Sigmoid激活函数
        # )
        self.gate = nn.Parameter(torch.tensor([0.1]))

    def forward(self, x):
        x1 = self.LA(x)
        x2 = self.GSA(x)
        # combined = torch.cat([x1, x2], dim=1)  # 拼接
        out = x1 + self.gate * x2
        return out + x