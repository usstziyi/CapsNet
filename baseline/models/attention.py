import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    """通道注意力 (2.4.3节)"""
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # 共享的两层感知机
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)


class SpatialAttention(nn.Module):
    """空间注意力 (2.4.4节)"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 沿通道维做平均和最大池化
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # 拼接
        x_cat = torch.cat([avg_out, max_out], dim=1)
        # 卷积生成权重图
        out = self.conv(x_cat)
        return x * self.sigmoid(out)


class FrequencyAttention(nn.Module):
    """频率注意力 (2.4.5节) - 基于FFT的频域加权"""
    def __init__(self, channels, reduction=16):
        super(FrequencyAttention, self).__init__()
        # 采用类似通道注意力的门控结构作用于频谱
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, H, W = x.size()
        # 1. 二维快速傅里叶变换
        fft_x = torch.fft.rfft2(x, norm='ortho')
        # 2. 取幅度谱
        amp = torch.abs(fft_x)
        # 3. 全局池化获取每个通道的频域统计量
        pooled_amp = torch.mean(amp, dim=(2, 3)) # (B, C)
        # 4. 生成频域权重
        weight = self.fc(pooled_amp).view(B, C, 1, 1)
        # 5. 权重作用于原图 (这里采用类似通道注意力的加权方式)
        return x * weight