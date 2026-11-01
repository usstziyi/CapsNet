import torch
import torch.nn as nn
import torch.nn.functional as F

# 如果没有 einops，请 pip install einops
try:
    from einops import rearrange
except ImportError:
    print("请安装 einops 库: pip install einops")

class PrimaryCaps(nn.Module):
    """初级胶囊层"""
    def __init__(self, in_channels, out_capsules, capsule_dim, kernel_size, stride):
        super(PrimaryCaps, self).__init__()
        self.capsule_dim = capsule_dim
        self.out_capsules = out_capsules
        # 用卷积生成胶囊特征
        self.conv = nn.Conv2d(in_channels, out_capsules * capsule_dim, 
                              kernel_size=kernel_size, stride=stride, bias=False)

    def forward(self, x):
        # x: (B, C, H, W)
        out = self.conv(x) # (B, out_capsules * capsule_dim, H', W')
        B, _, H, W = out.size()
        # 重塑为胶囊结构: (B, out_capsules, capsule_dim, H', W')
        out = out.view(B, self.out_capsules, self.capsule_dim, H, W)
        # 将空间维度展平，变成 (B, N, capsule_dim)，其中 N = out_capsules * H' * W'
        out = out.permute(0, 1, 3, 4, 2).contiguous()
        out = out.view(B, -1, self.capsule_dim)
        
        # Squash 非线性
        return self.squash(out)

    @staticmethod
    def squash(s, dim=-1):
        norm = torch.norm(s, dim=dim, keepdim=True)
        norm_squared = norm ** 2
        return (norm_squared / (1 + norm_squared)) * (s / (norm + 1e-8))


class DigitCaps(nn.Module):
    """分类胶囊层 (带动态路由)"""
    def __init__(self, in_caps, in_dim, num_caps, out_dim, num_routing=3):
        super(DigitCaps, self).__init__()
        self.in_caps = in_caps
        self.in_dim = in_dim
        self.num_caps = num_caps
        self.out_dim = out_dim
        self.num_routing = num_routing
        
        # 变换矩阵 W: (1, in_caps, num_caps, out_dim, in_dim)
        self.W = nn.Parameter(torch.randn(1, in_caps, num_caps, out_dim, in_dim) * 0.01)

    def forward(self, x):
        # x: (B, in_caps, in_dim)
        B = x.size(0)
        
        # 1. 计算预测向量 u_hat: (B, in_caps, num_caps, out_dim)
        # 对应论文公式 2.19
        x_expanded = x.unsqueeze(2).unsqueeze(4) # (B, in_caps, 1, in_dim, 1)
        W_expanded = self.W.expand(B, -1, -1, -1, -1) # (B, in_caps, num_caps, out_dim, in_dim)
        u_hat = torch.matmul(W_expanded, x_expanded).squeeze(4) # (B, in_caps, num_caps, out_dim)
        
        # 2. 动态路由
        b_ij = torch.zeros(B, self.in_caps, self.num_caps, 1).to(x.device)
        u_hat_detached = u_hat.detach()
        
        for i in range(self.num_routing):
            c_ij = F.softmax(b_ij, dim=2) # 耦合系数
            
            # 计算高层胶囊输入 s_j
            s_j = (c_ij * u_hat_detached).sum(dim=1, keepdim=True) # (B, 1, num_caps, out_dim)
            
            # Squash 得到 v_j
            v_j = self.squash(s_j) # (B, 1, num_caps, out_dim)
            
            if i < self.num_routing - 1:
                # 更新 b_ij (点积一致性)
                b_ij = b_ij + (u_hat_detached * v_j).sum(dim=-1, keepdim=True)
                
        return v_j.squeeze(1) # (B, num_caps, out_dim)

    @staticmethod
    def squash(s, dim=-1):
        norm = torch.norm(s, dim=dim, keepdim=True)
        norm_squared = norm ** 2
        return (norm_squared / (1 + norm_squared)) * (s / (norm + 1e-8))