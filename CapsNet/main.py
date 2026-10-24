import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ==========================================
# 1. 核心函数：Squash (压缩函数)
# ==========================================
def squash(x, dim=-1):
    """
    将向量的模长压缩到 0 到 1 之间，同时保持方向不变。
    x: 输入向量
    """
    squared_norm = (x ** 2).sum(dim=dim, keepdim=True)
    scale = squared_norm / (1 + squared_norm)
    return scale * x / torch.sqrt(squared_norm + 1e-8)

# ==========================================
# 2. 核心层：PrimaryCapsule (初级胶囊层)
# ==========================================
class PrimaryCapsule(nn.Module):
    def __init__(self, in_channels=256, out_capsules=32, out_dim=8):
        super(PrimaryCapsule, self).__init__()
        # 使用普通卷积提取特征，将其重组为胶囊形式
        self.conv = nn.Conv2d(in_channels, out_capsules * out_dim, kernel_size=9, stride=2, padding=0)
        self.out_capsules = out_capsules # 胶囊数量
        self.out_dim = out_dim # 胶囊维度

    def forward(self, x):
        # x shape: [batch, 256, 20, 20]
        out = self.conv(x) # shape: [batch, 32*8, 6, 6]
        # 将通道维度拆分为 [胶囊数量, 胶囊维度]
        out = out.view(out.size(0), self.out_capsules, self.out_dim, -1)
        # 展平空间维度 [batch, 32, 8, 36] -> [batch, 32*36, 8]
        out = out.permute(0, 1, 3, 2).contiguous()
        out = out.view(out.size(0), -1, self.out_dim)
        # 应用 squash 激活
        # squash 是 跨通道 的非线性：它把同一胶囊的 8 个通道当成一条向量，一起算模长再整体缩放。
        # 这一步才让“8 个标量”变成一个“有方向、有长度”的实体
        return squash(out)

# ==========================================
# 3. 核心层：DigitCapsule (数字胶囊层 + 动态路由)
# ==========================================
class DigitCapsule(nn.Module):
    def __init__(self, in_capsules=32*36, in_dim=8, out_capsules=10, out_dim=16, num_routing=3):
        super(DigitCapsule, self).__init__()
        self.in_capsules = in_capsules
        self.in_dim = in_dim
        self.out_capsules = out_capsules
        self.out_dim = out_dim
        self.num_routing = num_routing
        
        # 权重矩阵 W: [1, 输入胶囊数, 输出胶囊数, 输出维度, 输入维度]
        self.W = nn.Parameter(torch.randn(1, in_capsules, out_capsules, out_dim, in_dim))

    def forward(self, x):
        batch_size = x.size(0)
        # 1. 扩展维度以便进行矩阵乘法
        # x: [batch, in_capsules, in_dim] -> [batch, in_capsules, 1, in_dim, 1]
        x = x.unsqueeze(2).unsqueeze(4)
        
        # 2. 预测向量 (Prediction Vectors)
        # u_hat: [batch, in_capsules, out_capsules, out_dim, 1]
        u_hat = torch.matmul(self.W, x).squeeze(-1)

        # 3. 动态路由 (Dynamic Routing)
        # b_ij 是耦合系数的初始对数（全为0）
        b_ij = torch.zeros(batch_size, self.in_capsules, self.out_capsules, 1).to(x.device)

        for i in range(self.num_routing):
            # 计算耦合系数 c_ij (Softmax)
            c_ij = F.softmax(b_ij, dim=2)
            
            # 计算加权和 s_j
            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)
            
            # 应用 squash 得到输出 v_j
            v_j = squash(s_j, dim=-1)
            
            # 如果不是最后一次迭代，更新 b_ij
            if i < self.num_routing - 1:
                # 路由协议：b_ij = b_ij + u_hat * v_j
                b_ij = b_ij + (u_hat * v_j).sum(dim=-1, keepdim=True)

        # 返回最终的输出 [batch, 1, out_capsules, out_dim]
        return v_j.squeeze(1)

# ==========================================
# 4. 组装完整模型：CapsNet
# ==========================================
class CapsNet(nn.Module):
    def __init__(self):
        super(CapsNet, self).__init__()
        # 普通卷积层（特征提取）
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=9, stride=1)
        # 初级胶囊层
        self.primary_caps = PrimaryCapsule(in_channels=256, out_capsules=32, out_dim=8)
        # 数字胶囊层
        self.digit_caps = DigitCapsule(in_capsules=32*36, in_dim=8, out_capsules=10, out_dim=16)

    def forward(self, x):
        # 1. 卷积层 + ReLU
        x = F.relu(self.conv1(x)) # [batch, 256, 20, 20]
        # 2. 初级胶囊
        x = self.primary_caps(x)  # [batch, 32*36, 8]
        # 3. 数字胶囊
        x = self.digit_caps(x)    # [batch, 10, 16]
        
        # 4. 计算分类概率（向量的模长）
        # 每个数字胶囊向量的长度就是该类别的概率
        probs = torch.sqrt((x ** 2).sum(dim=-1)) 
        return probs

# ==========================================
# 5. 训练与测试流程
# ==========================================
def train_and_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 数据加载
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 初始化模型、优化器和损失函数
    model = CapsNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # Margin Loss 是胶囊网络常用的损失函数，这里为了简化 Demo 直接用交叉熵
    criterion = nn.CrossEntropyLoss() 

    # 训练一个 Epoch（演示用，实际需要更多 epoch）
    epochs = 2
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            probs = model(data) # 输出是每个类别的概率（模长）
            loss = criterion(probs, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        # 测试
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                probs = model(data)
                pred = probs.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        acc = 100. * correct / total
        print(f"====> Epoch {epoch+1} Test Accuracy: {acc:.2f}%")

if __name__ == '__main__':
    train_and_test()