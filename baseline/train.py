import torch
import torch.nn as nn
import torch.optim as optim
from models import CSFAPMCCNN

def train():
    # 1. 超参数设定
    batch_size = 8
    epochs = 10
    learning_rate = 1e-3
    
    # 2. 实例化模型、损失函数、优化器
    model = CSFAPMCCNN(in_channels=1, num_classes=2)
    
    # 注意：由于胶囊层输出的是模长，需要使用 Margin Loss (论文公式 2.24)
    # 这里为了演示跑通，暂时用简单的 MSE 配合 0/1 标签代替
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 3. 模拟数据加载 (B, C, H, W) -> (8, 1, 11, 4500)
    # 假设 H=11 (通道数), W=4500 (时间点)
    x_alpha = torch.randn(batch_size, 1, 11, 4500)
    x_beta = torch.randn(batch_size, 1, 11, 4500)
    labels = torch.randint(0, 2, (batch_size,)).float() # 标签 0 或 1
    
    # 4. 训练循环
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(x_alpha, x_beta) # 输出形状 (B, num_classes)
        
        # 这里简单取两个类别的概率差或直接one-hot计算MSE
        # 真实场景下请替换为 MarginLoss
        # 这里假设 outputs 是 [P(类0), P(类1)]
        # 构建目标概率分布
        target_probs = torch.zeros(batch_size, 2)
        target_probs[torch.arange(batch_size), labels.long()] = 1.0
        
        loss = criterion(outputs, target_probs)
        
        # 反向传播与优化
        loss.backward()
        optimizer.step()
        
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

if __name__ == '__main__':
    train()