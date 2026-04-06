# models/custom_blocks.py
import torch
import torch.nn as nn

class GlobalContextAttention(nn.Module):
    """
    自适应全局上下文注意力机制 (Adaptive GCA)
    我们改进了设计：不再需要在 __init__ 中硬编码输入通道数，
    而是在第一次前向传播 (forward) 时，自动根据输入张量推断通道数并初始化网络！
    这完美绕过了 YOLO parse_model 的死板限制。
    """
    def __init__(self, reduction_ratio=16):
        super(GlobalContextAttention, self).__init__()
        self.reduction_ratio = reduction_ratio
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
        
        # 将这里的层设为 None，等待第一次数据流入时再构建
        self.fc1 = None
        self.fc2 = None
        self.is_initialized = False

    def forward(self, x):
        b, c, _, _ = x.size()
        
        # 【核心魔法】：延迟初始化 (Lazy Initialization)
        # 如果是第一次运行（或者模型在探测结构时），动态构建这两层全连接层
        if not self.is_initialized:
            # YOLO 模型导出到 ONNX 时不支持动态结构，所以要将其固定为 nn.Module
            reduced_channels = max(1, c // self.reduction_ratio)
            self.fc1 = nn.Conv2d(c, reduced_channels, kernel_size=1, bias=False).to(x.device)
            self.fc2 = nn.Conv2d(reduced_channels, c, kernel_size=1, bias=False).to(x.device)
            self.is_initialized = True

        # 正常的前向传播逻辑
        y = self.global_pool(x)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        weights = self.sigmoid(y)
        
        return x * weights.expand_as(x)