# utils/loss.py
import torch
import ultralytics.utils.loss as loss_module
from ultralytics.utils.metrics import bbox_iou as original_bbox_iou

def custom_bbox_iou_with_nwd(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    """
    魔改版 bbox_iou：融合了原生 CIoU 和 NWD 损失。
    这是许多论文中常用的小目标涨分 Trick。
    """
    # 1. 调用官方原生函数计算标准的 IoU (内部已包含 CIoU 等惩罚项)
    iou = original_bbox_iou(box1, box2, xywh=xywh, GIoU=GIoU, DIoU=DIoU, CIoU=CIoU, eps=eps)
    
    # 2. 提取框的宽高和中心点用于计算 NWD
    if xywh:
        b1_cx, b1_cy, b1_w, b1_h = box1.chunk(4, -1)
        b2_cx, b2_cy, b2_w, b2_h = box2.chunk(4, -1)
    else:
        # 如果传入的是 xyxy 格式，先转换为中心点和宽高
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        b1_cx, b1_cy = (b1_x1 + b1_x2) / 2, (b1_y1 + b1_y2) / 2
        b1_w, b1_h = b1_x2 - b1_x1, b1_y2 - b1_y1
        
        b2_cx, b2_cy = (b2_x1 + b2_x2) / 2, (b2_y1 + b2_y2) / 2
        b2_w, b2_h = b2_x2 - b2_x1, b2_y2 - b2_y1

    # 3. 计算二维高斯分布的 Wasserstein 距离 (W^2)
    center_distance = (b1_cx - b2_cx)**2 + (b1_cy - b2_cy)**2
    wh_distance = ((b1_w - b2_w)**2 + (b1_h - b2_h)**2) / 4.0
    wasserstein_2 = center_distance + wh_distance

    # 4. 归一化，得到 NWD (数值范围 0~1，越接近 1 说明两个框越匹配)
    # 常数 C (constant) 在无人机数据集中通常取 12.8
    constant = 12.8
    nwd = torch.exp(-torch.sqrt(wasserstein_2 + eps) / constant)
    
    # 5. 联合输出：将 NWD 与原生 IoU 各占 50% 权重进行融合
    # 这样既保留了 CIoU 对框形状的敏感度，又加入了 NWD 对小目标的宽容度！
    return (iou + nwd) / 2.0

def inject_nwd_loss():
    """魔法操作：将官方的 bbox_iou 替换为我们融合了 NWD 的版本"""
    loss_module.bbox_iou = custom_bbox_iou_with_nwd
    print("[INFO] 成功挂载科研魔改模块: NWD Loss (小目标 Wasserstein 损失)")