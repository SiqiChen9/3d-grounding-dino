"""
MONAI 3D RetinaNet 模型 - 用于与 3D Grounding DINO 对比的基线模型
单阶段 3D 检测器，使用焦点损失处理样本不平衡
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import ResNet
from monai.networks.layers import Norm, Conv
from typing import List, Tuple, Optional, Dict
import numpy as np
import yaml
import os
from pathlib import Path
from datetime import datetime
import json
from typing import Tuple


def load_config(config_path: str) -> Dict:
    """加载 YAML 配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict, save_path: str):
    """保存配置到 YAML 文件"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


class FocalLoss(nn.Module):
    """焦点损失 - 处理样本不平衡"""
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: 预测的类别概率 (N, num_classes, ...)
            targets: 真实标签 (N, ...)
        """
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        p_t = p * targets + (1 - p) * (1 - targets)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        return focal_loss.mean()


def compute_iou_3d(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Compute 3D IoU between two boxes (cx, cy, cz, w, h, d) in voxel coords.
    """
    def get_box_3d_coords(box):
        cx, cy, cz, w, h, d = box
        x_min = cx - w / 2
        x_max = cx + w / 2
        y_min = cy - h / 2
        y_max = cy + h / 2
        z_min = cz - d / 2
        z_max = cz + d / 2
        return x_min, x_max, y_min, y_max, z_min, z_max
    
    # Get coordinates
    x1_min, x1_max, y1_min, y1_max, z1_min, z1_max = get_box_3d_coords(box1)
    x2_min, x2_max, y2_min, y2_max, z2_min, z2_max = get_box_3d_coords(box2)
    
    # Intersection
    x_inter = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
    y_inter = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
    z_inter = max(0, min(z1_max, z2_max) - max(z1_min, z2_min))
    inter_volume = x_inter * y_inter * z_inter
    
    # Union
    vol1 = (x1_max - x1_min) * (y1_max - y1_min) * (z1_max - z1_min)
    vol2 = (x2_max - x2_min) * (y2_max - y2_min) * (z2_max - z2_min)
    union_volume = vol1 + vol2 - inter_volume
    
    if union_volume < 1e-6:
        return 0.0
    
    return inter_volume / union_volume


class Anchor3DGenerator:
    """3D 锚点生成器"""
    def __init__(
        self,
        image_size: Tuple[int, int, int] = (64, 64, 64),
        feature_stride: int = 8,
        scales: List[float] = None,
        ratios: List[float] = None
    ):
        self.image_size = image_size
        self.feature_stride = feature_stride
        self.scales = scales or [1.0, 1.25, 1.5]
        self.ratios = ratios or [1.0]
        
    def generate(
        self,
        device: torch.device,
        image_size: Optional[Tuple[int, int, int]] = None,
        feature_stride: Optional[int] = None,
        feature_map_size: Optional[Tuple[int, int, int]] = None,
        scales: Optional[List[float]] = None,
        ratios: Optional[List[float]] = None
    ) -> torch.Tensor:
        """
        生成锚点 (D, H, W, num_anchors, 6)
        统一格式: (cx, cy, cz, w, h, d)
        对于 scale s 和 ratio r，生成尺寸为:
        - base_size * s (立方体基准)
        - base_size * s * sqrt(r) 和 base_size * s / sqrt(r) (用于高度和宽度) 
        这样可以创建不同纵横比的 anchors
        """
        img_size = image_size or self.image_size
        stride = feature_stride or self.feature_stride
        use_scales = scales or self.scales
        use_ratios = ratios or self.ratios

        d, h, w = img_size  # (D, H, W) for tensor layout
        if feature_map_size is not None:
            fd, fh, fw = feature_map_size
        else:
            if isinstance(stride, (tuple, list)):
                fd = int(round(d / stride[0]))
                fh = int(round(h / stride[1]))
                fw = int(round(w / stride[2]))
            else:
                fd = d // stride
                fh = h // stride
                fw = w // stride
        
        # 在特征图上生成网格 (z, y, x)
        if isinstance(stride, (tuple, list)):
            sd, sh, sw = float(stride[0]), float(stride[1]), float(stride[2])
        else:
            sd = sh = sw = float(stride)

        dg = torch.arange(fd, dtype=torch.float32, device=device) * sd
        hg = torch.arange(fh, dtype=torch.float32, device=device) * sh
        wg = torch.arange(fw, dtype=torch.float32, device=device) * sw
        
        dd, hh, ww = torch.meshgrid(dg, hg, wg, indexing='ij')
        
        # 创建锚点
        anchors = []
        base_d, base_h, base_w = sd, sh, sw
        
        for scale in use_scales:
            for ratio in use_ratios:
                # 为不同维度应用不同的比例
                # base: 立方体尺寸
                # H 维度扩大 sqrt(ratio) 倍，W 维度缩小 sqrt(ratio) 倍
                size_d = base_d * scale
                size_h = base_h * scale * (ratio ** (1/2))
                size_w = base_w * scale / (ratio ** (1/2))

                # 统一格式: (cx, cy, cz, w, h, d) == (x, y, z, w, h, d)
                anchor = torch.stack([
                    ww, hh, dd,
                    torch.full_like(ww, size_w),
                    torch.full_like(hh, size_h),
                    torch.full_like(dd, size_d)
                ], dim=-1)
                anchors.append(anchor)
        
        return torch.stack(anchors, dim=3)  # (FD, FH, FW, num_scales*num_ratios, 6)


class AnchorMatcher3D:
    """为 RetinaNet 匹配 GT boxes 到 anchors"""
    def __init__(
        self,
        fg_iou_threshold: float = 0.5,
        bg_iou_threshold: float = 0.4,
        allow_low_quality_matches: bool = True
    ):
        self.fg_iou_threshold = fg_iou_threshold
        self.bg_iou_threshold = bg_iou_threshold
        self.allow_low_quality_matches = allow_low_quality_matches
    
    def match(
        self,
        gt_boxes: np.ndarray,  # (N_gt, 6) voxel coords (cx, cy, cz, w, h, d)
        gt_labels: np.ndarray,  # (N_gt,) 类别标签
        anchors: torch.Tensor,  # (FD, FH, FW, num_anchors, 6)
        feature_stride: int = 8,
        volume_shape: Tuple[int, int, int] = (64, 64, 64)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回:
            cls_targets: (1, num_anchors, FD, FH, FW) - 类别标签，-1 未分配
            bbox_targets: (1, num_anchors*6, FD, FH, FW) - 边界框回归目标
        """
        FD, FH, FW, num_anchors_per_loc, _ = anchors.shape
        device = anchors.device
        
        # 初始化目标 (-1 表示忽略)
        cls_targets = torch.full(
            (1, num_anchors_per_loc, FD, FH, FW),
            -1,
            dtype=torch.long,
            device=device
        )
        bbox_targets = torch.zeros(
            (1, num_anchors_per_loc * 6, FD, FH, FW),
            dtype=torch.float32,
            device=device
        )
        
        if len(gt_boxes) == 0:
            # 没有 GT，所有 anchor 都是背景
            cls_targets[cls_targets == -1] = 0
            return cls_targets, bbox_targets
        
        # 展平 anchors 为 (N_anchors, 6)
        anchors_flat = anchors.reshape(-1, 6)
        anchors_np = anchors_flat.cpu().numpy()
        
        # 逐个 GT box 分配到最佳 anchor
        for gt_idx in range(len(gt_boxes)):
            gt_box = gt_boxes[gt_idx]
            gt_label = gt_labels[gt_idx]
            
            # 计算所有 anchor 与该 GT box 的 IoU
            ious = np.array([compute_iou_3d(anchor, gt_box) for anchor in anchors_np])
            
            # 找最好匹配
            best_anchor_idx = np.argmax(ious)
            best_iou = ious[best_anchor_idx]
            
            # 如果 IoU 足够高，分配这个 anchor 到 GT
            if best_iou > self.bg_iou_threshold:
                # 转换平坦索引到 (d, h, w, a) where d=z, h=y, w=x in tensor layout
                a = best_anchor_idx % num_anchors_per_loc
                w = (best_anchor_idx // num_anchors_per_loc) % FW
                h = (best_anchor_idx // (num_anchors_per_loc * FW)) % FH
                d = best_anchor_idx // (num_anchors_per_loc * FW * FH)
                
                # 只在未分配时才设置
                if cls_targets[0, a, d, h, w] == -1:
                    # 设置分类目标
                    cls_targets[0, a, d, h, w] = int(gt_label)
                    
                    # 计算回归目标 (delta)
                    anchor = anchors[d, h, w, a].cpu().numpy()
                    dx = (gt_box[0] - anchor[0]) / (anchor[3] + 1e-8)
                    dy = (gt_box[1] - anchor[1]) / (anchor[4] + 1e-8)
                    dz = (gt_box[2] - anchor[2]) / (anchor[5] + 1e-8)
                    dw = np.log((gt_box[3] + 1e-8) / (anchor[3] + 1e-8))
                    dh = np.log((gt_box[4] + 1e-8) / (anchor[4] + 1e-8))
                    dd = np.log((gt_box[5] + 1e-8) / (anchor[5] + 1e-8))
                    
                    bbox_targets[0, a*6:(a+1)*6, d, h, w] = torch.tensor(
                        [dx, dy, dz, dw, dh, dd], dtype=torch.float32, device=device
                    )
        
        # 设置未分配的 anchor 为背景（标签 0）
        cls_targets[cls_targets == -1] = 0
        
        return cls_targets, bbox_targets


class FPN3D(nn.Module):
    """Simple 3D FPN with three levels (P3, P4, P5)."""
    def __init__(self, in_channels_list: List[int], out_channels: int = 256):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv3d(in_ch, out_channels, kernel_size=1)
            for in_ch in in_channels_list
        ])
        self.output_convs = nn.ModuleList([
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
            for _ in in_channels_list
        ])

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        # features expected as [C3, C4, C5] low->high level
        lateral = [conv(f) for conv, f in zip(self.lateral_convs, features)]

        # top-down pathway
        p5 = lateral[2]
        p4 = lateral[1] + F.interpolate(p5, size=lateral[1].shape[2:], mode='trilinear', align_corners=False)
        p3 = lateral[0] + F.interpolate(p4, size=lateral[0].shape[2:], mode='trilinear', align_corners=False)

        p5 = self.output_convs[2](p5)
        p4 = self.output_convs[1](p4)
        p3 = self.output_convs[0](p3)

        return [p3, p4, p5]


class DetectionHead3D(nn.Module):
    """3D 检测头 - 分类和回归"""
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        num_anchors: int = 9
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        # 分类分支
        self.cls_conv1 = nn.Conv3d(in_channels, 256, kernel_size=3, padding=1)
        self.cls_conv2 = nn.Conv3d(256, 256, kernel_size=3, padding=1)
        self.cls_conv3 = nn.Conv3d(256, 256, kernel_size=3, padding=1)
        self.cls_conv4 = nn.Conv3d(256, 256, kernel_size=3, padding=1)
        self.cls_out = nn.Conv3d(256, num_anchors * num_classes, kernel_size=3, padding=1)
        
        # 回归分支
        self.reg_conv1 = nn.Conv3d(in_channels, 256, kernel_size=3, padding=1)
        self.reg_conv2 = nn.Conv3d(256, 256, kernel_size=3, padding=1)
        self.reg_conv3 = nn.Conv3d(256, 256, kernel_size=3, padding=1)
        self.reg_conv4 = nn.Conv3d(256, 256, kernel_size=3, padding=1)
        self.reg_out = nn.Conv3d(256, num_anchors * 6, kernel_size=3, padding=1)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: 特征图 (B, C, D, H, W)
        Returns:
            cls_score: 分类分数 (B, num_anchors * num_classes, D, H, W)
            bbox_pred: 边界框预测 (B, num_anchors * 6, D, H, W)
        """
        # 分类
        cls_feat = F.relu(self.cls_conv1(x))
        cls_feat = F.relu(self.cls_conv2(cls_feat))
        cls_feat = F.relu(self.cls_conv3(cls_feat))
        cls_feat = F.relu(self.cls_conv4(cls_feat))
        cls_score = self.cls_out(cls_feat)
        
        # 回归
        reg_feat = F.relu(self.reg_conv1(x))
        reg_feat = F.relu(self.reg_conv2(reg_feat))
        reg_feat = F.relu(self.reg_conv3(reg_feat))
        reg_feat = F.relu(self.reg_conv4(reg_feat))
        bbox_pred = self.reg_out(reg_feat)
        
        return cls_score, bbox_pred


class RetinaNet3D(nn.Module):
    """MONAI 风格的 3D RetinaNet 检测器"""
    def __init__(
        self,
        image_size: Tuple[int, int, int] = (64, 64, 64),
        num_classes: int = 1,
        in_channels: int = 1,
        backbone_depth: int = 18,
        num_anchors: int = 9
    ):
        super().__init__()
        self.image_size = image_size
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.in_channels = in_channels
        
        # ResNet 骨干网络 - 直接使用简化骨干网络避免 MONAI 兼容性问题
        # (MONAI ResNet 在不同版本间参数差异大)
        self.backbone = self._create_simple_backbone()

        # FPN (P3, P4, P5) 统一输出通道
        self.fpn_out_channels = 256
        self.fpn = FPN3D(in_channels_list=[128, 256, 512], out_channels=self.fpn_out_channels)

        # 检测头 (多尺度共享权重)
        self.detection_head = DetectionHead3D(
            in_channels=self.fpn_out_channels,
            num_classes=num_classes,
            num_anchors=num_anchors
        )
        
        # 锚点生成器 (生成 3 scales × 3 ratios = 9 anchors)
        self.anchor_generator = Anchor3DGenerator(
            image_size=image_size,
            feature_stride=8,
            scales=[1.0, 1.26, 1.59],    # 3 scales: [1, 2^(1/3), 2^(2/3)]
            ratios=[1.0, 2.0, 0.5]       # 3 ratios for 3D aspect variations
        )
        
        # 焦点损失
        self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
        
        # L1 损失用于回归
        self.l1_loss = nn.SmoothL1Loss(reduction='mean')
    
    def _create_simple_backbone(self):
        """创建简化的骨干网络（当 MONAI ResNet 失败时使用）"""
        class SimpleBackbone(nn.Module):
            def __init__(self, in_channels=1):
                super().__init__()
                # Layer 1: in_channels -> 64
                self.layer1 = nn.Sequential(
                    nn.Conv3d(in_channels, 64, kernel_size=3, padding=1),
                    nn.BatchNorm3d(64),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(64, 64, kernel_size=3, padding=1),
                    nn.BatchNorm3d(64),
                    nn.ReLU(inplace=True),
                    nn.MaxPool3d(kernel_size=2, stride=2)
                )
                # Layer 2: 64 -> 128
                self.layer2 = nn.Sequential(
                    nn.Conv3d(64, 128, kernel_size=3, padding=1),
                    nn.BatchNorm3d(128),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(128, 128, kernel_size=3, padding=1),
                    nn.BatchNorm3d(128),
                    nn.ReLU(inplace=True),
                    nn.MaxPool3d(kernel_size=2, stride=2)
                )
                # Layer 3: 128 -> 256
                self.layer3 = nn.Sequential(
                    nn.Conv3d(128, 256, kernel_size=3, padding=1),
                    nn.BatchNorm3d(256),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(256, 256, kernel_size=3, padding=1),
                    nn.BatchNorm3d(256),
                    nn.ReLU(inplace=True),
                    nn.MaxPool3d(kernel_size=2, stride=2)
                )
                # Layer 4: 256 -> 512
                self.layer4 = nn.Sequential(
                    nn.Conv3d(256, 512, kernel_size=3, padding=1),
                    nn.BatchNorm3d(512),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(512, 512, kernel_size=3, padding=1),
                    nn.BatchNorm3d(512),
                    nn.ReLU(inplace=True),
                    nn.MaxPool3d(kernel_size=2, stride=2)
                )
            
            def forward(self, x):
                """
                输入: (B, C, D, H, W) - 任意大小
                输出: 多尺度特征 [C3, C4, C5]
                """
                x = self.layer1(x)   # (B, 64, D/2, H/2, W/2)
                c3 = self.layer2(x)  # (B, 128, D/4, H/4, W/4)
                c4 = self.layer3(c3) # (B, 256, D/8, H/8, W/8)
                c5 = self.layer4(c4) # (B, 512, D/16, H/16, W/16)
                return [c3, c4, c5]
        
        return SimpleBackbone(in_channels=self.in_channels)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: 输入体积 (B, C, D, H, W)
        Returns:
            cls_score: 分类分数
            bbox_pred: 边界框预测
        """
        # 确保输入是 float32 (避免数据类型不匹配)
        x = x.float()
        
        # 骨干网络前向传播 (多尺度)
        c_features = self.backbone(x)

        # 调试: 检查特征形状
        if any(f.shape[1] == 0 for f in c_features):
            raise RuntimeError("骨干网络输出错误: 特征通道为 0。"
                             "这通常意味着骨干网络初始化失败。")

        # FPN
        p_features = self.fpn(c_features)  # [P3, P4, P5]

        # 检测头 (多尺度)
        cls_scores = []
        bbox_preds = []
        for feat in p_features:
            cls_score, bbox_pred = self.detection_head(feat)
            cls_scores.append(cls_score)
            bbox_preds.append(bbox_pred)

        return {
            'cls_scores': cls_scores,
            'bbox_preds': bbox_preds,
            'features': p_features
        }
    
    def compute_loss(
        self,
        cls_score: torch.Tensor,
        bbox_pred: torch.Tensor,
        cls_targets: torch.Tensor,
        bbox_targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        计算损失函数
        
        Args:
            cls_score: 分类分数 (B, num_anchors * num_classes, D, H, W)
            bbox_pred: 边界框预测 (B, num_anchors * 6, D, H, W)
            cls_targets: 分类目标 (B, num_anchors, D_target, H_target, W_target)
            bbox_targets: 边界框目标 (B, num_anchors * 6, D_target, H_target, W_target)
        """
        # 获取实际的特征图大小
        _, _, d_feat, h_feat, w_feat = cls_score.shape
        
        # 如果目标大小与特征图大小不匹配，调整目标大小
        if cls_targets.shape[2:] != (d_feat, h_feat, w_feat):
            # 使用双线性插值调整目标大小
            # cls_targets: (B, num_anchors, D_target, H_target, W_target)
            # 需要转换为 (B*num_anchors, 1, D_target, H_target, W_target) 用 interpolate
            b, na, _, _, _ = cls_targets.shape
            cls_targets_resized = torch.nn.functional.interpolate(
                cls_targets.view(b * na, 1, -1, cls_targets.shape[3], cls_targets.shape[4]),
                size=(d_feat, h_feat, w_feat),
                mode='nearest'
            ).view(b, na, d_feat, h_feat, w_feat)
            cls_targets = cls_targets_resized
            
            bbox_targets_resized = torch.nn.functional.interpolate(
                bbox_targets.view(b * self.num_anchors * 6, 1, -1, bbox_targets.shape[3], bbox_targets.shape[4]),
                size=(d_feat, h_feat, w_feat),
                mode='nearest'
            ).view(b, self.num_anchors * 6, d_feat, h_feat, w_feat)
            bbox_targets = bbox_targets_resized
        
        # 分类损失
        cls_loss = self.focal_loss(cls_score, cls_targets)

        # 回归损失 (只对正样本计算)
        # 正样本: cls_targets > 0
        pos_mask = (cls_targets > 0).unsqueeze(2)  # (B, A, 1, D, H, W)
        bbox_pred_reshaped = bbox_pred.view(
            bbox_pred.shape[0], self.num_anchors, 6, *bbox_pred.shape[2:]
        )
        bbox_targets_reshaped = bbox_targets.view(
            bbox_targets.shape[0], self.num_anchors, 6, *bbox_targets.shape[2:]
        )
        bbox_loss_raw = F.smooth_l1_loss(
            bbox_pred_reshaped, bbox_targets_reshaped, reduction='none'
        )
        bbox_loss_masked = bbox_loss_raw * pos_mask.float()
        denom = pos_mask.sum() * 6.0
        bbox_loss = bbox_loss_masked.sum() / (denom + 1e-8)
        
        # 总损失
        total_loss = cls_loss + bbox_loss
        
        return {
            'cls_loss': cls_loss,
            'bbox_loss': bbox_loss,
            'total_loss': total_loss
        }

    @staticmethod
    def anchors_grid(image_size: Tuple[int, int, int], feat_size: Tuple[int, int, int],
                     strides: Tuple[int, int, int], anchors_per_loc: int = 9) -> torch.Tensor:
        """
        Generate a simple anchors grid (center coords) for a feature map size.
        Returns anchors centers in voxel coordinates as (cx, cy, cz).
        """
        D, H, W = image_size
        fd, fh, fw = feat_size
        sd, sh, sw = strides
        # centers at stride/2 + i*stride
        zs = (torch.arange(fd).float() * sd) + sd / 2.0
        ys = (torch.arange(fh).float() * sh) + sh / 2.0
        xs = (torch.arange(fw).float() * sw) + sw / 2.0
        zz, yy, xx = torch.meshgrid(zs, ys, xs, indexing='ij')
        centers = torch.stack([xx, yy, zz], dim=-1).view(-1, 3)
        # repeat for anchors_per_loc
        anchors = centers.unsqueeze(1).repeat(1, anchors_per_loc, 1).view(-1, 3)
        return anchors

    @staticmethod
    def iou_3d(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """
        Compute IoU between two sets of axis-aligned 3D boxes.
        Boxes format: (N, 6) as (cx, cy, cz, w, h, d)
        Returns IoU matrix (N, M)
        """
        if boxes1.numel() == 0 or boxes2.numel() == 0:
            return torch.zeros((boxes1.shape[0], boxes2.shape[0]), device=boxes1.device)

        # convert to min/max corners
        def corners(boxes):
            c = boxes[:, :3]
            s = boxes[:, 3:]
            mins = c - s / 2
            maxs = c + s / 2
            return mins, maxs

        mins1, maxs1 = corners(boxes1)
        mins2, maxs2 = corners(boxes2)

        N = boxes1.shape[0]
        M = boxes2.shape[0]

        mins1 = mins1.unsqueeze(1).expand(N, M, 3)
        maxs1 = maxs1.unsqueeze(1).expand(N, M, 3)
        mins2 = mins2.unsqueeze(0).expand(N, M, 3)
        maxs2 = maxs2.unsqueeze(0).expand(N, M, 3)

        inter_mins = torch.max(mins1, mins2)
        inter_maxs = torch.min(maxs1, maxs2)
        inter_sizes = (inter_maxs - inter_mins).clamp(min=0)
        inter_vol = inter_sizes[:, :, 0] * inter_sizes[:, :, 1] * inter_sizes[:, :, 2]

        vol1 = ((maxs1 - mins1)[:, :, 0] * (maxs1 - mins1)[:, :, 1] * (maxs1 - mins1)[:, :, 2])
        vol2 = ((maxs2 - mins2)[:, :, 0] * (maxs2 - mins2)[:, :, 1] * (maxs2 - mins2)[:, :, 2])

        union = vol1 + vol2 - inter_vol
        iou = inter_vol / (union + 1e-8)
        return iou

    @staticmethod
    def decode_boxes(anchors: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
        """Decode RetinaNet-style deltas to boxes (cx, cy, cz, w, h, d)."""
        ax, ay, az, aw, ah, ad = anchors.unbind(dim=1)
        dx, dy, dz, dw, dh, dd = deltas.unbind(dim=1)

        cx = dx * aw + ax
        cy = dy * ah + ay
        cz = dz * ad + az
        w = torch.exp(dw) * aw
        h = torch.exp(dh) * ah
        d = torch.exp(dd) * ad

        return torch.stack([cx, cy, cz, w, h, d], dim=1)

    @staticmethod
    def nms_3d(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
        """Simple 3D NMS on axis-aligned boxes. Returns indices to keep."""
        if boxes.numel() == 0:
            return torch.empty((0,), dtype=torch.long, device=boxes.device)

        # Convert to corners
        c = boxes[:, :3]
        s = boxes[:, 3:]
        mins = c - s / 2
        maxs = c + s / 2

        order = scores.argsort(descending=True)
        keep = []

        while order.numel() > 0:
            i = order[0]
            keep.append(i)
            if order.numel() == 1:
                break

            rest = order[1:]
            inter_mins = torch.max(mins[i], mins[rest])
            inter_maxs = torch.min(maxs[i], maxs[rest])
            inter_sizes = (inter_maxs - inter_mins).clamp(min=0)
            inter_vol = inter_sizes[:, 0] * inter_sizes[:, 1] * inter_sizes[:, 2]

            vol_i = (maxs[i] - mins[i]).prod()
            vol_rest = (maxs[rest] - mins[rest]).prod(dim=1)
            union = vol_i + vol_rest - inter_vol
            iou = inter_vol / (union + 1e-8)

            order = rest[iou <= iou_threshold]

        return torch.stack(keep)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='3D RetinaNet Detector - 与 3D Grounding DINO 对比模型')
    parser.add_argument('--config', type=str, default='./Retinanet/retinanet_config.yaml',
                        help='配置文件路径')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='数据集路径（如提供则覆盖配置文件）')
    parser.add_argument('--demo', action='store_true', help='仅运行演示（随机数据）')
    parser.add_argument('--device', type=str, default=None,
                        help='设备（默认自动选择 cuda/cpu）')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='恢复检查点的路径')
    
    args = parser.parse_args()
    
    # 加载配置文件
    print("=" * 70)
    print("3D RetinaNet 检测器")
    print("=" * 70)
    
    if os.path.exists(args.config):
        print(f"\n[✓] 加载配置文件: {args.config}")
        config = load_config(args.config)
    else:
        print(f"\n[✗] 配置文件不存在: {args.config}")
        print("    使用默认配置")
        config = {
            'data': {'dataset_path': '/home/woody/iwi5/iwi5378h/rsna2023', 'batch_size': 2},
            'model': {'num_classes': 1, 'num_anchors': 9, 'image_size': [64, 64, 64]},
            'training': {'epochs': 100, 'lr': 1e-4},
            'loss': {'focal_alpha': 0.25, 'focal_gamma': 2.0},
            'checkpoint': {'save_dir': './checkpoints/retinanet'},
            'logging': {'log_dir': './logs/retinanet'}
        }
    
    # 命令行参数覆盖配置文件
    if args.data_dir:
        config['data']['dataset_path'] = args.data_dir
    if args.device:
        device = args.device
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"[✓] 设备: {device}")
    print(f"[✓] 数据集路径: {config['data']['dataset_path']}")
    print(f"[✓] 批次大小: {config['data']['batch_size']}")
    print(f"[✓] 训练轮数: {config['training']['epochs']}")
    print(f"[✓] 学习率: {config['training']['lr']}")
    
    if args.demo:
        print("\n" + "=" * 70)
        print("演示模式 - 使用随机数据")
        print("=" * 70)
        
        # 创建模型
        model_cfg = config['model']
        model = RetinaNet3D(
            image_size=tuple(model_cfg.get('image_size', [64, 64, 64])),
            num_classes=model_cfg.get('num_classes', 1),
            in_channels=model_cfg.get('in_channels', 1),
            num_anchors=model_cfg.get('num_anchors', 9)
        ).to(device)
        
        print(f"\n[✓] 模型已创建")
        print(f"    - 输入大小: {tuple(model_cfg.get('image_size', [64, 64, 64]))}")
        print(f"    - 类别数: {model_cfg.get('num_classes', 1)}")
        print(f"    - 锚点数: {model_cfg.get('num_anchors', 9)}")
        
        # 输入数据
        x = torch.randn(1, 1, 64, 64, 64).to(device)
        
        # 前向传播
        print(f"\n正在执行前向传播...")
        outputs = model(x)
        
        print(f"\n[✓] 输出形状:")
        for i, (cls_score, bbox_pred) in enumerate(zip(outputs['cls_scores'], outputs['bbox_preds'])):
            print(f"    - Level P{i+3} 分类分数: {cls_score.shape}")
            print(f"    - Level P{i+3} 边界框预测: {bbox_pred.shape}")
        
        # 假设有目标数据
        cls_targets = torch.randint(0, 2, (1, 9, 8, 8, 8)).float().to(device)
        bbox_targets = torch.randn(1, 54, 8, 8, 8).to(device)
        
        # 计算损失
        print(f"\n正在计算损失...")
        losses = model.compute_loss(
            outputs['cls_scores'][0],
            outputs['bbox_preds'][0],
            cls_targets,
            bbox_targets
        )
        
        print(f"\n[✓] 损失:")
        for key, value in losses.items():
            print(f"    - {key}: {value.item():.6f}")
        
        print("\n演示完成！")
    
    else:
        print("\n" + "=" * 70)
        print("模式: RSNA 数据集训练")
        print("=" * 70)
        
        try:
            # 返回到项目根目录以导入 datasets
            import sys
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            sys.path.insert(0, project_root)
            
            from datasets import RSNAVolumeDataset, collate_fn
            from torch.utils.data import DataLoader, Subset
            from utils.metrics import compute_map
            
            # 加载数据集
            print(f"\n[*] 加载数据集...")
            data_cfg = config['data']
            dataset = RSNAVolumeDataset(
                data_dir=data_cfg['dataset_path'],
                target_width=data_cfg.get('target_width', 128),
                train=True,
                augment=data_cfg.get('augment', True),
                image_format=data_cfg.get('image_format', 'numpy'),
                num_samples=data_cfg.get('num_debug_samples', None)
            )

            # Validation dataset (same ordering, no augmentation)
            val_dataset = RSNAVolumeDataset(
                data_dir=data_cfg['dataset_path'],
                target_width=data_cfg.get('target_width', 128),
                train=False,
                augment=False,
                image_format=data_cfg.get('image_format', 'numpy'),
                num_samples=data_cfg.get('num_debug_samples', None)
            )

            # Split indices
            val_split = data_cfg.get('val_split', 0.2)
            random_seed = data_cfg.get('random_seed', 0)
            num_samples = len(dataset)
            num_val = int(round(num_samples * val_split))
            indices = torch.randperm(num_samples, generator=torch.Generator().manual_seed(random_seed)).tolist()
            val_indices = indices[:num_val]
            train_indices = indices[num_val:]

            train_dataset = Subset(dataset, train_indices)
            eval_dataset = Subset(val_dataset, val_indices)
            
            dataloader = DataLoader(
                train_dataset,
                batch_size=data_cfg['batch_size'],
                shuffle=True,
                collate_fn=collate_fn,
                num_workers=data_cfg.get('num_workers', 0),
                persistent_workers=data_cfg.get('persistent_workers', False) and data_cfg.get('num_workers', 0) > 0
            )

            val_loader = DataLoader(
                eval_dataset,
                batch_size=data_cfg.get('val_batch_size', data_cfg['batch_size']),
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=data_cfg.get('num_workers', 0),
                persistent_workers=data_cfg.get('persistent_workers', False) and data_cfg.get('num_workers', 0) > 0
            )
            
            print(f"[✓] 数据集大小: {len(dataset)}")
            print(f"[✓] 训练样本: {len(train_dataset)} | 验证样本: {len(eval_dataset)}")
            print(f"[✓] 训练批数: {len(dataloader)} | 验证批数: {len(val_loader)}")
            
            # 创建检查点目录
            checkpoint_dir = config['checkpoint']['save_dir']
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # 创建日志目录
            log_dir = config['logging']['log_dir']
            log_dir_with_time = os.path.join(
                log_dir,
                datetime.now().strftime(config['logging'].get('log_name', 'retinanet_%Y%m%d_%H%M%S'))
            )
            os.makedirs(log_dir_with_time, exist_ok=True)
            
            # 保存配置文件到日志目录
            save_config(config, os.path.join(log_dir_with_time, 'config.yaml'))
            print(f"[✓] 日志目录: {log_dir_with_time}")
            
            # 创建模型
            print(f"\n[*] 创建模型...")
            model_cfg = config['model']
            loss_cfg = config['loss']
            
            model = RetinaNet3D(
                image_size=tuple(model_cfg.get('image_size', [64, 64, 64])),
                num_classes=model_cfg.get('num_classes', 1),
                in_channels=model_cfg.get('in_channels', 1),
                num_anchors=model_cfg.get('num_anchors', 9)
            ).to(device)
            
            # 更新焦点损失参数
            model.focal_loss = FocalLoss(
                alpha=loss_cfg.get('focal_alpha', 0.25),
                gamma=loss_cfg.get('focal_gamma', 2.0)
            )
            
            print(f"[✓] 模型已创建")
            
            # 恢复检查点
            start_epoch = 0
            if args.checkpoint and os.path.exists(args.checkpoint):
                print(f"\n[*] 加载检查点: {args.checkpoint}")
                checkpoint = torch.load(args.checkpoint, map_location=device)
                model.load_state_dict(checkpoint['model'])
                start_epoch = checkpoint.get('epoch', 0)
                print(f"[✓] 从第 {start_epoch + 1} 轮继续训练")
            
            # 创建优化器
            train_cfg = config['training']
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=train_cfg['lr'],
                weight_decay=train_cfg.get('weight_decay', 0.01)
            )
            
            # 创建学习率调度器
            total_epochs = train_cfg['epochs']
            warmup_epochs = train_cfg.get('warmup_epochs', 10)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=total_epochs - warmup_epochs,
                eta_min=0
            )
            
            print(f"[✓] 优化器: Adam (lr={train_cfg['lr']}, weight_decay={train_cfg.get('weight_decay', 0.01)})")
            print(f"[✓] 学习率调度: Cosine Annealing")
            
            # 训练循环
            print(f"\n" + "=" * 70)
            print(f"开始训练 ({total_epochs} 轮)")
            print("=" * 70)
            
            model.train()
            best_loss = float('inf')
            patience_counter = 0
            early_stop_patience = config['checkpoint'].get('early_stop_patience', 50)
            
            # 创建 metrics.jsonl 文件
            metrics_file = os.path.join(log_dir_with_time, 'metrics.jsonl')
            
            for epoch in range(start_epoch, total_epochs):
                total_loss = 0
                num_batches = 0
                loss_ce_sum = 0
                loss_bbox_sum = 0
                grad_norms = []
                
                for batch_idx, batch in enumerate(dataloader):
                    # 获取批数据 (collate_fn 返回 'volumes' 而不是 'volume')
                    volumes = batch['volumes'].to(device)
                    
                    # 前向传播
                    outputs = model(volumes)

                    # 多尺度 FPN 输出
                    cls_scores = outputs['cls_scores']
                    bbox_preds = outputs['bbox_preds']
                    img_size = tuple(volumes.shape[2:])
                    anchors_per_loc = model_cfg.get('num_anchors', 9)

                    loss = torch.tensor(0.0, device=device)
                    level_cls_loss = 0.0
                    level_bbox_loss = 0.0

                    for lvl, (cls_score_l, bbox_pred_l) in enumerate(zip(cls_scores, bbox_preds)):
                        # 生成该层 anchors
                        _, _, Df, Hf, Wf = cls_score_l.shape
                        sd = img_size[0] / Df
                        sh = img_size[1] / Hf
                        sw = img_size[2] / Wf
                        anchors = model.anchor_generator.generate(
                            device=device,
                            image_size=img_size,
                            feature_stride=(sd, sh, sw),
                            feature_map_size=(Df, Hf, Wf)
                        )  # (Df, Hf, Wf, num_anchors_per_loc, 6)
                        anchors_flat = anchors.reshape(-1, 6)
                        anchor_boxes = anchors_flat

                        # Gather GT boxes for this batch and match per image
                        batch_cls_targets = []
                        batch_bbox_targets = []
                        for b_i in range(volumes.shape[0]):
                            gt_boxes = batch['boxes'][b_i].to(device) if 'boxes' in batch and len(batch['boxes'][b_i])>0 else torch.zeros((0,6), device=device)
                            gt_labels = batch['labels'][b_i].to(device) if 'labels' in batch and len(batch['labels'][b_i])>0 else torch.zeros((0,), dtype=torch.long, device=device)

                            # GT boxes are normalized to [0,1] in dataset. Convert to voxel coords (x, y, z, w, h, d).
                            if gt_boxes.numel() > 0:
                                orig_d, orig_h, orig_w = batch['original_sizes'][b_i]
                                scale = torch.tensor([orig_w, orig_h, orig_d, orig_w, orig_h, orig_d], device=device, dtype=gt_boxes.dtype)
                                gt_boxes = gt_boxes * scale

                            if gt_boxes.numel() == 0:
                                # negative: all zeros
                                cls_t = torch.zeros(len(anchor_boxes), dtype=torch.float32, device=device)
                                bbox_t = torch.zeros(len(anchor_boxes), 6, device=device)
                            else:
                                ious = model.iou_3d(anchor_boxes, gt_boxes)  # (A, G)
                                best_iou, best_idx = ious.max(dim=1)
                                # positive if IoU >= 0.3
                                pos_mask = best_iou >= 0.3
                                cls_t = torch.zeros(len(anchor_boxes), dtype=torch.float32, device=device)
                                cls_t[pos_mask] = 1.0  # binary for single class

                                # bbox targets: RetinaNet-style encoding
                                # dx,dy,dz = (gt_center - anchor_center) / anchor_size
                                # dw,dh,dd = log(gt_size / anchor_size)
                                bbox_t = torch.zeros(len(anchor_boxes), 6, device=device)
                                if pos_mask.any():
                                    matched_gt = gt_boxes[best_idx[pos_mask]]
                                    matched_anchor = anchor_boxes[pos_mask]
                                    dx = (matched_gt[:, 0] - matched_anchor[:, 0]) / (matched_anchor[:, 3] + 1e-8)
                                    dy = (matched_gt[:, 1] - matched_anchor[:, 1]) / (matched_anchor[:, 4] + 1e-8)
                                    dz = (matched_gt[:, 2] - matched_anchor[:, 2]) / (matched_anchor[:, 5] + 1e-8)
                                    dw = torch.log((matched_gt[:, 3] + 1e-8) / (matched_anchor[:, 3] + 1e-8))
                                    dh = torch.log((matched_gt[:, 4] + 1e-8) / (matched_anchor[:, 4] + 1e-8))
                                    dd = torch.log((matched_gt[:, 5] + 1e-8) / (matched_anchor[:, 5] + 1e-8))
                                    bbox_t[pos_mask] = torch.stack([dx, dy, dz, dw, dh, dd], dim=1)

                            # reshape to feature map anchors layout: (num_anchors, Df, Hf, Wf)
                            cls_t = cls_t.view(Df, Hf, Wf, anchors_per_loc).permute(3,0,1,2).contiguous()
                            bbox_t = bbox_t.view(Df, Hf, Wf, anchors_per_loc, 6).permute(3,4,0,1,2).contiguous()

                            batch_cls_targets.append(cls_t)
                            batch_bbox_targets.append(bbox_t.view(anchors_per_loc*6, Df, Hf, Wf))

                        cls_targets = torch.stack(batch_cls_targets, dim=0)
                        bbox_targets = torch.stack(batch_bbox_targets, dim=0)

                        # 计算损失 (每层)
                        losses = model.compute_loss(
                            cls_score_l,
                            bbox_pred_l,
                            cls_targets,
                            bbox_targets
                        )

                        loss = loss + losses['total_loss']
                        level_cls_loss += losses.get('cls_loss', torch.tensor(0)).item()
                        level_bbox_loss += losses.get('bbox_loss', torch.tensor(0)).item()

                    # 记录各个损失分量
                    loss_ce_sum += level_cls_loss
                    loss_bbox_sum += level_bbox_loss
                    
                    optimizer.zero_grad()
                    loss.backward()
                    
                    # 计算梯度范数
                    total_norm = 0
                    for p in model.parameters():
                        if p.grad is not None:
                            total_norm += p.grad.data.norm(2).item() ** 2
                    total_norm = total_norm ** 0.5
                    grad_norms.append(total_norm)
                    
                    # 梯度裁剪
                    if train_cfg.get('clip_max_norm'):
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(),
                            train_cfg['clip_max_norm']
                        )
                    
                    optimizer.step()
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                    if (batch_idx + 1) % train_cfg.get('log_interval', 10) == 0:
                        print(f"[Epoch {epoch+1}/{total_epochs}] "
                              f"Batch {batch_idx+1}/{len(dataloader)} | "
                              f"Loss: {loss.item():.6f}")
                
                # 学习率调度 (warmup 后)
                if epoch >= warmup_epochs:
                    scheduler.step()
                
                avg_loss = total_loss / num_batches
                avg_loss_ce = loss_ce_sum / num_batches
                avg_loss_bbox = loss_bbox_sum / num_batches
                avg_grad_norm = np.mean(grad_norms) if grad_norms else 0
                current_lr = optimizer.param_groups[0]['lr']
                print(f"\n[Epoch {epoch+1}/{total_epochs}] 平均损失: {avg_loss:.6f} | CE: {avg_loss_ce:.6f} | BBox: {avg_loss_bbox:.6f} | Grad: {avg_grad_norm:.4f} | LR: {current_lr:.2e}")

                # 验证集评估
                val_interval = train_cfg.get('val_interval', 1)
                eval_cfg = config.get('eval', {})
                score_thresh = eval_cfg.get('score_thresh', 0.05)
                nms_iou = eval_cfg.get('nms_iou', 0.5)
                max_detections = eval_cfg.get('max_detections', 100)
                iou_thresholds = eval_cfg.get('iou_thresholds', [0.5])

                val_map = None
                if len(val_loader) > 0 and (epoch + 1) % val_interval == 0:
                    model.eval()
                    predictions = []
                    ground_truths = []

                    with torch.no_grad():
                        for val_batch in val_loader:
                            val_volumes = val_batch['volumes'].to(device)
                            val_outputs = model(val_volumes)

                            cls_scores = val_outputs['cls_scores']
                            bbox_preds = val_outputs['bbox_preds']
                            val_img_size = tuple(val_volumes.shape[2:])

                            # Per-image aggregation
                            for b_i in range(val_volumes.shape[0]):
                                all_boxes = []
                                all_scores = []
                                all_labels = []

                                for cls_score_l, bbox_pred_l in zip(cls_scores, bbox_preds):
                                    _, _, Df, Hf, Wf = cls_score_l.shape
                                    sd = val_img_size[0] / Df
                                    sh = val_img_size[1] / Hf
                                    sw = val_img_size[2] / Wf
                                    anchors = model.anchor_generator.generate(
                                        device=device,
                                        image_size=val_img_size,
                                        feature_stride=(sd, sh, sw),
                                        feature_map_size=(Df, Hf, Wf)
                                    )
                                    anchor_boxes = anchors.reshape(-1, 6)

                                    # Reshape predictions
                                    cls_l = cls_score_l[b_i].view(
                                        anchors_per_loc, model.num_classes, Df, Hf, Wf
                                    ).permute(2, 3, 4, 0, 1).contiguous().view(-1, model.num_classes)
                                    bbox_l = bbox_pred_l[b_i].view(
                                        anchors_per_loc, 6, Df, Hf, Wf
                                    ).permute(2, 3, 4, 0, 1).contiguous().view(-1, 6)

                                    scores = torch.sigmoid(cls_l)
                                    decoded = model.decode_boxes(anchor_boxes, bbox_l)

                                    for cls_id in range(model.num_classes):
                                        cls_scores_c = scores[:, cls_id]
                                        keep = cls_scores_c > score_thresh
                                        if keep.sum() == 0:
                                            continue
                                        boxes_c = decoded[keep]
                                        scores_c = cls_scores_c[keep]
                                        keep_idx = model.nms_3d(boxes_c, scores_c, nms_iou)
                                        boxes_c = boxes_c[keep_idx]
                                        scores_c = scores_c[keep_idx]
                                        labels_c = torch.full((boxes_c.shape[0],), cls_id, device=device, dtype=torch.long)

                                        all_boxes.append(boxes_c)
                                        all_scores.append(scores_c)
                                        all_labels.append(labels_c)

                                if len(all_boxes) == 0:
                                    pred_boxes = torch.zeros((0, 6), device=device)
                                    pred_scores = torch.zeros((0,), device=device)
                                    pred_labels = torch.zeros((0,), device=device, dtype=torch.long)
                                else:
                                    pred_boxes = torch.cat(all_boxes, dim=0)
                                    pred_scores = torch.cat(all_scores, dim=0)
                                    pred_labels = torch.cat(all_labels, dim=0)

                                    # top-k
                                    if pred_scores.numel() > max_detections:
                                        topk = torch.topk(pred_scores, k=max_detections)
                                        pred_boxes = pred_boxes[topk.indices]
                                        pred_scores = topk.values
                                        pred_labels = pred_labels[topk.indices]

                                # GT boxes in voxel coords
                                gt_boxes = val_batch['boxes'][b_i].to(device)
                                gt_labels = val_batch['labels'][b_i].to(device)
                                if gt_boxes.numel() > 0:
                                    orig_d, orig_h, orig_w = val_batch['original_sizes'][b_i]
                                    scale = torch.tensor([orig_w, orig_h, orig_d, orig_w, orig_h, orig_d], device=device, dtype=gt_boxes.dtype)
                                    gt_boxes = gt_boxes * scale

                                predictions.append({
                                    'boxes': pred_boxes.cpu().numpy(),
                                    'scores': pred_scores.cpu().numpy(),
                                    'labels': pred_labels.cpu().numpy()
                                })
                                ground_truths.append({
                                    'boxes': gt_boxes.cpu().numpy(),
                                    'labels': gt_labels.cpu().numpy()
                                })

                    map_results = compute_map(
                        predictions,
                        ground_truths,
                        num_classes=model.num_classes,
                        iou_thresholds=iou_thresholds
                    )
                    val_map = map_results.get(f"mAP@{iou_thresholds[0]}", map_results.get('mAP', 0.0))
                    print(f"[Epoch {epoch+1}/{total_epochs}] 验证 mAP@{iou_thresholds[0]}: {val_map:.4f}")
                    model.train()
                
                # 保存 metrics 到 JSONL 文件
                metrics_entry = {
                    'timestamp': datetime.now().isoformat(),
                    'epoch': epoch + 1,
                    'phase': 'train',
                    'loss_ce': avg_loss_ce,
                    'loss_bbox': avg_loss_bbox,
                    'loss_total': avg_loss,
                    'grad_norm': avg_grad_norm,
                    'lr': current_lr
                }
                with open(metrics_file, 'a') as f:
                    f.write(json.dumps(metrics_entry) + '\n')

                if val_map is not None:
                    val_entry = {
                        'timestamp': datetime.now().isoformat(),
                        'epoch': epoch + 1,
                        'phase': 'val',
                        'mAP': val_map,
                        f"mAP@{iou_thresholds[0]}": val_map
                    }
                    with open(metrics_file, 'a') as f:
                        f.write(json.dumps(val_entry) + '\n')
                
                # 检查点保存
                if (epoch + 1) % train_cfg.get('save_interval', 10) == 0:
                    checkpoint_path = os.path.join(
                        checkpoint_dir,
                        f'checkpoint_epoch_{epoch+1}.pth'
                    )
                    torch.save({
                        'epoch': epoch+1,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'config': config,
                        'loss': avg_loss
                    }, checkpoint_path)
                    print(f"[✓] 检查点已保存: {checkpoint_path}")
                
                # 最佳模型保存
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    patience_counter = 0
                    
                    best_path = os.path.join(checkpoint_dir, 'model_best.pth')
                    torch.save({
                        'epoch': epoch,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'config': config,
                        'loss': avg_loss
                    }, best_path)
                    print(f"[✓] 最佳模型已更新: {best_path}")
                
                else:
                    patience_counter += 1
                    if config['checkpoint'].get('early_stop') and patience_counter >= early_stop_patience:
                        print(f"\n[!] 早停: 在 {early_stop_patience} 轮内未见改进")
                        break
                
                print()
            
            # 最终模型保存
            final_path = os.path.join(checkpoint_dir, 'model_final.pth')
            torch.save({
                'epoch': total_epochs,
                'model': model.state_dict(),
                'config': config,
                'loss': avg_loss
            }, final_path)
            print(f"[✓] 最终模型已保存: {final_path}")
            
            print("\n" + "=" * 70)
            print("训练完成！")
            print("=" * 70)
            print(f"结果保存目录: {log_dir_with_time}")
        
        except Exception as e:
            print(f"\n[✗] 错误: {e}")
            import traceback
            traceback.print_exc()
