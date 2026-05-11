"""
3D RetinaNet 结果可视化脚本
用于生成检测结果的 2D 切片可视化图像
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import yaml
from tqdm import tqdm

# 导入模型和数据集
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Retinanet.retinanet_3d import RetinaNet3D, load_config
from datasets import RSNAVolumeDataset
from utils.visualization import (
    denormalize_box_3d,
    box_3d_to_2d_slice,
    CLASS_COLORS,
    CLASS_NAMES
)


class RetinaNetVisualizer:
    """RetinaNet3D 检测结果可视化器"""
    
    def __init__(
        self,
        checkpoint_path: str,
        config_path: str,
        device: str = 'cuda',
        conf_threshold: float = 0.5,
        nms_threshold: float = 0.5
    ):
        if str(device).startswith('cuda') and not torch.cuda.is_available():
            print("[!] CUDA 不可用，自动切换到 CPU")
            device = 'cpu'

        self.device = device
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        
        # 加载配置
        print("[*] 加载配置文件...")
        self.config = load_config(config_path)
        
        # 创建模型
        print("[*] 创建模型...")
        self.model = RetinaNet3D(
            num_classes=self.config['model']['num_classes'],
            num_anchors=self.config['model']['num_anchors']
        )
        
        # 加载检查点
        print("[*] 加载检查点...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        self.model.load_state_dict(state_dict)
        
        self.model = self.model.to(device)
        self.model.eval()
        print("[✓] 模型加载完成")
    
    def nms_3d(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        threshold: float = 0.5
    ) -> torch.Tensor:
        """
        3D NMS (非极大值抑制)
        
        Args:
            boxes: (N, 6) - [cx, cy, cz, w, h, d]
            scores: (N,)
            threshold: IoU 阈值
        
        Returns:
            保留的索引
        """
        if len(boxes) == 0:
            return torch.tensor([], dtype=torch.long, device=boxes.device)
        
        # 简单的 NMS 实现（基于中心距离）
        # 完整的 3D IoU NMS 可以后续优化
        keep = []
        idx = scores.argsort(descending=True)
        
        while len(idx) > 0:
            current = idx[0]
            keep.append(current)
            
            if len(idx) == 1:
                break
            
            # 计算中心点距离
            current_box = boxes[current]
            other_boxes = boxes[idx[1:]]
            
            # 简化版本：使用中心距离和大小重叠判断
            dist = torch.sqrt(
                ((other_boxes[:, :3] - current_box[:3]) ** 2).sum(dim=1)
            )
            size_sim = 1 - torch.abs(
                (other_boxes[:, 3:] - current_box[3:]).sum(dim=1) / 
                (current_box[3:].sum() + 1e-6)
            )
            
            # 移除相似的框
            suppress = (dist < current_box[3:].mean()) & (size_sim > 0.5)
            idx = idx[1:][~suppress]
        
        return torch.tensor(keep, dtype=torch.long, device=boxes.device)
    
    def postprocess_predictions(
        self,
        cls_scores: torch.Tensor,
        bbox_preds: torch.Tensor,
        anchors: torch.Tensor,
        feature_stride: int = 8,
        volume_shape: Tuple[int, int, int] = (64, 64, 64)
    ) -> List[Dict]:
        """
        后处理模型输出
        
        Args:
            cls_scores: (B, num_anchors*num_classes, D, H, W)
            bbox_preds: (B, num_anchors*6, D, H, W)
            anchors: (FD, FH, FW, num_anchors, 6)
            feature_stride: 特征步长
            volume_shape: 原始体积形状
        
        Returns:
            检测结果列表
        """
        B = cls_scores.shape[0]
        num_anchors = self.config['model']['num_anchors']
        num_classes = self.config['model']['num_classes']
        
        detections = []
        
        for b in range(B):
            # 处理类别分数
            cls_score = cls_scores[b]  # (num_anchors*num_classes, D, H, W)
            bbox_pred = bbox_preds[b]  # (num_anchors*6, D, H, W)
            
            # 重新形状
            FD, FH, FW = cls_score.shape[1:]
            cls_score = cls_score.permute(1, 2, 3, 0).reshape(FD, FH, FW, num_anchors, num_classes)
            bbox_pred = bbox_pred.permute(1, 2, 3, 0).reshape(FD, FH, FW, num_anchors, 6)
            
            # 获取最大分数的类别
            class_scores, class_ids = cls_score.max(dim=-1)  # (FD, FH, FW, num_anchors)
            
            # 展平
            class_scores_flat = class_scores.reshape(-1)
            class_ids_flat = class_ids.reshape(-1)
            bbox_pred_flat = bbox_pred.reshape(-1, 6)
            
            # 筛选高分数的检测
            mask = class_scores_flat > self.conf_threshold
            class_scores_filtered = class_scores_flat[mask]
            class_ids_filtered = class_ids_flat[mask]
            bbox_pred_filtered = bbox_pred_flat[mask]
            
            if len(class_scores_filtered) == 0:
                detections.append({
                    'boxes': torch.tensor([], device=self.device),
                    'scores': torch.tensor([], device=self.device),
                    'class_ids': torch.tensor([], device=self.device)
                })
                continue
            
            # 将预测的边界框偏移应用到锚点
            # 简化版本：直接使用预测的边界框
            boxes = bbox_pred_filtered * torch.tensor(
                [volume_shape[2], volume_shape[1], volume_shape[0],
                 volume_shape[2], volume_shape[1], volume_shape[0]],
                device=self.device, dtype=torch.float32
            ).clamp(min=1.0)
            
            # NMS
            keep_idx = self.nms_3d(boxes, class_scores_filtered, self.nms_threshold)
            
            detections.append({
                'boxes': boxes[keep_idx],
                'scores': class_scores_filtered[keep_idx],
                'class_ids': class_ids_filtered[keep_idx]
            })
        
        return detections
    
    @torch.no_grad()
    def visualize_sample(
        self,
        volume: np.ndarray,
        mask: np.ndarray = None,
        sample_id: int = 0,
        save_path: str = None,
        slices: List[int] = None
    ) -> Tuple[matplotlib.figure.Figure, List[Dict]]:
        """
        可视化单个样本的检测结果
        
        Args:
            volume: (D, H, W) 体积
            mask: (D, H, W) 分割掩码
            sample_id: 样本id（仅用于标题）
            save_path: 保存路径
            slices: 要显示的切片索引
        
        Returns:
            图表和检测结果
        """
        volume = np.asarray(volume)
        if volume.ndim == 4 and volume.shape[0] == 1:
            volume = volume[0]
        if volume.ndim != 3:
            raise ValueError(
                f"volume 必须是 (D,H,W) 或 (1,D,H,W)，当前形状: {tuple(volume.shape)}"
            )

        if mask is not None:
            mask = np.asarray(mask)
            if mask.ndim == 4 and mask.shape[0] == 1:
                mask = mask[0]

        # 准备输入
        volume_tensor = torch.from_numpy(volume).float().unsqueeze(0).unsqueeze(0)
        volume_tensor = volume_tensor.to(self.device)
        
        # 模型推理
        with torch.no_grad():
            outputs = self.model(volume_tensor)

        if isinstance(outputs, dict):
            cls_scores = outputs['cls_score']
            bbox_preds = outputs['bbox_pred']
        elif isinstance(outputs, (tuple, list)) and len(outputs) >= 2:
            cls_scores, bbox_preds = outputs[:2]
        else:
            raise ValueError(f"未知模型输出格式: {type(outputs)}")
        
        # 后处理
        detections = self.postprocess_predictions(cls_scores, bbox_preds, None, volume_shape=volume.shape)
        det = detections[0]
        
        # 选择切片
        if slices is None:
            D = volume.shape[0]
            slices = [D // 4, D // 2, 3 * D // 4]  # 前、中、后
        
        # 创建图表
        n_slices = len(slices)
        fig, axes = plt.subplots(1, n_slices, figsize=(6 * n_slices, 6))
        if n_slices == 1:
            axes = [axes]
        
        fig.suptitle(f'3D RetinaNet 检测 - 样本 {sample_id}', fontsize=16, fontweight='bold')
        
        # 对每个切片进行可视化
        for idx, slice_idx in enumerate(slices):
            ax = axes[idx]
            
            # 显示体积切片
            slice_img = volume[slice_idx]
            # 归一化到 [0, 1]
            slice_img = (slice_img - slice_img.min()) / (slice_img.max() - slice_img.min() + 1e-6)
            ax.imshow(slice_img, cmap='gray')
            
            # 显示分割掩码（如果有）
            if mask is not None:
                mask_slice = mask[slice_idx]
                for class_id in range(1, 6):
                    class_mask = (mask_slice == class_id).astype(np.float32)
                    if class_mask.sum() > 0:
                        color = CLASS_COLORS.get(class_id - 1, (0.5, 0.5, 0.5))
                        ax.contour(class_mask, levels=[0.5], colors=color, linewidths=2)
            
            # 显示检测框
            if len(det['boxes']) > 0:
                for box_idx, (box, score, class_id) in enumerate(zip(
                    det['boxes'].cpu().numpy(),
                    det['scores'].cpu().numpy(),
                    det['class_ids'].cpu().numpy()
                )):
                    # 检查框是否与该切片相交
                    cx, cy, cz, w, h, d = box
                    z_min = max(0, int(cz - d / 2))
                    z_max = min(volume.shape[0], int(cz + d / 2))
                    
                    if z_min <= slice_idx <= z_max:
                        # 在切片上绘制矩形框
                        x_min = max(0, int(cx - w / 2))
                        x_max = min(volume.shape[2], int(cx + w / 2))
                        y_min = max(0, int(cy - h / 2))
                        y_max = min(volume.shape[1], int(cy + h / 2))
                        
                        color = CLASS_COLORS.get(int(class_id), (1, 1, 1))
                        rect = patches.Rectangle(
                            (x_min, y_min), x_max - x_min, y_max - y_min,
                            linewidth=2, edgecolor=color, facecolor='none'
                        )
                        ax.add_patch(rect)
                        
                        # 添加标签
                        class_name = CLASS_NAMES.get(int(class_id), 'unknown')
                        ax.text(
                            x_min, y_min - 5,
                            f'{class_name}: {score:.2f}',
                            fontsize=8,
                            color='white',
                            bbox=dict(boxstyle='round', facecolor=color, alpha=0.7)
                        )
            
            ax.set_title(f'Slice {slice_idx}/{volume.shape[0]}', fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
        
        plt.tight_layout()
        
        if save_path:
            print(f"[✓] 保存可视化到: {save_path}")
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig, detections
    
    def visualize_dataset(
        self,
        dataset: RSNAVolumeDataset,
        output_dir: str,
        num_samples: int = 5
    ):
        """
        可视化数据集中的多个样本
        
        Args:
            dataset: 数据集
            output_dir: 输出目录
            num_samples: 要可视化的样本数
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n[*] 可视化 {num_samples} 个样本...")
        
        for i in range(min(num_samples, len(dataset))):
            print(f"[*] 处理样本 {i+1}/{num_samples}...")
            
            sample = dataset[i]
            volume = sample['volume']  # (D, H, W)
            mask = sample.get('mask', None)  # (D, H, W)
            
            # 转换为 numpy
            if isinstance(volume, torch.Tensor):
                volume = volume.numpy()
            if isinstance(mask, torch.Tensor):
                mask = mask.numpy()
            
            # 生成可视化
            save_path = output_dir / f"retinanet_sample_{i:03d}.png"
            fig, detections = self.visualize_sample(
                volume, mask, sample_id=i, save_path=str(save_path)
            )
            plt.close(fig)
        
        print(f"[✓] 可视化完成！结果保存在: {output_dir}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='3D RetinaNet 结果可视化')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/retinanet/model_final.pth',
                       help='模型检查点路径')
    parser.add_argument('--config', type=str, default='./Retinanet/retinanet_config.yaml',
                       help='配置文件路径')
    parser.add_argument('--data-dir', type=str, default='/home/woody/iwi5/iwi5378h/rsna2023',
                       help='数据集目录')
    parser.add_argument('--output-dir', type=str, default='./results/retinanet_viz',
                       help='输出目录')
    parser.add_argument('--num-samples', type=int, default=5,
                       help='要可视化的样本数')
    parser.add_argument('--conf-threshold', type=float, default=0.5,
                       help='置信度阈值')
    parser.add_argument('--device', type=str, default='cuda',
                       help='设备')
    
    args = parser.parse_args()
    
    # 创建可视化器
    visualizer = RetinaNetVisualizer(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        device=args.device,
        conf_threshold=args.conf_threshold
    )
    
    # 加载数据集
    print("[*] 加载数据集...")
    dataset = RSNAVolumeDataset(
        data_dir=args.data_dir,
        target_width=64,  # 与训练配置一致
        train=False,
        image_format='numpy'
    )
    print(f"[✓] 数据集大小: {len(dataset)}")
    
    # 可视化
    visualizer.visualize_dataset(
        dataset=dataset,
        output_dir=args.output_dir,
        num_samples=args.num_samples
    )


if __name__ == '__main__':
    main()
