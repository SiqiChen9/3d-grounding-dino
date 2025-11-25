"""
Loss functions for 3D Grounding-DETR.
Includes Hungarian matching and set-based losses.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from typing import List, Dict, Tuple
import numpy as np


def box_iou_3d(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute 3D IoU between two sets of boxes.
    
    Args:
        boxes1: (N, 6) in format (cx, cy, cz, w, h, d)
        boxes2: (M, 6) in format (cx, cy, cz, w, h, d)
    
    Returns:
        (N, M) IoU matrix
    """
    # Convert to corner format (x1, y1, z1, x2, y2, z2)
    boxes1_corners = torch.zeros_like(boxes1)
    boxes1_corners[:, 0] = boxes1[:, 0] - boxes1[:, 3] / 2
    boxes1_corners[:, 1] = boxes1[:, 1] - boxes1[:, 4] / 2
    boxes1_corners[:, 2] = boxes1[:, 2] - boxes1[:, 5] / 2
    boxes1_corners[:, 3] = boxes1[:, 0] + boxes1[:, 3] / 2
    boxes1_corners[:, 4] = boxes1[:, 1] + boxes1[:, 4] / 2
    boxes1_corners[:, 5] = boxes1[:, 2] + boxes1[:, 5] / 2
    
    boxes2_corners = torch.zeros_like(boxes2)
    boxes2_corners[:, 0] = boxes2[:, 0] - boxes2[:, 3] / 2
    boxes2_corners[:, 1] = boxes2[:, 1] - boxes2[:, 4] / 2
    boxes2_corners[:, 2] = boxes2[:, 2] - boxes2[:, 5] / 2
    boxes2_corners[:, 3] = boxes2[:, 0] + boxes2[:, 3] / 2
    boxes2_corners[:, 4] = boxes2[:, 1] + boxes2[:, 4] / 2
    boxes2_corners[:, 5] = boxes2[:, 2] + boxes2[:, 5] / 2
    
    # Compute intersection
    lt = torch.max(
        boxes1_corners[:, None, :3],
        boxes2_corners[None, :, :3]
    )  # (N, M, 3)
    
    rb = torch.min(
        boxes1_corners[:, None, 3:],
        boxes2_corners[None, :, 3:]
    )  # (N, M, 3)
    
    wh = (rb - lt).clamp(min=0)  # (N, M, 3)
    inter = wh[:, :, 0] * wh[:, :, 1] * wh[:, :, 2]  # (N, M)
    
    # Compute union
    area1 = boxes1[:, 3] * boxes1[:, 4] * boxes1[:, 5]  # (N,)
    area2 = boxes2[:, 3] * boxes2[:, 4] * boxes2[:, 5]  # (M,)
    union = area1[:, None] + area2[None, :] - inter  # (N, M)
    
    iou = inter / (union + 1e-6)
    return iou


def generalized_box_iou_3d(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute generalized 3D IoU (GIoU).
    
    Args:
        boxes1: (N, 6)
        boxes2: (M, 6)
    
    Returns:
        (N, M) GIoU matrix
    """
    # Standard IoU
    iou = box_iou_3d(boxes1, boxes2)
    
    # Convert to corners
    boxes1_corners = torch.zeros((boxes1.shape[0], 6), device=boxes1.device)
    boxes1_corners[:, 0] = boxes1[:, 0] - boxes1[:, 3] / 2
    boxes1_corners[:, 1] = boxes1[:, 1] - boxes1[:, 4] / 2
    boxes1_corners[:, 2] = boxes1[:, 2] - boxes1[:, 5] / 2
    boxes1_corners[:, 3] = boxes1[:, 0] + boxes1[:, 3] / 2
    boxes1_corners[:, 4] = boxes1[:, 1] + boxes1[:, 4] / 2
    boxes1_corners[:, 5] = boxes1[:, 2] + boxes1[:, 5] / 2
    
    boxes2_corners = torch.zeros((boxes2.shape[0], 6), device=boxes2.device)
    boxes2_corners[:, 0] = boxes2[:, 0] - boxes2[:, 3] / 2
    boxes2_corners[:, 1] = boxes2[:, 1] - boxes2[:, 4] / 2
    boxes2_corners[:, 2] = boxes2[:, 2] - boxes2[:, 5] / 2
    boxes2_corners[:, 3] = boxes2[:, 0] + boxes2[:, 3] / 2
    boxes2_corners[:, 4] = boxes2[:, 1] + boxes2[:, 4] / 2
    boxes2_corners[:, 5] = boxes2[:, 2] + boxes2[:, 5] / 2
    
    # Enclosing box
    lt = torch.min(
        boxes1_corners[:, None, :3],
        boxes2_corners[None, :, :3]
    )
    rb = torch.max(
        boxes1_corners[:, None, 3:],
        boxes2_corners[None, :, 3:]
    )
    
    wh = (rb - lt).clamp(min=0)
    enclosing_volume = wh[:, :, 0] * wh[:, :, 1] * wh[:, :, 2]
    
    # Compute union
    area1 = boxes1[:, 3] * boxes1[:, 4] * boxes1[:, 5]
    area2 = boxes2[:, 3] * boxes2[:, 4] * boxes2[:, 5]
    union = area1[:, None] + area2[None, :] - iou * (area1[:, None] + area2[None, :] - iou)
    
    # GIoU
    giou = iou - (enclosing_volume - union) / (enclosing_volume + 1e-6)
    
    return giou


class HungarianMatcher(nn.Module):
    """
    Hungarian matching between predictions and ground truth.
    """
    
    def __init__(
        self,
        cost_class: float = 1.0,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0
    ):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
    
    @torch.no_grad()
    def forward(
        self,
        pred_logits: torch.Tensor,
        pred_boxes: torch.Tensor,
        target_labels: List[torch.Tensor],
        target_boxes: List[torch.Tensor]
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            pred_logits: (B, num_queries, num_classes+1)
            pred_boxes: (B, num_queries, 6)
            target_labels: List of (num_targets_i,)
            target_boxes: List of (num_targets_i, 6)
        
        Returns:
            List of (src_idx, tgt_idx) tuples for each batch
        """
        batch_size, num_queries = pred_logits.shape[:2]
        
        # Flatten batch dimension
        pred_logits_flat = pred_logits.flatten(0, 1)  # (B*num_queries, num_classes+1)
        pred_boxes_flat = pred_boxes.flatten(0, 1)  # (B*num_queries, 6)
        
        # Compute costs for each batch element
        indices = []
        
        for i in range(batch_size):
            # Get targets for this batch
            tgt_labels = target_labels[i]
            tgt_boxes = target_boxes[i]
            
            if len(tgt_labels) == 0:
                # No targets, no matching
                indices.append((torch.tensor([], dtype=torch.int64), 
                              torch.tensor([], dtype=torch.int64)))
                continue
            
            # Get predictions for this batch
            pred_logits_i = pred_logits[i]  # (num_queries, num_classes+1)
            pred_boxes_i = pred_boxes[i]  # (num_queries, 6)
            
            # Classification cost
            pred_probs = F.softmax(pred_logits_i, dim=-1)  # (num_queries, num_classes+1)
            cost_class = -pred_probs[:, tgt_labels]  # (num_queries, num_targets)
            
            # Box L1 cost
            cost_bbox = torch.cdist(pred_boxes_i, tgt_boxes, p=1)  # (num_queries, num_targets)
            
            # Box GIoU cost
            cost_giou = -generalized_box_iou_3d(pred_boxes_i, tgt_boxes)  # (num_queries, num_targets)
            
            # Final cost matrix
            C = (
                self.cost_class * cost_class +
                self.cost_bbox * cost_bbox +
                self.cost_giou * cost_giou
            )
            C = C.cpu().numpy()
            
            # Hungarian algorithm
            src_idx, tgt_idx = linear_sum_assignment(C)
            
            indices.append((
                torch.tensor(src_idx, dtype=torch.int64),
                torch.tensor(tgt_idx, dtype=torch.int64)
            ))
        
        return indices


class SetCriterion(nn.Module):
    """
    Set-based loss for DETR.
    Computes classification and box regression losses.
    """
    
    def __init__(
        self,
        num_classes: int,
        matcher: HungarianMatcher,
        weight_dict: Dict[str, float],
        eos_coef: float = 0.1
    ):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        
        # Class weights (down-weight background)
        empty_weight = torch.ones(num_classes + 1)
        empty_weight[-1] = eos_coef
        self.register_buffer('empty_weight', empty_weight)
    
    def loss_labels(
        self,
        pred_logits: torch.Tensor,
        target_labels: List[torch.Tensor],
        indices: List[Tuple]
    ) -> Dict[str, torch.Tensor]:
        """Classification loss."""
        batch_size, num_queries = pred_logits.shape[:2]
        
        # Prepare targets
        target_classes = torch.full(
            (batch_size, num_queries),
            self.num_classes,  # Background class
            dtype=torch.int64,
            device=pred_logits.device
        )
        
        for i, (src_idx, tgt_idx) in enumerate(indices):
            if len(src_idx) > 0:
                target_classes[i, src_idx] = target_labels[i][tgt_idx]
        
        # Cross-entropy loss
        loss_ce = F.cross_entropy(
            pred_logits.transpose(1, 2),
            target_classes,
            self.empty_weight
        )
        
        return {'loss_ce': loss_ce}
    
    def loss_boxes(
        self,
        pred_boxes: torch.Tensor,
        target_boxes: List[torch.Tensor],
        indices: List[Tuple]
    ) -> Dict[str, torch.Tensor]:
        """Box regression loss (L1 + GIoU)."""
        # Gather matched predictions and targets
        src_boxes = []
        tgt_boxes = []
        
        for i, (src_idx, tgt_idx) in enumerate(indices):
            if len(src_idx) > 0:
                src_boxes.append(pred_boxes[i, src_idx])
                tgt_boxes.append(target_boxes[i][tgt_idx])
        
        if len(src_boxes) == 0:
            # No boxes to match
            return {
                'loss_l1': pred_boxes.sum() * 0,
                'loss_giou': pred_boxes.sum() * 0
            }
        
        src_boxes = torch.cat(src_boxes, dim=0)
        tgt_boxes = torch.cat(tgt_boxes, dim=0)
        
        # L1 loss
        loss_l1 = F.l1_loss(src_boxes, tgt_boxes, reduction='mean')
        
        # GIoU loss
        giou = torch.diag(generalized_box_iou_3d(src_boxes, tgt_boxes))
        loss_giou = (1 - giou).mean()
        
        return {
            'loss_l1': loss_l1,
            'loss_giou': loss_giou
        }
    
    def forward(
        self,
        pred_logits: torch.Tensor,
        pred_boxes: torch.Tensor,
        target_labels: List[torch.Tensor],
        target_boxes: List[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all losses.
        
        Args:
            pred_logits: (B, num_queries, num_classes+1)
            pred_boxes: (B, num_queries, 6)
            target_labels: List of (num_targets_i,)
            target_boxes: List of (num_targets_i, 6)
        
        Returns:
            Dict of losses
        """
        # Hungarian matching
        indices = self.matcher(pred_logits, pred_boxes, target_labels, target_boxes)
        
        # Compute losses
        losses = {}
        losses.update(self.loss_labels(pred_logits, target_labels, indices))
        losses.update(self.loss_boxes(pred_boxes, target_boxes, indices))
        
        # Weighted loss
        weighted_losses = {
            k: v * self.weight_dict.get(k, 1.0)
            for k, v in losses.items()
        }
        weighted_losses['loss_total'] = sum(weighted_losses.values())
        
        return weighted_losses
