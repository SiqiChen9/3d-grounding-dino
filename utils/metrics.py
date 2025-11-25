"""
Evaluation metrics for 3D object detection.
Includes IoU computation and mAP calculation.
"""
import torch
import numpy as np
from typing import List, Dict, Tuple


def compute_iou_3d(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Compute 3D IoU between two boxes.
    
    Args:
        box1: (6,) in format (cx, cy, cz, w, h, d)
        box2: (6,) in format (cx, cy, cz, w, h, d)
    
    Returns:
        IoU score
    """
    # Convert to corners
    box1_min = box1[:3] - box1[3:] / 2
    box1_max = box1[:3] + box1[3:] / 2
    
    box2_min = box2[:3] - box2[3:] / 2
    box2_max = box2[:3] + box2[3:] / 2
    
    # Intersection
    inter_min = np.maximum(box1_min, box2_min)
    inter_max = np.minimum(box1_max, box2_max)
    inter_size = np.maximum(0, inter_max - inter_min)
    inter_volume = np.prod(inter_size)
    
    # Union
    box1_volume = np.prod(box1[3:])
    box2_volume = np.prod(box2[3:])
    union_volume = box1_volume + box2_volume - inter_volume
    
    iou = inter_volume / (union_volume + 1e-6)
    return iou


def compute_ap(
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    gt_boxes: np.ndarray,
    iou_threshold: float = 0.5
) -> float:
    """
    Compute Average Precision for a single class.
    
    Args:
        pred_boxes: (N, 6) predicted boxes
        pred_scores: (N,) confidence scores
        gt_boxes: (M, 6) ground truth boxes
        iou_threshold: IoU threshold for positive match
    
    Returns:
        Average Precision
    """
    if len(pred_boxes) == 0:
        return 0.0
    
    if len(gt_boxes) == 0:
        return 0.0
    
    # Sort by score
    sorted_indices = np.argsort(-pred_scores)
    pred_boxes = pred_boxes[sorted_indices]
    pred_scores = pred_scores[sorted_indices]
    
    # Match predictions to ground truth
    num_gt = len(gt_boxes)
    matched_gt = np.zeros(num_gt, dtype=bool)
    
    tp = np.zeros(len(pred_boxes))
    fp = np.zeros(len(pred_boxes))
    
    for i, pred_box in enumerate(pred_boxes):
        # Find best matching GT box
        best_iou = 0
        best_gt_idx = -1
        
        for j, gt_box in enumerate(gt_boxes):
            if matched_gt[j]:
                continue
            
            iou = compute_iou_3d(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j
        
        # Check if match is good enough
        if best_iou >= iou_threshold and best_gt_idx >= 0:
            if not matched_gt[best_gt_idx]:
                tp[i] = 1
                matched_gt[best_gt_idx] = True
            else:
                fp[i] = 1
        else:
            fp[i] = 1
    
    # Compute precision and recall
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    
    recalls = tp_cumsum / num_gt
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
    
    # Compute AP (area under PR curve)
    ap = 0
    for i in range(len(precisions) - 1):
        ap += (recalls[i+1] - recalls[i]) * precisions[i+1]
    
    return ap


def compute_map(
    predictions: List[Dict],
    ground_truths: List[Dict],
    num_classes: int,
    iou_thresholds: List[float] = [0.1, 0.3, 0.5]
) -> Dict[str, float]:
    """
    Compute mean Average Precision across classes and IoU thresholds.
    
    Args:
        predictions: List of dicts with 'boxes', 'scores', 'labels'
        ground_truths: List of dicts with 'boxes', 'labels'
        num_classes: Number of classes
        iou_thresholds: List of IoU thresholds to evaluate
    
    Returns:
        Dict with mAP scores
    """
    results = {}
    
    for iou_thresh in iou_thresholds:
        aps = []
        
        for class_id in range(num_classes):
            # Gather all predictions and GT for this class
            class_pred_boxes = []
            class_pred_scores = []
            class_gt_boxes = []
            
            for pred, gt in zip(predictions, ground_truths):
                # Predictions for this class
                pred_mask = pred['labels'] == class_id
                if pred_mask.sum() > 0:
                    class_pred_boxes.append(pred['boxes'][pred_mask])
                    class_pred_scores.append(pred['scores'][pred_mask])
                
                # Ground truth for this class
                gt_mask = gt['labels'] == class_id
                if gt_mask.sum() > 0:
                    class_gt_boxes.append(gt['boxes'][gt_mask])
            
            if len(class_pred_boxes) == 0 or len(class_gt_boxes) == 0:
                continue
            
            # Concatenate
            class_pred_boxes = np.concatenate(class_pred_boxes, axis=0)
            class_pred_scores = np.concatenate(class_pred_scores, axis=0)
            class_gt_boxes = np.concatenate(class_gt_boxes, axis=0)
            
            # Compute AP
            ap = compute_ap(
                class_pred_boxes,
                class_pred_scores,
                class_gt_boxes,
                iou_threshold=iou_thresh
            )
            aps.append(ap)
        
        # Mean AP
        if len(aps) > 0:
            map_score = np.mean(aps)
        else:
            map_score = 0.0
        
        results[f'mAP@{iou_thresh}'] = map_score
    
    # Overall mAP (average across thresholds)
    results['mAP'] = np.mean([results[f'mAP@{t}'] for t in iou_thresholds])
    
    return results
