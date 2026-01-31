"""
Evaluation metrics for 3D object detection.
Includes IoU computation and mAP calculation.
"""
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


def compute_tp_fp_per_sample(
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    gt_boxes: np.ndarray,
    iou_threshold: float = 0.5
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Compute TP/FP for predictions against GT boxes for a single sample.
    Predictions are matched to GT in descending score order (greedy matching).
    
    Args:
        pred_boxes: (N, 6) predicted boxes
        pred_scores: (N,) confidence scores
        gt_boxes: (M, 6) ground truth boxes
        iou_threshold: IoU threshold for positive match
    
    Returns:
        Tuple of (tp_array, scores_array, num_gt) - tp/scores sorted by descending score
    """
    num_gt = len(gt_boxes)
    
    if len(pred_boxes) == 0:
        return np.array([]), np.array([]), num_gt
    
    if num_gt == 0:
        # All predictions are false positives
        sorted_idx = np.argsort(-pred_scores)
        return np.zeros(len(pred_boxes)), pred_scores[sorted_idx], 0
    
    # Sort predictions by score (descending) - high confidence first
    sorted_idx = np.argsort(-pred_scores)
    pred_boxes = pred_boxes[sorted_idx]
    pred_scores = pred_scores[sorted_idx]
    
    # Match predictions to ground truth (greedy: high score first)
    matched_gt = np.zeros(num_gt, dtype=bool)
    tp = np.zeros(len(pred_boxes))
    
    for i, pred_box in enumerate(pred_boxes):
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
            tp[i] = 1
            matched_gt[best_gt_idx] = True
    
    return tp, pred_scores.copy(), num_gt


def compute_map(
    predictions: List[Dict],
    ground_truths: List[Dict],
    num_classes: int,
    iou_thresholds: List[float] = [0.1, 0.3, 0.5]
) -> Dict[str, float]:
    """
    Compute mean Average Precision across classes and IoU thresholds.
    
    Algorithm:
    1. For each class and IoU threshold:
       a. For each sample, compute TP/FP by matching predictions to GT
          (greedy matching: highest score prediction matched first)
       b. Aggregate TP/FP across all samples
       c. Sort globally by score, compute cumulative TP/FP
       d. Compute Precision = TP / (TP + FP), Recall = TP / total_GT
       e. AP = area under PR curve
    2. mAP = mean of AP across classes
    
    Args:
        predictions: List of dicts with 'boxes', 'scores', 'labels'
        ground_truths: List of dicts with 'boxes', 'labels'
        num_classes: Number of classes
        iou_thresholds: List of IoU thresholds to evaluate
    
    Returns:
        Dict with mAP scores at each threshold and overall mAP
    """
    results = {}
    
    for iou_thresh in iou_thresholds:
        aps = []
        
        for class_id in range(num_classes):
            all_tp = []
            all_scores = []
            total_gt = 0
            
            for pred, gt in zip(predictions, ground_truths):
                # Filter by class
                pred_mask = pred['labels'] == class_id
                gt_mask = gt['labels'] == class_id
                
                pred_boxes_class = pred['boxes'][pred_mask]
                pred_scores_class = pred['scores'][pred_mask]
                gt_boxes_class = gt['boxes'][gt_mask]
                
                # Compute TP/FP for this sample
                tp, scores, num_gt = compute_tp_fp_per_sample(
                    pred_boxes_class,
                    pred_scores_class,
                    gt_boxes_class,
                    iou_threshold=iou_thresh
                )
                
                if len(tp) > 0:
                    all_tp.append(tp)
                    all_scores.append(scores)
                total_gt += num_gt
            
            # No GT for this class - skip
            if total_gt == 0:
                continue
            
            # No predictions for this class - AP is 0
            if len(all_tp) == 0:
                aps.append(0.0)
                continue
            
            # Aggregate across samples
            all_tp = np.concatenate(all_tp)
            all_scores = np.concatenate(all_scores)
            
            # Sort globally by score (descending)
            sorted_idx = np.argsort(-all_scores)
            all_tp = all_tp[sorted_idx]
            
            # Compute PR curve
            tp_cumsum = np.cumsum(all_tp)
            fp_cumsum = np.cumsum(1 - all_tp)
            
            recalls = tp_cumsum / total_gt
            precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
            
            # AP = area under PR curve (using trapezoidal rule)
            # Prepend (0, 1) point for proper area calculation
            recalls = np.concatenate([[0], recalls])
            precisions = np.concatenate([[1], precisions])
            
            # Ensure precision is monotonically decreasing (for interpolation)
            for i in range(len(precisions) - 2, -1, -1):
                precisions[i] = max(precisions[i], precisions[i + 1])
            
            # Compute area under curve
            ap = np.sum((recalls[1:] - recalls[:-1]) * precisions[1:])
            aps.append(ap)
        
        # Mean AP across classes
        results[f'mAP@{iou_thresh}'] = np.mean(aps) if len(aps) > 0 else 0.0
    
    # Overall mAP (average across thresholds)
    results['mAP'] = np.mean([results[f'mAP@{t}'] for t in iou_thresholds])
    
    return results
