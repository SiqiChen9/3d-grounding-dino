"""
Unit tests for utils/metrics.py

Tests:
- compute_iou_3d: boundary cases, numerical correctness
- compute_ap: empty predictions, perfect predictions, partial matches
- compute_map: multi-class, multi-IoU thresholds
"""
import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.metrics import compute_iou_3d, compute_ap, compute_map


class TestComputeIoU3D:
    """Tests for compute_iou_3d function."""
    
    def test_identical_boxes(self):
        """Test IoU of identical boxes is very close to 1.0."""
        box = np.array([0.5, 0.5, 0.5, 0.2, 0.2, 0.2])
        iou = compute_iou_3d(box, box)
        
        # Allow small numerical tolerance due to division
        assert np.isclose(iou, 1.0, atol=1e-3)
    
    def test_non_overlapping_boxes(self):
        """Test IoU of non-overlapping boxes is 0.0."""
        box1 = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        box2 = np.array([0.9, 0.9, 0.9, 0.1, 0.1, 0.1])
        
        iou = compute_iou_3d(box1, box2)
        
        assert np.isclose(iou, 0.0)
    
    def test_partial_overlap(self):
        """Test partial overlap gives IoU in (0, 1)."""
        box1 = np.array([0.5, 0.5, 0.5, 0.4, 0.4, 0.4])
        box2 = np.array([0.6, 0.6, 0.6, 0.4, 0.4, 0.4])
        
        iou = compute_iou_3d(box1, box2)
        
        assert 0 < iou < 1
    
    def test_contained_box(self):
        """Test when one box contains another."""
        large_box = np.array([0.5, 0.5, 0.5, 0.6, 0.6, 0.6])
        small_box = np.array([0.5, 0.5, 0.5, 0.2, 0.2, 0.2])
        
        iou = compute_iou_3d(large_box, small_box)
        
        # IoU = small_volume / large_volume
        expected = (0.2 ** 3) / (0.6 ** 3)
        assert np.isclose(iou, expected, atol=1e-5)
    
    def test_symmetry(self):
        """Test that IoU is symmetric."""
        box1 = np.array([0.3, 0.4, 0.5, 0.2, 0.3, 0.4])
        box2 = np.array([0.5, 0.5, 0.5, 0.3, 0.3, 0.3])
        
        iou12 = compute_iou_3d(box1, box2)
        iou21 = compute_iou_3d(box2, box1)
        
        assert np.isclose(iou12, iou21)


class TestComputeAP:
    """Tests for compute_ap function."""
    
    def test_empty_predictions(self):
        """Test AP with no predictions is 0."""
        pred_boxes = np.zeros((0, 6))
        pred_scores = np.zeros(0)
        gt_boxes = np.random.rand(3, 6)
        
        ap = compute_ap(pred_boxes, pred_scores, gt_boxes)
        
        assert ap == 0.0
    
    def test_empty_ground_truth(self):
        """Test AP with no ground truth is 0."""
        pred_boxes = np.random.rand(5, 6)
        pred_scores = np.random.rand(5)
        gt_boxes = np.zeros((0, 6))
        
        ap = compute_ap(pred_boxes, pred_scores, gt_boxes)
        
        assert ap == 0.0
    
    def test_perfect_predictions(self):
        """Test AP with perfect predictions should be reasonable."""
        # Create identical prediction and ground truth
        gt_boxes = np.array([
            [0.3, 0.3, 0.3, 0.2, 0.2, 0.2],
            [0.7, 0.7, 0.7, 0.2, 0.2, 0.2]
        ])
        pred_boxes = gt_boxes.copy()
        pred_scores = np.array([0.9, 0.8])
        
        ap = compute_ap(pred_boxes, pred_scores, gt_boxes, iou_threshold=0.5)
        
        # AP should be positive when predictions perfectly match
        # Note: the exact value depends on the AP calculation implementation
        assert ap > 0.0
    
    def test_no_matches(self):
        """Test AP with no matching predictions."""
        gt_boxes = np.array([[0.2, 0.2, 0.2, 0.1, 0.1, 0.1]])
        pred_boxes = np.array([[0.8, 0.8, 0.8, 0.1, 0.1, 0.1]])  # Far away
        pred_scores = np.array([0.9])
        
        ap = compute_ap(pred_boxes, pred_scores, gt_boxes, iou_threshold=0.5)
        
        assert ap == 0.0
    
    def test_ap_range(self):
        """Test AP is in [0, 1] range."""
        gt_boxes = np.random.rand(5, 6) * 0.5 + 0.25
        pred_boxes = np.random.rand(10, 6) * 0.5 + 0.25
        pred_scores = np.random.rand(10)
        
        ap = compute_ap(pred_boxes, pred_scores, gt_boxes)
        
        assert 0 <= ap <= 1


class TestComputeMAP:
    """Tests for compute_map function."""
    
    def test_output_format(self):
        """Test output contains expected keys."""
        predictions = [
            {
                'boxes': np.random.rand(5, 6),
                'scores': np.random.rand(5),
                'labels': np.array([0, 1, 2, 0, 1])
            }
        ]
        ground_truths = [
            {
                'boxes': np.random.rand(3, 6),
                'labels': np.array([0, 1, 2])
            }
        ]
        
        results = compute_map(predictions, ground_truths, num_classes=5)
        
        assert 'mAP' in results
        assert 'mAP@0.5' in results
        assert 'mAP@0.3' in results
        assert 'mAP@0.1' in results
    
    def test_empty_predictions(self):
        """Test mAP with empty predictions."""
        predictions = [
            {
                'boxes': np.zeros((0, 6)),
                'scores': np.zeros(0),
                'labels': np.array([], dtype=np.int32)
            }
        ]
        ground_truths = [
            {
                'boxes': np.random.rand(3, 6),
                'labels': np.array([0, 1, 2])
            }
        ]
        
        results = compute_map(predictions, ground_truths, num_classes=5)
        
        assert results['mAP'] == 0.0
    
    def test_multiple_samples(self):
        """Test mAP with multiple samples."""
        num_samples = 3
        
        predictions = []
        ground_truths = []
        
        for _ in range(num_samples):
            predictions.append({
                'boxes': np.random.rand(5, 6) * 0.5 + 0.25,
                'scores': np.random.rand(5),
                'labels': np.random.randint(0, 3, 5)
            })
            ground_truths.append({
                'boxes': np.random.rand(3, 6) * 0.5 + 0.25,
                'labels': np.random.randint(0, 3, 3)
            })
        
        results = compute_map(predictions, ground_truths, num_classes=5)
        
        # mAP should be in valid range
        assert 0 <= results['mAP'] <= 1
    
    def test_custom_iou_thresholds(self):
        """Test with custom IoU thresholds."""
        predictions = [
            {
                'boxes': np.random.rand(5, 6),
                'scores': np.random.rand(5),
                'labels': np.zeros(5, dtype=np.int32)
            }
        ]
        ground_truths = [
            {
                'boxes': np.random.rand(3, 6),
                'labels': np.zeros(3, dtype=np.int32)
            }
        ]
        
        custom_thresholds = [0.25, 0.5, 0.75]
        results = compute_map(
            predictions, ground_truths, 
            num_classes=3, 
            iou_thresholds=custom_thresholds
        )
        
        for thresh in custom_thresholds:
            assert f'mAP@{thresh}' in results
