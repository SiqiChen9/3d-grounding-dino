"""
Unit tests for models/losses.py

Tests:
- box_iou_3d: IoU computation for various cases
- generalized_box_iou_3d: GIoU computation and range
- HungarianMatcher: matching format, empty targets
- SetCriterion: loss backpropagation, reasonable values
"""
import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.losses import (
    box_iou_3d,
    generalized_box_iou_3d,
    HungarianMatcher,
    SetCriterion
)


class TestBoxIoU3D:
    """Tests for box_iou_3d function."""
    
    def test_identical_boxes_iou_one(self, device):
        """Test that identical boxes have IoU = 1."""
        box = torch.tensor([[0.5, 0.5, 0.5, 0.2, 0.2, 0.2]], device=device)
        iou = box_iou_3d(box, box)
        
        assert torch.allclose(iou, torch.ones(1, 1, device=device), atol=1e-5)
    
    def test_non_overlapping_boxes_iou_zero(self, device):
        """Test that non-overlapping boxes have IoU = 0."""
        box1 = torch.tensor([[0.1, 0.1, 0.1, 0.1, 0.1, 0.1]], device=device)
        box2 = torch.tensor([[0.9, 0.9, 0.9, 0.1, 0.1, 0.1]], device=device)
        
        iou = box_iou_3d(box1, box2)
        
        assert torch.allclose(iou, torch.zeros(1, 1, device=device), atol=1e-5)
    
    def test_partial_overlap(self, device):
        """Test partial overlap gives IoU in (0, 1)."""
        box1 = torch.tensor([[0.5, 0.5, 0.5, 0.4, 0.4, 0.4]], device=device)
        box2 = torch.tensor([[0.6, 0.6, 0.6, 0.4, 0.4, 0.4]], device=device)
        
        iou = box_iou_3d(box1, box2)
        
        assert 0 < iou.item() < 1
    
    def test_output_shape(self, device):
        """Test output shape is (N, M) for N and M boxes."""
        boxes1 = torch.rand(5, 6, device=device)
        boxes2 = torch.rand(3, 6, device=device)
        
        iou = box_iou_3d(boxes1, boxes2)
        
        assert iou.shape == (5, 3)
    
    def test_symmetry(self, device):
        """Test IoU is symmetric."""
        box1 = torch.rand(3, 6, device=device)
        box2 = torch.rand(4, 6, device=device)
        
        iou12 = box_iou_3d(box1, box2)
        iou21 = box_iou_3d(box2, box1)
        
        assert torch.allclose(iou12, iou21.T, atol=1e-5)
    
    def test_iou_range(self, device):
        """Test IoU values are in [0, 1]."""
        boxes1 = torch.rand(10, 6, device=device) * 0.5 + 0.25
        boxes2 = torch.rand(10, 6, device=device) * 0.5 + 0.25
        
        iou = box_iou_3d(boxes1, boxes2)
        
        assert (iou >= 0).all()
        assert (iou <= 1).all()


class TestGeneralizedBoxIoU3D:
    """Tests for generalized_box_iou_3d function."""
    
    def test_identical_boxes_giou_one(self, device):
        """Test that identical boxes have GIoU = 1."""
        box = torch.tensor([[0.5, 0.5, 0.5, 0.2, 0.2, 0.2]], device=device)
        giou = generalized_box_iou_3d(box, box)
        
        assert torch.allclose(giou, torch.ones(1, 1, device=device), atol=1e-5)
    
    def test_giou_range(self, device):
        """Test GIoU values are in [-1, 1]."""
        boxes1 = torch.rand(10, 6, device=device) * 0.5 + 0.25
        boxes2 = torch.rand(10, 6, device=device) * 0.5 + 0.25
        
        giou = generalized_box_iou_3d(boxes1, boxes2)
        
        assert (giou >= -1).all()
        assert (giou <= 1).all()
    
    def test_giou_less_than_iou(self, device):
        """Test GIoU <= IoU (GIoU is always less or equal to IoU)."""
        boxes1 = torch.rand(5, 6, device=device) * 0.3 + 0.35
        boxes2 = torch.rand(5, 6, device=device) * 0.3 + 0.35
        
        iou = box_iou_3d(boxes1, boxes2)
        giou = generalized_box_iou_3d(boxes1, boxes2)
        
        # GIoU should be <= IoU with small tolerance for numerical errors
        assert (giou <= iou + 1e-5).all()
    
    def test_output_shape(self, device):
        """Test output shape is (N, M)."""
        boxes1 = torch.rand(4, 6, device=device)
        boxes2 = torch.rand(6, 6, device=device)
        
        giou = generalized_box_iou_3d(boxes1, boxes2)
        
        assert giou.shape == (4, 6)


class TestHungarianMatcher:
    """Tests for HungarianMatcher module."""
    
    def test_matching_format(self, device, batch_size, num_queries, num_classes):
        """Test that matcher returns correct format."""
        matcher = HungarianMatcher()
        
        pred_logits = torch.randn(batch_size, num_queries, num_classes + 1, device=device)
        pred_boxes = torch.sigmoid(torch.randn(batch_size, num_queries, 6, device=device))
        
        target_labels = [torch.randint(0, num_classes, (3,), device=device) for _ in range(batch_size)]
        target_boxes = [torch.rand(3, 6, device=device) for _ in range(batch_size)]
        
        indices = matcher(pred_logits, pred_boxes, target_labels, target_boxes)
        
        # Returns list of tuples
        assert isinstance(indices, list)
        assert len(indices) == batch_size
        
        for src_idx, tgt_idx in indices:
            # Indices should be tensors
            assert isinstance(src_idx, torch.Tensor)
            assert isinstance(tgt_idx, torch.Tensor)
            # Same length
            assert len(src_idx) == len(tgt_idx)
    
    def test_empty_targets(self, device, batch_size, num_queries, num_classes):
        """Test handling of empty targets."""
        matcher = HungarianMatcher()
        
        pred_logits = torch.randn(batch_size, num_queries, num_classes + 1, device=device)
        pred_boxes = torch.sigmoid(torch.randn(batch_size, num_queries, 6, device=device))
        
        # Empty targets for each sample
        target_labels = [torch.tensor([], dtype=torch.long, device=device) for _ in range(batch_size)]
        target_boxes = [torch.zeros(0, 6, device=device) for _ in range(batch_size)]
        
        indices = matcher(pred_logits, pred_boxes, target_labels, target_boxes)
        
        # Should return empty indices for all samples
        for src_idx, tgt_idx in indices:
            assert len(src_idx) == 0
            assert len(tgt_idx) == 0
    
    def test_single_target(self, device, num_queries, num_classes):
        """Test with single target per sample."""
        matcher = HungarianMatcher()
        
        pred_logits = torch.randn(1, num_queries, num_classes + 1, device=device)
        pred_boxes = torch.sigmoid(torch.randn(1, num_queries, 6, device=device))
        
        target_labels = [torch.tensor([0], device=device)]
        target_boxes = [torch.rand(1, 6, device=device)]
        
        indices = matcher(pred_logits, pred_boxes, target_labels, target_boxes)
        
        src_idx, tgt_idx = indices[0]
        assert len(src_idx) == 1
        assert len(tgt_idx) == 1
        assert tgt_idx[0] == 0  # Should match to target 0


class TestSetCriterion:
    """Tests for SetCriterion module."""
    
    @pytest.fixture
    def criterion(self, num_classes, loss_weight_dict, device):
        """Create SetCriterion instance."""
        matcher = HungarianMatcher()
        return SetCriterion(
            num_classes=num_classes,
            matcher=matcher,
            weight_dict=loss_weight_dict,
            eos_coef=0.1
        ).to(device)
    
    def test_loss_computation(self, criterion, device, batch_size, num_queries, num_classes):
        """Test that losses are computed and have correct keys."""
        pred_logits = torch.randn(batch_size, num_queries, num_classes + 1, device=device)
        pred_boxes = torch.sigmoid(torch.randn(batch_size, num_queries, 6, device=device))
        
        target_labels = [torch.randint(0, num_classes, (3,), device=device) for _ in range(batch_size)]
        target_boxes = [torch.rand(3, 6, device=device) for _ in range(batch_size)]
        
        losses = criterion(pred_logits, pred_boxes, target_labels, target_boxes)
        
        # Check required keys (note: implementation uses loss_l1 not loss_bbox)
        assert 'loss_ce' in losses
        assert 'loss_l1' in losses
        assert 'loss_giou' in losses
    
    def test_loss_backpropagation(self, criterion, device, batch_size, num_queries, num_classes):
        """Test that losses can be backpropagated."""
        # Create leaf tensors first, then apply sigmoid
        pred_logits = torch.randn(batch_size, num_queries, num_classes + 1, device=device, requires_grad=True)
        pred_boxes_raw = torch.randn(batch_size, num_queries, 6, device=device, requires_grad=True)
        pred_boxes = torch.sigmoid(pred_boxes_raw)
        
        target_labels = [torch.randint(0, num_classes, (3,), device=device) for _ in range(batch_size)]
        target_boxes = [torch.rand(3, 6, device=device) for _ in range(batch_size)]
        
        losses = criterion(pred_logits, pred_boxes, target_labels, target_boxes)
        
        total_loss = sum(losses.values())
        total_loss.backward()
        
        assert pred_logits.grad is not None
        assert pred_boxes_raw.grad is not None  # Check grad on leaf tensor
    
    def test_loss_values_reasonable(self, criterion, device, batch_size, num_queries, num_classes):
        """Test that loss values are in reasonable range."""
        pred_logits = torch.randn(batch_size, num_queries, num_classes + 1, device=device)
        pred_boxes = torch.sigmoid(torch.randn(batch_size, num_queries, 6, device=device))
        
        target_labels = [torch.randint(0, num_classes, (3,), device=device) for _ in range(batch_size)]
        target_boxes = [torch.rand(3, 6, device=device) for _ in range(batch_size)]
        
        losses = criterion(pred_logits, pred_boxes, target_labels, target_boxes)
        
        # Losses should be finite and non-negative
        for key, value in losses.items():
            assert torch.isfinite(value), f"{key} is not finite"
            assert value >= 0, f"{key} is negative"
    
    def test_empty_targets(self, criterion, device, batch_size, num_queries, num_classes):
        """Test with empty targets."""
        pred_logits = torch.randn(batch_size, num_queries, num_classes + 1, device=device)
        pred_boxes = torch.sigmoid(torch.randn(batch_size, num_queries, 6, device=device))
        
        target_labels = [torch.tensor([], dtype=torch.long, device=device) for _ in range(batch_size)]
        target_boxes = [torch.zeros(0, 6, device=device) for _ in range(batch_size)]
        
        losses = criterion(pred_logits, pred_boxes, target_labels, target_boxes)
        
        # Should still return losses (even if some are zero)
        assert 'loss_ce' in losses
