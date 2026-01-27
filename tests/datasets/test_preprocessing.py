"""
Unit tests for datasets/preprocessing.py

Tests:
- normalize_intensity: value range [0, 1], windowing
- resize_volume: output size
- mask_to_boxes_3d: bounding box format, normalization
- apply_augmentation_3d: rotation, scale, crop, elastic deformation, intensity jitter
"""
import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from datasets.preprocessing import (
    normalize_intensity,
    resize_volume,
    mask_to_boxes_3d,
    apply_augmentation_3d
)


class TestNormalizeIntensity:
    """Tests for normalize_intensity function."""
    
    def test_output_range_with_clip(self):
        """Test output is in [0, 1] with clipping."""
        volume = np.random.randn(32, 64, 64) * 500 + 50  # HU-like values
        
        normalized = normalize_intensity(volume, clip=True)
        
        assert normalized.min() >= 0
        assert normalized.max() <= 1
    
    def test_output_range_without_clip(self):
        """Test output can exceed [0, 1] without clipping."""
        # Values outside windowing range
        volume = np.array([[[1000, -500]]])  # Far outside [-125, 225] default window
        
        normalized = normalize_intensity(volume, clip=False)
        
        # Some values should be outside [0, 1]
        assert normalized.min() < 0 or normalized.max() > 1
    
    def test_windowing_center(self):
        """Test that window center maps to 0.5."""
        window_center = 50
        window_width = 350
        
        volume = np.array([[[window_center]]])
        normalized = normalize_intensity(volume, window_center, window_width)
        
        assert np.isclose(normalized[0, 0, 0], 0.5)
    
    def test_windowing_edges(self):
        """Test that window edges map to 0 and 1."""
        window_center = 50
        window_width = 350
        
        min_value = window_center - window_width / 2  # -125
        max_value = window_center + window_width / 2  # 225
        
        volume = np.array([[[min_value, max_value]]])
        normalized = normalize_intensity(volume, window_center, window_width, clip=True)
        
        assert np.isclose(normalized[0, 0, 0], 0)
        assert np.isclose(normalized[0, 0, 1], 1)
    
    def test_output_dtype(self):
        """Test output dtype."""
        volume = np.random.randn(16, 32, 32).astype(np.float32)
        normalized = normalize_intensity(volume)
        
        # Should be a floating point type
        assert normalized.dtype in [np.float32, np.float64]


class TestResizeVolume:
    """Tests for resize_volume function."""
    
    def test_output_shape(self):
        """Test output has target shape."""
        volume = np.random.randn(32, 64, 64)
        target_size = (16, 32, 32)
        
        resized = resize_volume(volume, target_size)
        
        assert resized.shape == target_size
    
    def test_different_target_sizes(self):
        """Test with various target sizes."""
        volume = np.random.randn(32, 64, 64)
        
        for target_size in [(16, 32, 32), (64, 128, 128), (32, 48, 48)]:
            resized = resize_volume(volume, target_size)
            assert resized.shape == target_size
    
    def test_output_dtype(self):
        """Test output is float32."""
        volume = np.random.randn(16, 32, 32)
        resized = resize_volume(volume, (8, 16, 16))
        
        assert resized.dtype == np.float32


class TestMaskToBoxes3D:
    """Tests for mask_to_boxes_3d function."""
    
    def test_empty_mask(self):
        """Test with empty mask (all zeros)."""
        mask = np.zeros((32, 64, 64), dtype=np.int32)
        
        boxes = mask_to_boxes_3d(mask)
        
        assert len(boxes) == 0
    
    def test_single_region(self, sample_mask_np):
        """Test with mask containing regions."""
        boxes = mask_to_boxes_3d(sample_mask_np, min_volume=10)
        
        # Should find some boxes (if regions are large enough)
        for box_dict in boxes:
            # Check format
            assert 'box' in box_dict
            assert 'label' in box_dict
            assert len(box_dict['box']) == 6
            assert box_dict['label'] > 0  # Not background
    
    def test_box_normalization(self, sample_mask_np):
        """Test that boxes are normalized to [0, 1]."""
        boxes = mask_to_boxes_3d(sample_mask_np, min_volume=10)
        
        for box_dict in boxes:
            box = box_dict['box']
            # All values should be in [0, 1]
            assert (box >= 0).all()
            assert (box <= 1).all()
    
    def test_min_volume_filter(self):
        """Test minimum volume filter."""
        # Create mask with small region
        mask = np.zeros((32, 64, 64), dtype=np.int32)
        mask[5:7, 10:12, 10:12] = 1  # 2x2x2 = 8 voxels
        
        # Should be filtered out
        boxes = mask_to_boxes_3d(mask, min_volume=100)
        assert len(boxes) == 0
        
        # Should pass with lower threshold
        boxes = mask_to_boxes_3d(mask, min_volume=5)
        assert len(boxes) == 1
    
    def test_multiple_labels(self):
        """Test with multiple class labels."""
        mask = np.zeros((32, 64, 64), dtype=np.int32)
        mask[5:15, 10:30, 10:30] = 1
        mask[20:30, 35:55, 35:55] = 2
        
        boxes = mask_to_boxes_3d(mask, min_volume=10)
        
        # Should find boxes for both labels
        labels = [box['label'] for box in boxes]
        assert 1 in labels
        assert 2 in labels


class TestApplyAugmentation3D:
    """Tests for apply_augmentation_3d function."""
    
    def test_output_shapes(self):
        """Test that output shapes match input."""
        volume = np.random.randn(32, 64, 64).astype(np.float32)
        boxes = [
            {'box': np.array([0.5, 0.5, 0.5, 0.2, 0.2, 0.2]), 'label': 1}
        ]
        
        aug_volume, aug_boxes = apply_augmentation_3d(volume, boxes)
        
        assert aug_volume.shape == volume.shape
        assert len(aug_boxes) == len(boxes)
    
    def test_box_values_clipped(self):
        """Test that all box values remain in [0, 1]."""
        volume = np.random.randn(32, 64, 64).astype(np.float32)
        boxes = [
            {'box': np.array([0.9, 0.9, 0.9, 0.2, 0.2, 0.2]), 'label': 1},
            {'box': np.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.2]), 'label': 2}
        ]
        
        # Apply multiple times to trigger augmentations
        for _ in range(10):
            aug_volume, aug_boxes = apply_augmentation_3d(
                volume, boxes, 
                rotate_prob=0.5, 
                rotate_range=30.0,
                scale_prob=0.5
            )
            
            for box_dict in aug_boxes:
                box = box_dict['box']
                assert (box >= 0).all(), "Box value below 0"
                assert (box <= 1).all(), "Box value above 1"
    
    def test_intensity_jitter(self):
        """Test intensity jittering."""
        volume = np.ones((16, 32, 32), dtype=np.float32) * 0.5
        boxes = [{'box': np.array([0.5, 0.5, 0.5, 0.2, 0.2, 0.2]), 'label': 1}]
        
        aug_volume, _ = apply_augmentation_3d(
            volume, boxes,
            rotate_prob=0.0,
            scale_prob=0.0,
            elastic_prob=0.0,
            intensity_jitter=0.1
        )
        
        # Values should be perturbed
        assert not np.allclose(aug_volume, volume)
    
    def test_no_augmentation(self):
        """Test with all augmentations disabled."""
        volume = np.random.randn(16, 32, 32).astype(np.float32)
        boxes = [{'box': np.array([0.5, 0.5, 0.5, 0.2, 0.2, 0.2]), 'label': 1}]
        
        aug_volume, aug_boxes = apply_augmentation_3d(
            volume, boxes,
            rotate_prob=0.0,
            scale_prob=0.0,
            elastic_prob=0.0,
            intensity_jitter=0.0
        )
        
        # Should be identical (copy)
        assert np.array_equal(aug_volume, volume)
        assert np.array_equal(aug_boxes[0]['box'], boxes[0]['box'])
