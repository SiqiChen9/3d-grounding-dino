"""
Unit tests for utils/visualization.py

Tests:
- resize_volume: output size, scaling factors
- denormalize_box_3d: normalized to absolute coordinates
- box_3d_to_2d_slice: different axis projections
"""
import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.visualization import (
    resize_volume,
    denormalize_box_3d,
    box_3d_to_2d_slice
)


class TestResizeVolume:
    """Tests for resize_volume function."""
    
    def test_output_shape(self, sample_volume_np):
        """Test output has correct H, W dimensions."""
        target_size = 512
        resized, (scale_d, scale_h, scale_w) = resize_volume(sample_volume_np, target_size)
        
        assert resized.shape[1] == target_size  # H
        assert resized.shape[2] == target_size  # W
    
    def test_proportional_depth_scaling(self, sample_volume_np):
        """Test that D is scaled proportionally."""
        target_size = 512
        original_d, original_h, original_w = sample_volume_np.shape
        
        resized, (scale_d, scale_h, scale_w) = resize_volume(sample_volume_np, target_size)
        
        # D should be scaled by same factor as H or W
        expected_d = int(original_d * scale_d)
        # Allow small tolerance due to rounding
        assert abs(resized.shape[0] - expected_d) <= 1
    
    def test_scaling_factors(self, sample_volume_np):
        """Test that scaling factors are correct."""
        target_size = 256
        original_d, original_h, original_w = sample_volume_np.shape
        
        resized, (scale_d, scale_h, scale_w) = resize_volume(sample_volume_np, target_size)
        
        # scale_h and scale_w should give target_size
        assert np.isclose(original_h * scale_h, target_size, atol=1)
        assert np.isclose(original_w * scale_w, target_size, atol=1)
    
    def test_different_target_sizes(self, sample_volume_np):
        """Test with different target sizes."""
        for target_size in [128, 256, 512]:
            resized, _ = resize_volume(sample_volume_np, target_size)
            
            assert resized.shape[1] == target_size
            assert resized.shape[2] == target_size
    
    def test_dtype_preserved(self, sample_volume_np):
        """Test that output is float32."""
        resized, _ = resize_volume(sample_volume_np.astype(np.float32), 256)
        
        # Result should be float type
        assert resized.dtype in [np.float32, np.float64]


class TestDenormalizeBox3D:
    """Tests for denormalize_box_3d function."""
    
    def test_center_normalized(self):
        """Test denormalization of centered box."""
        box = np.array([0.5, 0.5, 0.5, 0.2, 0.2, 0.2])
        volume_shape = (100, 200, 200)
        
        abs_box = denormalize_box_3d(box, volume_shape)
        
        # Center should be at middle of volume
        assert np.isclose(abs_box[0], 100)  # cx = 0.5 * 200
        assert np.isclose(abs_box[1], 100)  # cy = 0.5 * 200
        assert np.isclose(abs_box[2], 50)   # cz = 0.5 * 100
    
    def test_size_denormalization(self):
        """Test that size is correctly denormalized."""
        box = np.array([0.5, 0.5, 0.5, 0.1, 0.2, 0.3])
        volume_shape = (100, 200, 200)  # D, H, W
        
        abs_box = denormalize_box_3d(box, volume_shape)
        
        # Size should be scaled by volume dimensions
        assert np.isclose(abs_box[3], 0.1 * 200)  # w
        assert np.isclose(abs_box[4], 0.2 * 200)  # h
        assert np.isclose(abs_box[5], 0.3 * 100)  # d
    
    def test_corner_boxes(self):
        """Test boxes at corners."""
        volume_shape = (64, 128, 128)
        
        # Box at origin corner
        box = np.array([0.0, 0.0, 0.0, 0.1, 0.1, 0.1])
        abs_box = denormalize_box_3d(box, volume_shape)
        
        assert abs_box[0] == 0  # cx
        assert abs_box[1] == 0  # cy
        assert abs_box[2] == 0  # cz
    
    def test_output_shape(self):
        """Test output shape is (6,)."""
        box = np.array([0.3, 0.4, 0.5, 0.1, 0.1, 0.1])
        volume_shape = (64, 128, 128)
        
        abs_box = denormalize_box_3d(box, volume_shape)
        
        assert abs_box.shape == (6,)


class TestBox3DTo2DSlice:
    """Tests for box_3d_to_2d_slice function."""
    
    def test_axial_projection(self):
        """Test axial (z-slice) projection."""
        # Box: cx, cy, cz, w, h, d
        box_3d = np.array([100, 100, 50, 40, 40, 20])
        
        # Slice at cz (should intersect)
        result = box_3d_to_2d_slice(box_3d, slice_idx=50, axis='axial')
        
        assert result is not None
        x_min, y_min, width, height = result
        
        # x_min should be cx - w/2
        assert np.isclose(x_min, 80)  # 100 - 40/2
        # y_min should be cy - h/2
        assert np.isclose(y_min, 80)  # 100 - 40/2
        assert np.isclose(width, 40)
        assert np.isclose(height, 40)
    
    def test_no_intersection(self):
        """Test when slice doesn't intersect box."""
        box_3d = np.array([100, 100, 50, 40, 40, 20])
        
        # Slice far from box center
        result = box_3d_to_2d_slice(box_3d, slice_idx=100, axis='axial')
        
        assert result is None
    
    def test_sagittal_projection(self):
        """Test sagittal (x-slice) projection."""
        box_3d = np.array([100, 100, 50, 40, 40, 20])
        
        result = box_3d_to_2d_slice(box_3d, slice_idx=100, axis='sagittal')
        
        assert result is not None
    
    def test_coronal_projection(self):
        """Test coronal (y-slice) projection."""
        box_3d = np.array([100, 100, 50, 40, 40, 20])
        
        result = box_3d_to_2d_slice(box_3d, slice_idx=100, axis='coronal')
        
        assert result is not None
    
    def test_edge_intersection(self):
        """Test intersection at box edge."""
        box_3d = np.array([100, 100, 50, 40, 40, 20])
        
        # Slice at edge of box (cz - d/2 = 40)
        result = box_3d_to_2d_slice(box_3d, slice_idx=40, axis='axial')
        
        # Should still intersect (at the edge)
        assert result is not None
