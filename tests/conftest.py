"""
Shared pytest fixtures for 3D Grounding-DETR tests.
"""
import pytest
import torch
import numpy as np
from typing import Tuple, List, Dict


# ============================================================================
# Device and Basic Configuration Fixtures
# ============================================================================

@pytest.fixture
def device():
    """Test device (CPU for compatibility)."""
    return torch.device("cpu")


@pytest.fixture
def batch_size():
    """Default batch size for tests."""
    return 2


@pytest.fixture
def hidden_dim():
    """Default hidden dimension."""
    return 256


@pytest.fixture
def num_classes():
    """Default number of classes (excluding background)."""
    return 5


@pytest.fixture
def num_queries():
    """Default number of object queries."""
    return 100


# ============================================================================
# Volume and Spatial Dimension Fixtures
# ============================================================================

@pytest.fixture
def volume_size():
    """Default volume size (D, H, W)."""
    return (32, 64, 64)


@pytest.fixture
def small_volume_size():
    """Smaller volume size for faster tests."""
    return (16, 32, 32)


@pytest.fixture
def spatial_size():
    """Spatial size after backbone processing."""
    return (4, 8, 8)  # D', H', W'


# ============================================================================
# Sample Data Fixtures
# ============================================================================

@pytest.fixture
def sample_volume(batch_size, small_volume_size, device):
    """Generate sample CT volume tensor."""
    D, H, W = small_volume_size
    return torch.randn(batch_size, 1, D, H, W, device=device)


@pytest.fixture
def sample_boxes(batch_size, device):
    """
    Generate sample 3D bounding boxes.
    
    Returns:
        List of tensors, each (N_i, 6) with format (cx, cy, cz, w, h, d) normalized to [0, 1].
    """
    boxes = []
    for i in range(batch_size):
        num_boxes = np.random.randint(1, 5)  # 1-4 boxes per sample
        # Generate random boxes with reasonable sizes
        cx = torch.rand(num_boxes, device=device) * 0.6 + 0.2  # [0.2, 0.8]
        cy = torch.rand(num_boxes, device=device) * 0.6 + 0.2
        cz = torch.rand(num_boxes, device=device) * 0.6 + 0.2
        w = torch.rand(num_boxes, device=device) * 0.3 + 0.1   # [0.1, 0.4]
        h = torch.rand(num_boxes, device=device) * 0.3 + 0.1
        d = torch.rand(num_boxes, device=device) * 0.3 + 0.1
        
        box = torch.stack([cx, cy, cz, w, h, d], dim=1)
        boxes.append(box)
    return boxes


@pytest.fixture
def sample_labels(sample_boxes, num_classes, device):
    """
    Generate sample labels corresponding to sample_boxes.
    
    Returns:
        List of tensors, each (N_i,) with class labels in [0, num_classes-1].
    """
    labels = []
    for boxes in sample_boxes:
        num_boxes = boxes.shape[0]
        label = torch.randint(0, num_classes, (num_boxes,), device=device)
        labels.append(label)
    return labels


@pytest.fixture
def sample_text_features(batch_size, num_classes, hidden_dim, device):
    """Generate sample text features (B, num_classes, hidden_dim)."""
    return torch.randn(batch_size, num_classes, hidden_dim, device=device)


@pytest.fixture
def sample_image_features(batch_size, hidden_dim, spatial_size, device):
    """Generate sample image features (N, B, hidden_dim) where N = D' * H' * W'."""
    D, H, W = spatial_size
    N = D * H * W
    return torch.randn(N, batch_size, hidden_dim, device=device)


@pytest.fixture
def sample_queries(batch_size, num_queries, hidden_dim, device):
    """Generate sample object queries (num_queries, B, hidden_dim)."""
    return torch.randn(num_queries, batch_size, hidden_dim, device=device)


# ============================================================================
# Model Output Fixtures
# ============================================================================

@pytest.fixture
def sample_pred_logits(batch_size, num_queries, num_classes, device):
    """Generate sample prediction logits (B, num_queries, num_classes+1)."""
    return torch.randn(batch_size, num_queries, num_classes + 1, device=device)


@pytest.fixture
def sample_pred_boxes(batch_size, num_queries, device):
    """Generate sample predicted boxes (B, num_queries, 6) with sigmoid-like values."""
    boxes = torch.sigmoid(torch.randn(batch_size, num_queries, 6, device=device))
    # Ensure boxes are valid (center + size/2 <= 1 and center - size/2 >= 0)
    boxes[..., 3:] = boxes[..., 3:] * 0.5  # Limit size to max 0.5
    return boxes


# ============================================================================
# Numpy Array Fixtures (for non-PyTorch tests)
# ============================================================================

@pytest.fixture
def sample_volume_np(small_volume_size):
    """Generate sample numpy volume (D, H, W)."""
    D, H, W = small_volume_size
    return np.random.randn(D, H, W).astype(np.float32)


@pytest.fixture
def sample_mask_np(small_volume_size, num_classes):
    """Generate sample segmentation mask (D, H, W)."""
    D, H, W = small_volume_size
    mask = np.zeros((D, H, W), dtype=np.int32)
    
    # Add some random regions for different classes
    for class_id in range(1, min(num_classes + 1, 4)):  # Add up to 3 classes
        # Create a small random region
        z_start = np.random.randint(0, D // 2)
        y_start = np.random.randint(0, H // 2)
        x_start = np.random.randint(0, W // 2)
        z_size = np.random.randint(D // 8, D // 4)
        y_size = np.random.randint(H // 8, H // 4)
        x_size = np.random.randint(W // 8, W // 4)
        
        mask[z_start:z_start+z_size, y_start:y_start+y_size, x_start:x_start+x_size] = class_id
    
    return mask


@pytest.fixture
def sample_box_np():
    """Generate sample numpy box (6,) in (cx, cy, cz, w, h, d) format."""
    return np.array([0.5, 0.5, 0.5, 0.2, 0.2, 0.2], dtype=np.float32)


# ============================================================================
# Configuration Fixtures
# ============================================================================

@pytest.fixture
def model_config(num_classes, num_queries, hidden_dim):
    """Default model configuration dictionary."""
    return {
        'num_classes': num_classes,
        'num_queries': num_queries,
        'hidden_dim': hidden_dim,
        'backbone_embed_dim': 48,  # Smaller for faster tests
        'backbone_depths': [1, 1, 1, 1],  # Fewer layers for faster tests
        'backbone_num_heads': [2, 4, 8, 16],
        'num_encoder_layers': 1,
        'num_decoder_layers': 1,
        'num_heads': 4,
        'dim_feedforward': 512,
        'dropout': 0.1,
        'trainable_pseudo_features': True
    }


@pytest.fixture
def loss_weight_dict():
    """Default loss weight dictionary."""
    return {
        'loss_ce': 1.0,
        'loss_bbox': 5.0,
        'loss_giou': 2.0
    }
