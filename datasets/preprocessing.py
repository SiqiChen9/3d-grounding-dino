"""
Data preprocessing utilities for 3D CT volumes.
Includes intensity normalization, resampling, and bounding box extraction.
"""
import numpy as np
import cv2
from scipy import ndimage
from typing import Tuple, List, Optional


def normalize_intensity(
    volume: np.ndarray,
    window_center: float = 50,
    window_width: float = 350,
    clip: bool = True
) -> np.ndarray:
    """
    Apply CT windowing and normalize to [0, 1].
    
    Args:
        volume: Input CT volume (HU values)
        window_center: Center of windowing (default: 50 for abdominal)
        window_width: Width of windowing (default: 350 for abdominal)
        clip: Whether to clip values to [0, 1]
    
    Returns:
        Normalized volume in [0, 1] range
    """
    min_value = window_center - window_width / 2
    max_value = window_center + window_width / 2
    
    volume_normalized = (volume - min_value) / (max_value - min_value)
    
    if clip:
        volume_normalized = np.clip(volume_normalized, 0, 1)
    
    return volume_normalized


def resample_volume(
    volume: np.ndarray,
    original_spacing: Tuple[float, float, float],
    target_spacing: Tuple[float, float, float] = (2.0, 2.0, 2.0),
    order: int = 1
) -> np.ndarray:
    """
    Resample volume to target spacing.
    
    Args:
        volume: Input volume (D, H, W)
        original_spacing: Original voxel spacing (z, y, x) in mm
        target_spacing: Target voxel spacing (z, y, x) in mm
        order: Interpolation order (0=nearest, 1=linear, 3=cubic)
    
    Returns:
        Resampled volume (float32)
    """
    resize_factor = np.array(original_spacing) / np.array(target_spacing)
    new_shape = np.round(volume.shape * resize_factor).astype(int)
    
    resampled = ndimage.zoom(volume, resize_factor, order=order)
    
    return resampled.astype(np.float32)


def resize_volume(
    volume: np.ndarray,
    target_size: Tuple[int, int, int],
    order: int = 1
) -> np.ndarray:
    """
    Resize volume to target size.
    
    Args:
        volume: Input volume (D, H, W)
        target_size: Target size (D, H, W)
        order: Interpolation order
    
    Returns:
        Resized volume (float32)
    """
    zoom_factors = np.array(target_size) / np.array(volume.shape)
    resized = ndimage.zoom(volume, zoom_factors, order=order)
    
    # Ensure float32 dtype
    return resized.astype(np.float32)


def mask_to_boxes_3d(
    mask: np.ndarray,
    min_volume: int = 100
) -> List[dict]:
    """
    Convert segmentation mask to 3D bounding boxes.
    
    Args:
        mask: Binary or multi-label mask (D, H, W)
        min_volume: Minimum volume (voxels) for valid boxes
    
    Returns:
        List of boxes, each dict with:
            - 'box': (cx, cy, cz, w, h, d) normalized to [0, 1]
            - 'label': class label
    """
    boxes = []
    unique_labels = np.unique(mask)
    
    # Skip background (label 0)
    for label in unique_labels:
        if label == 0:
            continue
        
        # Get binary mask for this label
        binary_mask = (mask == label).astype(np.uint8)
        
        # Find connected components
        labeled_mask, num_components = ndimage.label(binary_mask)
        
        for component_id in range(1, num_components + 1):
            component_mask = (labeled_mask == component_id)
            
            # Get bounding box
            coords = np.argwhere(component_mask)
            
            if len(coords) < min_volume:
                continue
            
            # Get min/max coordinates
            z_min, y_min, x_min = coords.min(axis=0)
            z_max, y_max, x_max = coords.max(axis=0)
            
            # Calculate center and size
            cx = (x_min + x_max) / 2
            cy = (y_min + y_max) / 2
            cz = (z_min + z_max) / 2
            w = x_max - x_min + 1
            h = y_max - y_min + 1
            d = z_max - z_min + 1
            
            # Normalize to [0, 1]
            D, H, W = mask.shape
            box_normalized = np.array([
                cx / W,
                cy / H,
                cz / D,
                w / W,
                h / H,
                d / D
            ])
            
            # Clip to ensure valid range [0, 1]
            box_normalized = np.clip(box_normalized, 0.0, 1.0)
            
            boxes.append({
                'box': box_normalized,
                'label': int(label)
            })
    
    return boxes


def apply_augmentation_3d(
    volume: np.ndarray,
    boxes: List[dict],
    flip_prob: float = 0.5,
    rotate_prob: float = 0.3,
    intensity_jitter: float = 0.1
) -> Tuple[np.ndarray, List[dict]]:
    """
    Apply 3D augmentations to volume and boxes.
    
    Args:
        volume: Input volume (D, H, W)
        boxes: List of bounding boxes
        flip_prob: Probability of random flip
        rotate_prob: Probability of 90-degree rotation
        intensity_jitter: Standard deviation for intensity jittering
    
    Returns:
        Augmented volume and boxes
    """
    aug_volume = volume.copy()
    aug_boxes = [box.copy() for box in boxes]
    
    # Random flip along x-axis
    if np.random.rand() < flip_prob:
        aug_volume = np.flip(aug_volume, axis=2)
        for box in aug_boxes:
            box['box'][0] = 1.0 - box['box'][0]  # cx
    
    # Random flip along y-axis
    if np.random.rand() < flip_prob:
        aug_volume = np.flip(aug_volume, axis=1)
        for box in aug_boxes:
            box['box'][1] = 1.0 - box['box'][1]  # cy
    
    # Random 90-degree rotation in xy plane
    if np.random.rand() < rotate_prob:
        k = np.random.randint(1, 4)  # 90, 180, or 270 degrees
        aug_volume = np.rot90(aug_volume, k=k, axes=(1, 2))
        
        # Rotate boxes (simplified - only works for k=2, 180 degrees)
        if k == 2:
            for box in aug_boxes:
                box['box'][0] = 1.0 - box['box'][0]
                box['box'][1] = 1.0 - box['box'][1]
    
    # Intensity jittering
    if intensity_jitter > 0:
        noise = np.random.normal(0, intensity_jitter, aug_volume.shape)
        aug_volume = np.clip(aug_volume + noise, 0, 1)
    
    # Clip all box coordinates to [0, 1] after all transformations
    for box in aug_boxes:
        box['box'] = np.clip(box['box'], 0.0, 1.0)
    
    return aug_volume, aug_boxes


def collate_fn(batch: List[dict]) -> dict:
    """
    Custom collate function for batching variable-sized volumes.
    
    Args:
        batch: List of samples, each with 'volume', 'boxes', 'labels'
    
    Returns:
        Batched dict with:
            - 'volumes': List of tensors (different sizes)
            - 'boxes': List of box lists
            - 'labels': List of label lists
    """
    volumes = [item['volume'] for item in batch]
    boxes = [item['boxes'] for item in batch]
    labels = [item['labels'] for item in batch]
    
    return {
        'volumes': volumes,
        'boxes': boxes,
        'labels': labels
    }
