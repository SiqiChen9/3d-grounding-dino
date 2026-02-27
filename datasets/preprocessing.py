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
    min_volume: int = 50,
    merge_per_label: bool = True
) -> List[dict]:
    """
    Convert segmentation mask to 3D bounding boxes.
    
    Args:
        mask: Binary or multi-label mask (D, H, W)
        min_volume: Minimum volume (voxels) for valid components (noise filter)
        merge_per_label: If True, merge all boxes of same label into one
    
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
    
    # Merge boxes per label if requested
    if merge_per_label and boxes:
        merged = {}
        for b in boxes:
            label = b['label']
            box = b['box']  # cx, cy, cz, w, h, d
            x1, y1, z1 = box[0] - box[3]/2, box[1] - box[4]/2, box[2] - box[5]/2
            x2, y2, z2 = box[0] + box[3]/2, box[1] + box[4]/2, box[2] + box[5]/2
            if label not in merged:
                merged[label] = [x1, y1, z1, x2, y2, z2]
            else:
                merged[label] = [
                    min(merged[label][0], x1), min(merged[label][1], y1), min(merged[label][2], z1),
                    max(merged[label][3], x2), max(merged[label][4], y2), max(merged[label][5], z2)
                ]
        boxes = [{'box': np.array([
            (m[0]+m[3])/2, (m[1]+m[4])/2, (m[2]+m[5])/2,
            m[3]-m[0], m[4]-m[1], m[5]-m[2]
        ]), 'label': l} for l, m in merged.items()]
    
    return boxes


def elastic_deformation_3d(
    volume: np.ndarray,
    mask: np.ndarray,
    alpha: float = 15.0,
    sigma: float = 3.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply elastic deformation to volume and mask using shared displacement fields.
    
    Args:
        volume: Input volume (D, H, W)
        mask: Segmentation mask (D, H, W)
        alpha: Deformation intensity
        sigma: Gaussian smoothing sigma
        
    Returns:
        Deformed volume and deformed mask
    """
    shape = volume.shape
    
    # Generate random displacement fields
    dz = ndimage.gaussian_filter(np.random.randn(*shape) * alpha, sigma, mode='reflect')
    dy = ndimage.gaussian_filter(np.random.randn(*shape) * alpha, sigma, mode='reflect')
    dx = ndimage.gaussian_filter(np.random.randn(*shape) * alpha, sigma, mode='reflect')
    
    # Create coordinate grids
    z, y, x = np.meshgrid(
        np.arange(shape[0]),
        np.arange(shape[1]),
        np.arange(shape[2]),
        indexing='ij'
    )
    
    # Apply displacement
    indices = [
        np.clip(z + dz, 0, shape[0] - 1),
        np.clip(y + dy, 0, shape[1] - 1),
        np.clip(x + dx, 0, shape[2] - 1)
    ]
    
    deformed_volume = ndimage.map_coordinates(volume, indices, order=1, mode='constant', cval=0)
    deformed_mask = ndimage.map_coordinates(mask.astype(np.float64), indices, order=0, mode='constant', cval=0)  # nearest for labels
    
    return deformed_volume, deformed_mask


def rotate_volume_3d(
    volume: np.ndarray,
    angle: float,
    axes: Tuple[int, int] = (1, 2),
    order: int = 1
) -> np.ndarray:
    """
    Rotate volume by arbitrary angle in specified plane.
    
    Args:
        volume: Input volume (D, H, W)
        angle: Rotation angle in degrees
        axes: Plane of rotation (default: XY plane)
        order: Interpolation order (1=linear for volume, 0=nearest for mask)
        
    Returns:
        Rotated volume
    """
    return ndimage.rotate(volume, angle, axes=axes, reshape=False, order=order, mode='constant', cval=0)


def random_scale_volume(
    volume: np.ndarray,
    scale: float,
    order: int = 1
) -> np.ndarray:
    """
    Scale a volume and crop/pad back to original size.
    
    Args:
        volume: Input volume (D, H, W)
        scale: Scale factor
        order: Interpolation order (1=linear for volume, 0=nearest for mask)
        
    Returns:
        Scaled volume with original shape
    """
    original_shape = np.array(volume.shape)
    
    scaled_volume = ndimage.zoom(volume, scale, order=order)
    actual_scaled_shape = np.array(scaled_volume.shape)
    
    result_volume = np.zeros(original_shape, dtype=volume.dtype)
    
    if scale > 1.0:
        # Crop from center
        start = (actual_scaled_shape - original_shape) // 2
        end = start + original_shape
        result_volume = scaled_volume[
            start[0]:end[0],
            start[1]:end[1],
            start[2]:end[2]
        ]
    else:
        # Pad with zeros
        start = (original_shape - actual_scaled_shape) // 2
        end = start + actual_scaled_shape
        result_volume[
            start[0]:end[0],
            start[1]:end[1],
            start[2]:end[2]
        ] = scaled_volume
    
    return result_volume


def apply_augmentation_3d(
    volume: np.ndarray,
    mask: np.ndarray,
    rotate_prob: float = 0.5,
    rotate_range: float = 30.0,
    scale_prob: float = 0.5,
    scale_range: Tuple[float, float] = (0.85, 1.15),
    elastic_prob: float = 0.3,
    elastic_alpha: float = 15.0,
    elastic_sigma: float = 3.0,
    intensity_jitter: float = 0.1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply 3D augmentations to volume and mask jointly.
    Bounding boxes are extracted from the augmented mask afterwards.
    Designed for human CT scans - no flipping (anatomically incorrect).
    
    Args:
        volume: Input volume (D, H, W), normalized to [0, 1]
        mask: Segmentation mask (D, H, W), integer labels
        rotate_prob: Probability of random rotation
        rotate_range: Max rotation angle in degrees (±rotate_range)
        scale_prob: Probability of random scaling
        scale_range: Range of scale factors (min, max)
        elastic_prob: Probability of elastic deformation
        elastic_alpha: Elastic deformation intensity
        elastic_sigma: Elastic deformation smoothness
        intensity_jitter: Standard deviation for intensity jittering
    
    Returns:
        Augmented volume and augmented mask
    """
    aug_volume = volume.copy()
    aug_mask = mask.copy()
    
    # Random rotation in XY plane (±rotate_range degrees)
    if np.random.rand() < rotate_prob:
        angle = np.random.uniform(-rotate_range, rotate_range)
        aug_volume = rotate_volume_3d(aug_volume, angle, axes=(1, 2), order=1)
        aug_mask = rotate_volume_3d(aug_mask, angle, axes=(1, 2), order=0).astype(mask.dtype)
    
    # Random scaling
    if np.random.rand() < scale_prob:
        scale = np.random.uniform(scale_range[0], scale_range[1])
        aug_volume = random_scale_volume(aug_volume, scale, order=1)
        aug_mask = random_scale_volume(aug_mask, scale, order=0).astype(mask.dtype)
    
    # Elastic deformation
    if np.random.rand() < elastic_prob:
        aug_volume, aug_mask = elastic_deformation_3d(
            aug_volume, aug_mask, alpha=elastic_alpha, sigma=elastic_sigma
        )
        aug_mask = aug_mask.astype(mask.dtype)
    
    # Intensity jittering (volume only, not mask)
    if intensity_jitter > 0:
        noise = np.random.normal(0, intensity_jitter, aug_volume.shape)
        aug_volume = np.clip(aug_volume + noise, 0, 1)
    
    return aug_volume, aug_mask


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
