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
    alpha: float = 15.0,
    sigma: float = 3.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate elastic deformation displacement fields.
    
    Args:
        volume: Input volume (D, H, W)
        alpha: Deformation intensity
        sigma: Gaussian smoothing sigma
        
    Returns:
        Deformed volume and displacement fields (dz, dy, dx)
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
    
    deformed = ndimage.map_coordinates(volume, indices, order=1, mode='reflect')
    
    return deformed, dz, dy, dx


def rotate_volume_3d(
    volume: np.ndarray,
    angle: float,
    axes: Tuple[int, int] = (1, 2)
) -> np.ndarray:
    """
    Rotate volume by arbitrary angle in specified plane.
    
    Args:
        volume: Input volume (D, H, W)
        angle: Rotation angle in degrees
        axes: Plane of rotation (default: XY plane)
        
    Returns:
        Rotated volume
    """
    return ndimage.rotate(volume, angle, axes=axes, reshape=False, order=1, mode='nearest')


def rotate_boxes_3d(
    boxes: List[dict],
    angle: float,
    volume_shape: Tuple[int, int, int]
) -> List[dict]:
    """
    Rotate bounding boxes by arbitrary angle in XY plane.
    
    Args:
        boxes: List of bounding boxes with normalized coordinates
        angle: Rotation angle in degrees
        volume_shape: Shape of the volume (D, H, W)
        
    Returns:
        Rotated boxes
    """
    angle_rad = np.radians(angle)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    
    rotated_boxes = []
    for box in boxes:
        box_copy = box.copy()
        box_copy['box'] = box['box'].copy()
        
        # Get center coordinates (normalized)
        cx, cy = box['box'][0], box['box'][1]
        
        # Convert to centered coordinates (origin at center of volume)
        cx_centered = cx - 0.5
        cy_centered = cy - 0.5
        
        # Rotate center point
        new_cx = cx_centered * cos_a - cy_centered * sin_a + 0.5
        new_cy = cx_centered * sin_a + cy_centered * cos_a + 0.5
        
        box_copy['box'][0] = new_cx
        box_copy['box'][1] = new_cy
        
        # For small angles, width/height change is minimal
        # For more accurate rotation, we could compute the bounding box of rotated corners
        # but for angles within ±30°, this approximation is acceptable
        w, h = box['box'][3], box['box'][4]
        # Approximate new bounding box dimensions after rotation
        new_w = abs(w * cos_a) + abs(h * sin_a)
        new_h = abs(w * sin_a) + abs(h * cos_a)
        box_copy['box'][3] = new_w
        box_copy['box'][4] = new_h
        
        rotated_boxes.append(box_copy)
    
    return rotated_boxes


def random_scale_3d(
    volume: np.ndarray,
    boxes: List[dict],
    scale_range: Tuple[float, float] = (0.8, 1.2)
) -> Tuple[np.ndarray, List[dict]]:
    """
    Apply random scaling to volume and adjust boxes accordingly.
    
    Args:
        volume: Input volume (D, H, W)
        boxes: List of bounding boxes
        scale_range: Range of scale factors (min, max)
        
    Returns:
        Scaled volume and adjusted boxes
    """
    scale = np.random.uniform(scale_range[0], scale_range[1])
    
    original_shape = np.array(volume.shape)
    
    # Scale the volume
    scaled_volume = ndimage.zoom(volume, scale, order=1)
    
    # Use actual scaled shape (may differ from original_shape * scale due to rounding)
    actual_scaled_shape = np.array(scaled_volume.shape)
    
    # If scaled larger, crop to original size from center
    # If scaled smaller, pad to original size
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
        # Adjust boxes: shift towards center
        scale_boxes = []
        for box in boxes:
            box_copy = box.copy()
            box_copy['box'] = box['box'].copy()
            # Scale and shift center coordinates
            box_copy['box'][0] = (box['box'][0] - 0.5) / scale + 0.5
            box_copy['box'][1] = (box['box'][1] - 0.5) / scale + 0.5
            box_copy['box'][2] = (box['box'][2] - 0.5) / scale + 0.5
            # Scale dimensions
            box_copy['box'][3] = box['box'][3] / scale
            box_copy['box'][4] = box['box'][4] / scale
            box_copy['box'][5] = box['box'][5] / scale
            scale_boxes.append(box_copy)
    else:
        # Pad with zeros (or edge values)
        start = (original_shape - actual_scaled_shape) // 2
        # Calculate actual end indices based on scaled volume shape
        end = start + actual_scaled_shape
        result_volume[
            start[0]:end[0],
            start[1]:end[1],
            start[2]:end[2]
        ] = scaled_volume
        # Adjust boxes: shift away from center
        scale_boxes = []
        for box in boxes:
            box_copy = box.copy()
            box_copy['box'] = box['box'].copy()
            # Scale and shift center coordinates
            box_copy['box'][0] = (box['box'][0] - 0.5) * scale + 0.5
            box_copy['box'][1] = (box['box'][1] - 0.5) * scale + 0.5
            box_copy['box'][2] = (box['box'][2] - 0.5) * scale + 0.5
            # Scale dimensions
            box_copy['box'][3] = box['box'][3] * scale
            box_copy['box'][4] = box['box'][4] * scale
            box_copy['box'][5] = box['box'][5] * scale
            scale_boxes.append(box_copy)
    
    return result_volume, scale_boxes


def random_crop_3d(
    volume: np.ndarray,
    boxes: List[dict],
    crop_ratio_range: Tuple[float, float] = (0.8, 1.0)
) -> Tuple[np.ndarray, List[dict]]:
    """
    Apply random cropping to volume and adjust boxes accordingly.
    Ensures at least one box center remains in the cropped region.
    
    Args:
        volume: Input volume (D, H, W)
        boxes: List of bounding boxes
        crop_ratio_range: Range of crop ratios per dimension
        
    Returns:
        Cropped and resized volume with adjusted boxes
    """
    if len(boxes) == 0:
        return volume, boxes
    
    original_shape = np.array(volume.shape)
    
    # Random crop ratio for each dimension
    crop_ratios = np.random.uniform(crop_ratio_range[0], crop_ratio_range[1], size=3)
    crop_shape = (original_shape * crop_ratios).astype(int)
    crop_shape = np.maximum(crop_shape, 16)  # Minimum crop size
    # Ensure crop_shape doesn't exceed original_shape
    crop_shape = np.minimum(crop_shape, original_shape)
    
    # Choose a random box to ensure it stays in view
    anchor_box = boxes[np.random.randint(len(boxes))]
    anchor_center = np.array([
        anchor_box['box'][2] * original_shape[0],  # cz -> D
        anchor_box['box'][1] * original_shape[1],  # cy -> H  
        anchor_box['box'][0] * original_shape[2],  # cx -> W
    ])
    
    # Calculate valid crop start range that includes the anchor
    max_start = np.minimum(
        anchor_center.astype(int),
        original_shape - crop_shape
    )
    max_start = np.maximum(max_start, 0)  # Ensure non-negative
    
    min_start = np.maximum(
        (anchor_center - crop_shape + 1).astype(int),
        0
    )
    
    # Ensure min_start <= max_start
    min_start = np.minimum(min_start, max_start)
    
    # Random start within valid range
    start = np.array([
        np.random.randint(min_start[i], max_start[i] + 1)
        for i in range(3)
    ])
    
    # Ensure we don't exceed boundaries
    end = np.minimum(start + crop_shape, original_shape)
    start = end - crop_shape  # Adjust start if end was clipped
    start = np.maximum(start, 0)
    
    # Crop the volume
    cropped = volume[
        start[0]:start[0] + crop_shape[0],
        start[1]:start[1] + crop_shape[1],
        start[2]:start[2] + crop_shape[2]
    ]
    
    # Resize back to original shape using zoom factors
    actual_crop_shape = np.array(cropped.shape)
    zoom_factors = original_shape / actual_crop_shape
    result_volume = ndimage.zoom(cropped, zoom_factors, order=1)
    
    # Ensure result matches original shape (handle rounding issues)
    if result_volume.shape != tuple(original_shape):
        # Resize to exact original shape
        result_volume = resize_volume(result_volume, tuple(original_shape), order=1)
    
    # Adjust boxes
    cropped_boxes = []
    for box in boxes:
        box_copy = box.copy()
        box_copy['box'] = box['box'].copy()
        
        # Convert to pixel coordinates
        cx = box['box'][0] * original_shape[2]
        cy = box['box'][1] * original_shape[1]
        cz = box['box'][2] * original_shape[0]
        w = box['box'][3] * original_shape[2]
        h = box['box'][4] * original_shape[1]
        d = box['box'][5] * original_shape[0]
        
        # Shift by crop start
        cx_new = cx - start[2]
        cy_new = cy - start[1]
        cz_new = cz - start[0]
        
        # Scale to match resize (use actual crop shape)
        cx_new = cx_new * (original_shape[2] / actual_crop_shape[2])
        cy_new = cy_new * (original_shape[1] / actual_crop_shape[1])
        cz_new = cz_new * (original_shape[0] / actual_crop_shape[0])
        w_new = w * (original_shape[2] / actual_crop_shape[2])
        h_new = h * (original_shape[1] / actual_crop_shape[1])
        d_new = d * (original_shape[0] / actual_crop_shape[0])
        
        # Normalize back
        box_copy['box'][0] = cx_new / original_shape[2]
        box_copy['box'][1] = cy_new / original_shape[1]
        box_copy['box'][2] = cz_new / original_shape[0]
        box_copy['box'][3] = w_new / original_shape[2]
        box_copy['box'][4] = h_new / original_shape[1]
        box_copy['box'][5] = d_new / original_shape[0]
        
        # Only keep boxes with center inside valid range
        if (0 < box_copy['box'][0] < 1 and 
            0 < box_copy['box'][1] < 1 and 
            0 < box_copy['box'][2] < 1):
            cropped_boxes.append(box_copy)
    
    # If all boxes were cropped out, return original
    if len(cropped_boxes) == 0:
        return volume, boxes
    
    return result_volume, cropped_boxes


def apply_augmentation_3d(
    volume: np.ndarray,
    boxes: List[dict],
    rotate_prob: float = 0.5,
    rotate_range: float = 30.0,
    scale_prob: float = 0.5,
    scale_range: Tuple[float, float] = (0.85, 1.15),
    elastic_prob: float = 0.3,
    elastic_alpha: float = 15.0,
    elastic_sigma: float = 3.0,
    intensity_jitter: float = 0.1
) -> Tuple[np.ndarray, List[dict]]:
    """
    Apply 3D augmentations to volume and boxes.
    Designed for human CT scans - no flipping (anatomically incorrect).
    
    Args:
        volume: Input volume (D, H, W)
        boxes: List of bounding boxes
        rotate_prob: Probability of random rotation
        rotate_range: Max rotation angle in degrees (±rotate_range)
        scale_prob: Probability of random scaling
        scale_range: Range of scale factors (min, max)
        elastic_prob: Probability of elastic deformation
        elastic_alpha: Elastic deformation intensity
        elastic_sigma: Elastic deformation smoothness
        intensity_jitter: Standard deviation for intensity jittering
    
    Returns:
        Augmented volume and boxes
    """
    aug_volume = volume.copy()
    aug_boxes = [{'box': box['box'].copy(), 'label': box['label']} for box in boxes]
    
    # Random rotation in XY plane (±rotate_range degrees)
    if np.random.rand() < rotate_prob:
        angle = np.random.uniform(-rotate_range, rotate_range)
        aug_volume = rotate_volume_3d(aug_volume, angle, axes=(1, 2))
        aug_boxes = rotate_boxes_3d(aug_boxes, angle, aug_volume.shape)
    
    # Random scaling
    if np.random.rand() < scale_prob:
        aug_volume, aug_boxes = random_scale_3d(aug_volume, aug_boxes, scale_range)
    
    # Elastic deformation (applied to volume only, boxes are approximate)
    if np.random.rand() < elastic_prob:
        aug_volume, _, _, _ = elastic_deformation_3d(
            aug_volume, alpha=elastic_alpha, sigma=elastic_sigma
        )
    
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
