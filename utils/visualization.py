"""
Visualization utilities for 3D object detection on CT volumes.
Provides functions to visualize 3D bounding boxes on 2D slices.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from typing import List, Tuple, Optional, Dict
import cv2
from pathlib import Path
from scipy import ndimage


# Class color map (can be customized)
CLASS_COLORS = {
    0: (1.0, 1.0, 1.0),      # White - background
    1: (1.0, 0.0, 0.0),      # Red - liver
    2: (0.0, 1.0, 0.0),      # Green - spleen
    3: (0.0, 0.0, 1.0),      # Blue - LK
    4: (1.0, 1.0, 0.0),      # Yellow - RK
    5: (1.0, 0.0, 1.0),      # Magenta- bowel
}

CLASS_NAMES = {
    0: 'background',
    1: 'liver',
    2: 'spleen',
    3: 'LK',
    4: 'RK',
    5: 'bowel'
}


def resize_volume(
    volume: np.ndarray,
    target_size: int = 512
) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """
    Resize volume to have H and W dimensions equal to target_size,
    and resize D proportionally.
    
    Args:
        volume: Input volume (D, H, W)
        target_size: Target size for H and W dimensions
    
    Returns:
        Resized volume and scaling factors (scale_d, scale_h, scale_w)
    """
    D, H, W = volume.shape
    
    # Calculate scaling factors
    scale_h = target_size / H
    scale_w = target_size / W
    # Use the same scale for D to maintain aspect ratio
    scale_d = (scale_h + scale_w) / 2
    
    # Calculate new shape
    new_D = int(D * scale_d)
    new_H = target_size
    new_W = target_size
    
    # Resize using scipy's zoom (high-quality interpolation)
    resized_volume = ndimage.zoom(volume, (scale_d, scale_h, scale_w), order=1)
    
    return resized_volume, (scale_d, scale_h, scale_w)



def denormalize_box_3d(
    box: np.ndarray,
    volume_shape: Tuple[int, int, int]
) -> np.ndarray:
    """
    Convert normalized box (0-1) to absolute coordinates.
    
    Args:
        box: Normalized box (cx, cy, cz, w, h, d) in [0, 1]
        volume_shape: Volume shape (D, H, W)
    
    Returns:
        Absolute box (cx, cy, cz, w, h, d) in voxels
    """
    D, H, W = volume_shape
    
    box_abs = box.copy()
    box_abs[0] *= W  # cx
    box_abs[1] *= H  # cy
    box_abs[2] *= D  # cz
    box_abs[3] *= W  # w
    box_abs[4] *= H  # h
    box_abs[5] *= D  # d
    
    return box_abs


def box_3d_to_2d_slice(
    box_3d: np.ndarray,
    slice_idx: int,
    axis: str = 'axial'
) -> Optional[Tuple[float, float, float, float]]:
    """
    Project 3D box onto 2D slice.
    
    Args:
        box_3d: Absolute 3D box (cx, cy, cz, w, h, d)
        slice_idx: Slice index
        axis: View axis ('axial', 'sagittal', 'coronal')
    
    Returns:
        2D box (x_min, y_min, width, height) or None if box doesn't intersect slice
    """
    cx, cy, cz, w, h, d = box_3d
    
    # Calculate 3D box bounds
    x_min = cx - w / 2
    x_max = cx + w / 2
    y_min = cy - h / 2
    y_max = cy + h / 2
    z_min = cz - d / 2
    z_max = cz + d / 2
    
    if axis == 'axial':
        # View from top (z-axis)
        if not (z_min <= slice_idx <= z_max):
            return None
        return (x_min, y_min, w, h)
    
    elif axis == 'sagittal':
        # View from side (x-axis)
        if not (x_min <= slice_idx <= x_max):
            return None
        return (y_min, z_min, h, d)
    
    elif axis == 'coronal':
        # View from front (y-axis)
        if not (y_min <= slice_idx <= y_max):
            return None
        return (x_min, z_min, w, d)
    
    return None


def draw_box_on_slice(
    image: np.ndarray,
    box_3d: np.ndarray,
    slice_idx: int,
    axis: str = 'axial',
    color: Tuple[float, float, float] = (1.0, 0.0, 0.0),
    thickness: int = 1,
    label: Optional[str] = None,
    score: Optional[float] = None
) -> np.ndarray:
    """
    Draw 3D bounding box on 2D slice.
    
    Args:
        image: 2D slice image (H, W) or (H, W, 3)
        box_3d: Absolute 3D box (cx, cy, cz, w, h, d)
        slice_idx: Current slice index
        axis: View axis
        color: Box color (R, G, B) in [0, 1]
        thickness: Line thickness
        label: Optional label text
        score: Optional confidence score
    
    Returns:
        Image with box drawn
    """
    # Convert grayscale to RGB if needed
    if len(image.shape) == 2:
        image_rgb = np.stack([image] * 3, axis=-1)
    else:
        image_rgb = image.copy()
    
    # Ensure float range [0, 1]
    if image_rgb.max() > 1.0:
        image_rgb = image_rgb / 255.0
    
    # Get 2D box projection
    box_2d = box_3d_to_2d_slice(box_3d, slice_idx, axis)
    if box_2d is None:
        return image_rgb
    
    x_min, y_min, width, height = box_2d
    
    # Convert to integer coordinates
    x_min = int(x_min)
    y_min = int(y_min)
    x_max = int(x_min + width)
    y_max = int(y_min + height)
    
    # Draw rectangle
    cv2.rectangle(
        image_rgb,
        (x_min, y_min),
        (x_max, y_max),
        color,
        thickness
    )
    
    # Add label if provided
    if label is not None:
        text = label
        if score is not None:
            text += f' {score:.2f}'
        
        # Add text background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.3
        font_thickness = 1
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
        
        # Draw semi-transparent background
        cv2.rectangle(
            image_rgb,
            (x_min, y_min - text_h - baseline - 2),
            (x_min + text_w + 2, y_min),
            color,
            -1  # Filled
        )
        
        # Draw text in white for better contrast
        cv2.putText(
            image_rgb,
            text,
            (x_min + 1, y_min - baseline - 1),
            font,
            font_scale,
            (1.0, 1.0, 1.0),
            font_thickness,
            cv2.LINE_AA
        )
    
    return image_rgb


def draw_dashed_box_on_ax(
    ax: plt.Axes,
    box_3d: np.ndarray,
    slice_idx: int,
    axis: str = 'axial',
    color: Tuple[float, float, float] = (1.0, 0.0, 0.0),
    linewidth: float = 2,
    linestyle: str = '--',
    label: Optional[str] = None
):
    """
    Draw 3D bounding box on matplotlib axis with customizable line style.
    
    Args:
        ax: Matplotlib axis
        box_3d: Absolute 3D box (cx, cy, cz, w, h, d)
        slice_idx: Current slice index
        axis: View axis
        color: Box color (R, G, B) in [0, 1]
        linewidth: Line width
        linestyle: Line style ('-' for solid, '--' for dashed)
        label: Optional label text
    """
    # Get 2D box projection
    box_2d = box_3d_to_2d_slice(box_3d, slice_idx, axis)
    if box_2d is None:
        return
    
    x_min, y_min, width, height = box_2d
    
    # Create rectangle patch
    rect = patches.Rectangle(
        (x_min, y_min),
        width,
        height,
        linewidth=linewidth,
        edgecolor=color,
        facecolor='none',
        linestyle=linestyle
    )
    ax.add_patch(rect)
    
    # Add label if provided
    if label is not None:
        ax.text(
            x_min,
            y_min - 5,
            label,
            color=color,
            fontsize=8,
            weight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.5)
        )


def visualize_single_slice(
    volume: np.ndarray,
    slice_idx: int,
    pred_boxes: Optional[List[np.ndarray]] = None,
    pred_labels: Optional[List[int]] = None,
    pred_scores: Optional[List[float]] = None,
    gt_boxes: Optional[List[np.ndarray]] = None,
    gt_labels: Optional[List[int]] = None,
    axis: str = 'axial',
    score_threshold: float = 0.0,
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Visualize predictions and ground truth on a single slice.
    Volume is resized to 512x512 (H, W) with proportional D scaling.
    
    Args:
        volume: 3D volume (D, H, W)
        slice_idx: Slice index to visualize
        pred_boxes: List of predicted boxes (normalized)
        pred_labels: List of predicted class labels
        pred_scores: List of prediction scores
        gt_boxes: List of ground truth boxes (normalized)
        gt_labels: List of ground truth labels
        axis: View axis
        score_threshold: Minimum score to display predictions (default: 0.0, show all)
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    # Resize volume to 512x512
    resized_volume, scale_factors = resize_volume(volume, target_size=512)
    scale_d, scale_h, scale_w = scale_factors
    
    # Adjust slice_idx based on scaling
    if axis == 'axial':
        adjusted_slice_idx = int(slice_idx * scale_d)
        adjusted_slice_idx = min(adjusted_slice_idx, resized_volume.shape[0] - 1)
    elif axis == 'sagittal':
        adjusted_slice_idx = int(slice_idx * scale_w)
        adjusted_slice_idx = min(adjusted_slice_idx, resized_volume.shape[2] - 1)
    elif axis == 'coronal':
        adjusted_slice_idx = int(slice_idx * scale_h)
        adjusted_slice_idx = min(adjusted_slice_idx, resized_volume.shape[1] - 1)
    else:
        raise ValueError(f"Unknown axis: {axis}")
    
    # Extract slice
    if axis == 'axial':
        slice_img = resized_volume[adjusted_slice_idx, :, :]
    elif axis == 'sagittal':
        slice_img = resized_volume[:, :, adjusted_slice_idx]
    elif axis == 'coronal':
        slice_img = resized_volume[:, adjusted_slice_idx, :]
    
    # Normalize to [0, 1]
    slice_img = (slice_img - slice_img.min()) / (slice_img.max() - slice_img.min() + 1e-8)
    
    # Create figure
    if gt_boxes is not None and len(gt_boxes) > 0:
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        ax_pred, ax_gt = axes
    else:
        fig, axes = plt.subplots(1, 1, figsize=(figsize[0]//2, figsize[1]))
        ax_pred = axes
        ax_gt = None
    
    # Draw predictions
    ax_pred.imshow(slice_img, cmap='gray')
    if pred_boxes is not None and len(pred_boxes) > 0:
        for i, box in enumerate(pred_boxes):
            if pred_scores is not None and pred_scores[i] < score_threshold:
                continue
            
            # Denormalize box (using RESIZED volume shape)
            box_abs = denormalize_box_3d(box, resized_volume.shape)
            
            # Get color and label
            label_idx = pred_labels[i] if pred_labels is not None else 0
            color = CLASS_COLORS.get(label_idx, (1.0, 1.0, 1.0))
            label_name = CLASS_NAMES.get(label_idx, f'class_{label_idx}')
            score = pred_scores[i] if pred_scores is not None else None
            
            # Add score to label if available
            if score is not None:
                label_text = f'{label_name} {score:.2f}'
            else:
                label_text = label_name
            
            # Draw box with solid line
            draw_dashed_box_on_ax(
                ax_pred, box_abs, adjusted_slice_idx, axis,
                color=color, linewidth=2, linestyle='-',
                label=label_text
            )
    
    ax_pred.set_title(f'Predictions - {axis.capitalize()} Slice {slice_idx}')
    ax_pred.axis('off')
    
    # Draw ground truth
    if ax_gt is not None:
        ax_gt.imshow(slice_img, cmap='gray')
        if gt_boxes is not None and len(gt_boxes) > 0:
            for i, box in enumerate(gt_boxes):
                # Denormalize box (using RESIZED volume shape)
                box_abs = denormalize_box_3d(box, resized_volume.shape)
                
                # Get color and label
                label_idx = gt_labels[i] if gt_labels is not None else 0
                color = CLASS_COLORS.get(label_idx, (1.0, 1.0, 1.0))
                label_name = CLASS_NAMES.get(label_idx, f'class_{label_idx}')
                
                # Draw box with DASHED line
                draw_dashed_box_on_ax(
                    ax_gt, box_abs, adjusted_slice_idx, axis,
                    color=color, linewidth=2, linestyle='--',
                    label=label_name
                )
        
        ax_gt.set_title(f'Ground Truth - {axis.capitalize()} Slice {slice_idx}')
        ax_gt.axis('off')
    
    plt.tight_layout()
    return fig



def visualize_multi_slice(
    volume: np.ndarray,
    pred_boxes: Optional[List[np.ndarray]] = None,
    pred_labels: Optional[List[int]] = None,
    pred_scores: Optional[List[float]] = None,
    gt_boxes: Optional[List[np.ndarray]] = None,
    gt_labels: Optional[List[int]] = None,
    num_slices: int = 9,
    axis: str = 'axial',
    score_threshold: float = 0.0,
    figsize: Tuple[int, int] = (15, 10)
) -> plt.Figure:
    """
    Visualize predictions across multiple evenly-spaced slices.
    Volume is resized to 512x512 (H, W) with proportional D scaling.
    
    Args:
        volume: 3D volume (D, H, W)
        pred_boxes: List of predicted boxes (normalized)
        pred_labels: List of predicted class labels
        pred_scores: List of prediction scores
        gt_boxes: List of ground truth boxes (normalized)
        gt_labels: List of ground truth labels
        num_slices: Number of slices to show
        axis: View axis
        score_threshold: Minimum score to display (default: 0.0, show all)
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    # Resize volume to 512x512
    resized_volume, scale_factors = resize_volume(volume, target_size=512)
    scale_d, scale_h, scale_w = scale_factors
    
    # Determine slice indices in ORIGINAL volume
    if axis == 'axial':
        depth = volume.shape[0]
    elif axis == 'sagittal':
        depth = volume.shape[2]
    elif axis == 'coronal':
        depth = volume.shape[1]
    else:
        raise ValueError(f"Unknown axis: {axis}")
    
    slice_indices = np.linspace(0, depth - 1, num_slices, dtype=int)
    
    # Create grid
    cols = int(np.ceil(np.sqrt(num_slices)))
    rows = int(np.ceil(num_slices / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten() if num_slices > 1 else [axes]
    
    for idx, slice_idx in enumerate(slice_indices):
        ax = axes[idx]
        
        # Adjust slice_idx based on scaling
        if axis == 'axial':
            adjusted_slice_idx = int(slice_idx * scale_d)
            adjusted_slice_idx = min(adjusted_slice_idx, resized_volume.shape[0] - 1)
        elif axis == 'sagittal':
            adjusted_slice_idx = int(slice_idx * scale_w)
            adjusted_slice_idx = min(adjusted_slice_idx, resized_volume.shape[2] - 1)
        elif axis == 'coronal':
            adjusted_slice_idx = int(slice_idx * scale_h)
            adjusted_slice_idx = min(adjusted_slice_idx, resized_volume.shape[1] - 1)
        
        # Extract slice from RESIZED volume
        if axis == 'axial':
            slice_img = resized_volume[adjusted_slice_idx, :, :]
        elif axis == 'sagittal':
            slice_img = resized_volume[:, :, adjusted_slice_idx]
        else:  # coronal
            slice_img = resized_volume[:, adjusted_slice_idx, :]
        
        # Normalize
        slice_img = (slice_img - slice_img.min()) / (slice_img.max() - slice_img.min() + 1e-8)
        
        # Display image
        ax.imshow(slice_img, cmap='gray')
        
        # Draw predictions with SOLID lines
        if pred_boxes is not None and len(pred_boxes) > 0:
            for i, box in enumerate(pred_boxes):
                if pred_scores is not None and pred_scores[i] < score_threshold:
                    continue
                
                box_abs = denormalize_box_3d(box, resized_volume.shape)
                label_idx = pred_labels[i] if pred_labels is not None else 0
                color = CLASS_COLORS.get(label_idx, (1.0, 1.0, 1.0))
                
                draw_dashed_box_on_ax(
                    ax, box_abs, adjusted_slice_idx, axis,
                    color=color, linewidth=1.5, linestyle='-'
                )
        
        # Draw ground truth with DASHED lines
        if gt_boxes is not None and len(gt_boxes) > 0:
            for i, box in enumerate(gt_boxes):
                box_abs = denormalize_box_3d(box, resized_volume.shape)
                label_idx = gt_labels[i] if gt_labels is not None else 0
                color = CLASS_COLORS.get(label_idx, (1.0, 1.0, 1.0))
                
                # Draw GT with dashed line
                draw_dashed_box_on_ax(
                    ax, box_abs, adjusted_slice_idx, axis,
                    color=color, linewidth=1.5, linestyle='--'
                )
        
        ax.set_title(f'{axis.capitalize()} {slice_idx}')
        ax.axis('off')
    
    # Hide unused subplots
    for idx in range(num_slices, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'Multi-Slice View ({axis.capitalize()})', fontsize=16, y=0.98)
    plt.tight_layout()
    return fig



def save_visualization(
    fig: plt.Figure,
    output_path: str,
    dpi: int = 150
):
    """
    Save figure to file.
    
    Args:
        fig: Matplotlib figure
        output_path: Output file path
        dpi: Resolution in dots per inch
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    print(f"Saved visualization to {output_path}")
