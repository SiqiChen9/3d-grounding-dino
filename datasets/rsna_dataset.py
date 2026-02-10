"""
RSNA CT Volume Dataset for 3D object detection.
Loads NIfTI segmentations and JPEG slices, converts to 3D bounding boxes.
"""
import os
import numpy as np
import nibabel as nib
from PIL import Image
from typing import Tuple, List, Optional, Dict
import torch
from torch.utils.data import Dataset

from .preprocessing import (
    normalize_intensity,
    resize_volume,
    mask_to_boxes_3d,
    apply_augmentation_3d
)


class RSNAVolumeDataset(Dataset):
    """
    PyTorch Dataset for RSNA CT volumes with segmentation masks.
    
    Loads:
        - NIfTI segmentation files
        - JPEG slice images (stacked into volumes)
        - Converts masks to 3D bounding boxes
    """
    
    def __init__(
        self,
        data_dir: str,
        segmentation_dir: str = "segmentations",
        image_dir: str = "train_images",
        volume_size: Tuple[int, int, int] = (64, 128, 128),  # D, H, W
        train: bool = True,
        augment: bool = False,
        num_samples: Optional[int] = None
    ):
        """
        Args:
            data_dir: Root directory containing datasets
            segmentation_dir: Directory with .nii segmentation files
            image_dir: Directory with JPEG slices
            volume_size: Target volume size (D, H, W)
            train: Training or validation mode
            augment: Apply data augmentation
            num_samples: Limit dataset size (for debugging)
        """
        self.data_dir = data_dir
        self.segmentation_dir = os.path.join(data_dir, segmentation_dir)
        self.image_dir = os.path.join(data_dir, image_dir)
        self.volume_size = volume_size
        self.train = train
        self.augment = augment
        
        # Find all segmentation files
        self.seg_files = []
        if os.path.exists(self.segmentation_dir):
            self.seg_files = sorted([
                f for f in os.listdir(self.segmentation_dir)
                if f.endswith('.nii') or f.endswith('.nii.gz')
            ])
        
        if num_samples is not None:
            self.seg_files = self.seg_files[:num_samples]
        
        print(f"Found {len(self.seg_files)} segmentation files")
    
    def __len__(self) -> int:
        return len(self.seg_files)
    
    def load_nifti_volume(self, seg_file: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load NIfTI segmentation and corresponding JPEG images.
        
        Args:
            seg_file: Segmentation filename
        
        Returns:
            volume: Image volume (D, H, W)
            mask: Segmentation mask (D, H, W)
        """
        # Load segmentation
        seg_path = os.path.join(self.segmentation_dir, seg_file)
        seg_nib = nib.load(seg_path)
        seg_data = seg_nib.get_fdata()

        # 1. Rotate segmentation to match image orientation
        seg_data = np.rot90(seg_data, k=1)
        
        # 2. Flip depth dimension
        seg_data = seg_data[:, :, ::-1]

        # 3. Transpose from (H, W, D) to (D, H, W)
        seg_data = np.transpose(seg_data, (2, 0, 1))
        
        # Get study ID from filename (assuming format: study_id.nii)
        study_id = seg_file.replace('.nii.gz', '').replace('.nii', '')
        
        # Find corresponding image directory
        # Try different patterns: direct match, or search in subdirectories
        image_study_dir = None
        
        # Pattern 1: images are in data_dir/train_images/patient_id/study_id/
        if os.path.exists(self.image_dir):
            for patient_dir in os.listdir(self.image_dir):
                patient_path = os.path.join(self.image_dir, patient_dir)
                if os.path.isdir(patient_path):
                    study_path = os.path.join(patient_path, study_id)
                    if os.path.exists(study_path):
                        image_study_dir = study_path
                        break
        
        if image_study_dir is None:
            # If not found, create a dummy volume
            print(f"Warning: No images found for {study_id}, using dummy data")
            volume = np.zeros(seg_data.shape, dtype=np.float32)
        else:
            # Load JPEG slices
            jpeg_files = sorted([
                f for f in os.listdir(image_study_dir)
                if f.endswith('.jpeg') or f.endswith('.jpg')
            ], key=lambda f: int(f.replace('.jpeg', '')))
            
            if len(jpeg_files) == 0:
                volume = np.zeros(seg_data.shape, dtype=np.float32)
            else:
                # Load images
                images = []
                for jpeg_file in jpeg_files:
                    img_path = os.path.join(image_study_dir, jpeg_file)
                    img = np.array(Image.open(img_path))
                    
                    # Convert to grayscale if needed
                    if len(img.shape) == 3:
                        img = img.mean(axis=2)
                    
                    images.append(img)
                
                # Stack into volume
                volume = np.stack(images, axis=0)  # (D, H, W)
        return volume, seg_data
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single sample.
        
        Returns:
            dict with:
                - 'volume': Tensor (1, D, H, W)
                - 'boxes': Tensor (N, 6) - 3D boxes in (cx, cy, cz, w, h, d)
                - 'labels': Tensor (N,) - class labels
                - 'study_id': str
        """
        seg_file = self.seg_files[idx]
        study_id = seg_file.replace('.nii.gz', '').replace('.nii', '')
        
        # Load volume and mask
        volume, mask = self.load_nifti_volume(seg_file)
        
        # Normalize intensity (assume already in HU or similar range)
        # For JPEG images, they're already in [0, 255], so normalize to [0, 1]
        volume = volume.astype(np.float32) / 255.0
        
        # Resize to target size
        if volume.shape != self.volume_size:
            volume = resize_volume(volume, self.volume_size, order=1)
            mask = resize_volume(mask, self.volume_size, order=0)  # Nearest for labels
            # Ensure mask is also float32 for consistency
            mask = mask.astype(np.float32)
        
        # Convert mask to bounding boxes
        boxes_data = mask_to_boxes_3d(mask, min_volume=50)
        
        # Extract boxes and labels
        if len(boxes_data) == 0:
            # No objects found, add dummy box
            boxes = np.zeros((1, 6), dtype=np.float32)
            labels = np.zeros((1,), dtype=np.int64)
        else:
            boxes = np.array([b['box'] for b in boxes_data], dtype=np.float32)
            labels = np.array([b['label'] for b in boxes_data], dtype=np.int64)
        
        # Apply augmentation if enabled
        if self.augment and self.train:
            boxes_list = [{'box': box, 'label': label}
                         for box, label in zip(boxes, labels)]
            volume, boxes_list = apply_augmentation_3d(
                volume, boxes_list,
                flip_prob=0.5,
                rotate_prob=0.3,
                intensity_jitter=0.1
            )
            boxes = np.array([b['box'] for b in boxes_list], dtype=np.float32)
            labels = np.array([b['label'] for b in boxes_list], dtype=np.int64)
        
        # Convert to tensors
        volume_tensor = torch.from_numpy(volume).unsqueeze(0)  # (1, D, H, W)
        boxes_tensor = torch.from_numpy(boxes)  # (N, 6)
        labels_tensor = torch.from_numpy(labels)  # (N,)
        
        return {
            'volume': volume_tensor,
            'boxes': boxes_tensor,
            'labels': labels_tensor,
            'study_id': study_id
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate function for batching.
    
    Since volumes have the same size but different numbers of boxes,
    we keep boxes as a list.
    """
    volumes = torch.stack([item['volume'] for item in batch], dim=0)
    boxes = [item['boxes'] for item in batch]
    labels = [item['labels'] for item in batch]
    study_ids = [item['study_id'] for item in batch]
    
    return {
        'volumes': volumes,  # (B, 1, D, H, W)
        'boxes': boxes,      # List of (Ni, 6) tensors
        'labels': labels,    # List of (Ni,) tensors
        'study_ids': study_ids
    }