"""
RSNA CT Volume Dataset for 3D object detection.
Loads NIfTI segmentations and DICOM/JPEG slices, converts to 3D bounding boxes.
"""
import os
import numpy as np
import nibabel as nib
from PIL import Image
from typing import Tuple, List, Optional, Dict
import torch
from torch.utils.data import Dataset
import pandas as pd

try:
    import pydicom
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False
    print("Warning: pydicom not installed. DCM file support disabled.")

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
        - DICOM or JPEG slice images (stacked into volumes)
        - Converts masks to 3D bounding boxes
    """
    
    def __init__(
        self,
        data_dir: str,
        segmentation_dir: str = "segmentations",
        image_dir: str = "train_images",
        target_width: int = 64,  # Target width (2nd dimension), other dimensions scale proportionally
        train: bool = True,
        augment: bool = False,
        num_samples: Optional[int] = None,
        image_format: str = "dcm",  # 'dcm' or 'jpeg'
    ):
        """
        Args:
            data_dir: Root directory containing datasets
            segmentation_dir: Directory with .nii segmentation files
            image_dir: Directory with DICOM/JPEG slices
            target_width: Target width (2nd dimension), all dimensions scale proportionally
            train: Training or validation mode
            augment: Apply data augmentation
            num_samples: Limit dataset size (for debugging)
            image_format: Image format to load ('dcm' or 'jpeg')
        """
        self.data_dir = data_dir
        self.segmentation_dir = os.path.join(data_dir, segmentation_dir)
        self.image_dir = os.path.join(data_dir, image_dir)
        self.target_width = target_width
        self.train = train
        self.augment = augment
        self.image_format = image_format.lower()
        
        # Validate image format
        if self.image_format not in ['jpeg', 'dcm']:
            raise ValueError(f"image_format must be 'dcm' or 'jpeg', got '{image_format}'")
        
        # Check pydicom availability for DCM format
        if self.image_format == 'dcm' and not PYDICOM_AVAILABLE:
            raise ImportError("pydicom is required for DCM format. Install with: pip install pydicom")
        
        # Find all segmentation files
        self.seg_files = []
        if os.path.exists(self.segmentation_dir):
            self.seg_files = sorted([
                f for f in os.listdir(self.segmentation_dir)
                if f.endswith('.nii') or f.endswith('.nii.gz')
            ])
        
        # Filter out segmentation files that don't have corresponding images
        valid_seg_files = []
        for seg_file in self.seg_files:
            study_id = seg_file.replace('.nii.gz', '').replace('.nii', '')
            if self._find_image_directory(study_id) is not None:
                valid_seg_files.append(seg_file)
            else:
                print(f"Excluding {study_id}: no corresponding image data found")
        
        self.seg_files = valid_seg_files
        
        if num_samples is not None:
            self.seg_files = self.seg_files[:num_samples]
        
        print(f"Found {len(self.seg_files)} segmentation files with corresponding images")
        print(f"Image format: {self.image_format}")
    
    def __len__(self) -> int:
        return len(self.seg_files)
    
    def _normalize_volume(self, volume: np.ndarray, format_used: str) -> np.ndarray:
        """
        Normalize volume intensity based on the image format.
        
        Args:
            volume: Image volume (D, H, W)
            format_used: The image format ('jpeg' or 'dcm')
            
        Returns:
            Normalized volume with values in [0, 1]
        """
        volume = volume.astype(np.float32)
        
        if format_used == 'jpeg':
            # JPEG images are in [0, 255], normalize to [0, 1]
            volume = volume / 255.0
        elif format_used == 'dcm':
            # DICOM images are in Hounsfield Units (HU)
            # Typical CT window for soft tissue: [-100, 300] HU
            # Clip to a reasonable range and normalize to [0, 1]
            HU_MIN = -1000  # Air
            HU_MAX = 1000   # Bone
            volume = np.clip(volume, HU_MIN, HU_MAX)
            volume = (volume - HU_MIN) / (HU_MAX - HU_MIN)
            
        return volume
    
    def load_nifti_volume(self, seg_file: str) -> Tuple[np.ndarray, np.ndarray, str]:
        """
        Load NIfTI segmentation and corresponding DICOM/JPEG images.
        
        Args:
            seg_file: Segmentation filename
        
        Returns:
            volume: Image volume (D, H, W)
            mask: Segmentation mask (D, H, W)
            format_used: The image format that was loaded ('jpeg' or 'dcm')
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
        
        # Find corresponding image directory and load volume
        volume, image_format_used = self._load_image_volume(study_id, seg_data.shape)
        
        return volume, seg_data, image_format_used
    
    def _find_image_directory(self, study_id: str) -> Optional[str]:
        """
        Find the image directory for a given study ID.
        
        Args:
            study_id: The study ID to search for
            
        Returns:
            Path to the image directory or None if not found
        """
        if not os.path.exists(self.image_dir):
            return None
            
        for patient_dir in os.listdir(self.image_dir):
            patient_path = os.path.join(self.image_dir, patient_dir)
            if os.path.isdir(patient_path):
                study_path = os.path.join(patient_path, study_id)
                if os.path.exists(study_path):
                    return study_path
        return None
    
    def _load_image_volume(self, study_id: str, target_shape: Tuple[int, int, int]) -> Tuple[np.ndarray, str]:
        """
        Load image volume from DICOM or JPEG files.
        
        Args:
            study_id: Study ID to load
            target_shape: Expected volume shape (D, H, W)
            
        Returns:
            volume: Image volume (D, H, W)
            format_used: The image format that was loaded ('jpeg' or 'dcm')
            
        Raises:
            FileNotFoundError: If images cannot be found or loaded
        """
        if self.image_format == 'jpeg':
            volume, format_used = self._try_load_jpeg(study_id, target_shape)
        elif self.image_format == 'dcm':
            volume, format_used = self._try_load_dcm(study_id, target_shape)
            
        return volume, format_used
    
    def _try_load_jpeg(self, study_id: str, target_shape: Tuple[int, int, int]) -> Tuple[np.ndarray, str]:
        """
        Try to load JPEG images for a study.
        
        Args:
            study_id: Study ID to load
            target_shape: Expected volume shape (D, H, W)
            
        Returns:
            volume: Image volume (D, H, W)
            format_used: 'jpeg'
            
        Raises:
            FileNotFoundError: If image directory or JPEG files not found
        """
        image_study_dir = self._find_image_directory(study_id)
        
        if image_study_dir is None:
            raise FileNotFoundError(f"No image directory found for study {study_id}")
        
        # Load JPEG slices
        jpeg_files = sorted([
            f for f in os.listdir(image_study_dir)
            if f.endswith('.jpeg') or f.endswith('.jpg')],
            key=lambda f: int(os.path.splitext(f)[0])
        )
        
        if len(jpeg_files) == 0:
            raise FileNotFoundError(f"No JPEG files found in {image_study_dir}")
        
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
        volume = np.stack(images, axis=0).astype(np.float32)  # (D, H, W)
        return volume, 'jpeg'
    
    def _try_load_dcm(self, study_id: str, target_shape: Tuple[int, int, int]) -> Tuple[np.ndarray, str]:
        """
        Try to load DICOM images for a study.
        
        Args:
            study_id: Study ID to load
            target_shape: Expected volume shape (D, H, W)
            
        Returns:
            volume: Image volume (D, H, W) with proper HU values
            format_used: 'dcm'
            
        Raises:
            FileNotFoundError: If image directory or DCM files not found
        """
        image_study_dir = self._find_image_directory(study_id)
        
        if image_study_dir is None:
            raise FileNotFoundError(f"No DCM images found for study {study_id}")
        
        # Load DICOM slices
        dcm_files = sorted([
            f for f in os.listdir(image_study_dir)
            if f.endswith('.dcm')],
            key=lambda f: int(os.path.splitext(f)[0])
        )
        
        if len(dcm_files) == 0:
            raise FileNotFoundError(f"No DCM files in {image_study_dir}")
        
        # Load DICOM images and extract pixel data with HU conversion
        slices = []
        for dcm_file in dcm_files:
            dcm_path = os.path.join(image_study_dir, dcm_file)
            ds = pydicom.dcmread(dcm_path)
            
            # Get pixel array
            pixel_array = ds.pixel_array.astype(np.float32)
            
            # Convert to Hounsfield Units (HU) if rescale parameters are available
            if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
                slope = float(ds.RescaleSlope)
                intercept = float(ds.RescaleIntercept)
                pixel_array = pixel_array * slope + intercept
            
            slices.append(pixel_array)
        
        # Stack into volume
        volume = np.stack(slices, axis=0).astype(np.float32)  # (D, H, W)
        return volume, 'dcm'
    
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
        volume, mask, format_used = self.load_nifti_volume(seg_file)
        
        # Normalize intensity based on image format
        volume = self._normalize_volume(volume, format_used)
        
        # Proportionally resize based on target_width (2nd dimension)
        # Original shape: (D, H, W) where W is the width (2nd dimension in NIfTI after transpose)
        original_width = volume.shape[1]  # Width is the 2nd dimension (H in D, H, W)
        scale_factor = self.target_width / original_width
        
        # Calculate target size maintaining aspect ratio
        target_size = (
            max(1, int(round(volume.shape[0] * scale_factor))),  # D
            self.target_width,  # H (target width)
            max(1, int(round(volume.shape[2] * scale_factor)))   # W
        )
        
        if volume.shape != target_size:
            volume = resize_volume(volume, target_size, order=1)
            mask = resize_volume(mask, target_size, order=0)  # Nearest for labels
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
            # Convert labels from 1-5 to 0-4 for model compatibility
            labels = np.array([b['label'] - 1 for b in boxes_data], dtype=np.int64)
        
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
    Custom collate function for batching with variable-sized volumes.
    
    Since volumes may have different sizes due to proportional scaling,
    we pad them to the maximum size in the batch.
    Boxes are kept as a list since each sample has different numbers of boxes.
    """
    # Find maximum dimensions in the batch
    max_d = max(item['volume'].shape[1] for item in batch)
    max_h = max(item['volume'].shape[2] for item in batch)
    max_w = max(item['volume'].shape[3] for item in batch)
    
    # Pad volumes to the same size
    padded_volumes = []
    masks = []  # Mask indicating valid (non-padded) regions
    
    for item in batch:
        vol = item['volume']  # (1, D, H, W)
        d, h, w = vol.shape[1], vol.shape[2], vol.shape[3]
        
        # Create padded volume (zero-padding)
        padded = torch.zeros(1, max_d, max_h, max_w, dtype=vol.dtype)
        padded[:, :d, :h, :w] = vol
        padded_volumes.append(padded)
        
        # Create mask (1 for valid, 0 for padded)
        mask = torch.zeros(max_d, max_h, max_w, dtype=torch.bool)
        mask[:d, :h, :w] = True
        masks.append(mask)
    
    volumes = torch.stack(padded_volumes, dim=0)  # (B, 1, max_D, max_H, max_W)
    masks = torch.stack(masks, dim=0)  # (B, max_D, max_H, max_W)
    
    boxes = [item['boxes'] for item in batch]
    labels = [item['labels'] for item in batch]
    study_ids = [item['study_id'] for item in batch]
    
    # Store original sizes for reference
    original_sizes = [(item['volume'].shape[1], item['volume'].shape[2], item['volume'].shape[3]) 
                      for item in batch]
    
    return {
        'volumes': volumes,           # (B, 1, max_D, max_H, max_W)
        'masks': masks,               # (B, max_D, max_H, max_W) - True for valid regions
        'boxes': boxes,               # List of (Ni, 6) tensors
        'labels': labels,             # List of (Ni,) tensors
        'study_ids': study_ids,
        'original_sizes': original_sizes  # List of (D, H, W) tuples
    }
