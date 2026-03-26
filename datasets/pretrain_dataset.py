"""
RSNA CT Volume Dataset for Swin3D backbone pretraining.
Loads DICOM CT volumes with patient-level multi-label organ injury labels.
Uses all ~2,873 patients (not just the 193 with segmentation masks).
"""
import os
import csv
import numpy as np
from typing import Tuple, List, Optional, Dict
import torch
from torch.utils.data import Dataset
from scipy import ndimage

try:
    import pydicom
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False
    print("Warning: pydicom not installed. DCM file support disabled.")


# Label columns in train_2024.csv (14 binary labels)
LABEL_COLUMNS = [
    'bowel_healthy', 'bowel_injury',
    'extravasation_healthy', 'extravasation_injury',
    'kidney_healthy', 'kidney_low', 'kidney_high',
    'liver_healthy', 'liver_low', 'liver_high',
    'spleen_healthy', 'spleen_low', 'spleen_high',
    'any_injury',
]

# Organ-level injury grades (5 organs, mapped to severity 0/1/2)
# This is an alternative simpler label scheme
ORGAN_NAMES = ['bowel', 'extravasation', 'kidney', 'liver', 'spleen']


class RSNAPretrainDataset(Dataset):
    """
    PyTorch Dataset for Swin3D pretraining using patient-level labels.
    
    Loads full CT volumes from DICOM files and provides multi-label
    classification targets from train_2024.csv.
    
    Label scheme: 14-dim binary vector (one-hot per organ status).
    """
    
    def __init__(
        self,
        data_dir: str,
        csv_file: str = "train_2024.csv",
        image_dir: str = "train_images",
        target_size: Tuple[int, int, int] = (128, 128, 128),
        augment: bool = False,
        num_samples: Optional[int] = None,
        series_selection: str = "first",
    ):
        """
        Args:
            data_dir: Root directory containing the RSNA dataset.
            csv_file: CSV file with patient-level labels.
            image_dir: Directory with DICOM slices (patient_id/series_id/*.dcm).
            target_size: Fixed output volume size (D, H, W).
            augment: Whether to apply data augmentation.
            num_samples: Limit dataset size (for debugging).
            series_selection: How to pick series when patient has multiple.
                'first' = use alphabetically first series.
        """
        self.data_dir = data_dir
        self.image_dir = os.path.join(data_dir, image_dir)
        self.target_size = target_size
        self.augment = augment
        self.series_selection = series_selection
        
        if not PYDICOM_AVAILABLE:
            raise ImportError("pydicom is required. Install with: pip install pydicom")
        
        # Load labels from CSV
        self.labels_dict = self._load_labels(os.path.join(data_dir, csv_file))
        
        # Find patients that have both labels and images
        self.samples = self._build_sample_list()
        
        if num_samples is not None:
            self.samples = self.samples[:num_samples]
        
        print(f"RSNAPretrainDataset: {len(self.samples)} samples "
              f"(target_size={target_size}, augment={augment})")
    
    def _load_labels(self, csv_path: str) -> Dict[str, np.ndarray]:
        """Load patient-level labels from CSV."""
        labels = {}
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                patient_id = row['patient_id'].strip()
                label_vec = np.array(
                    [int(row[col].strip()) for col in LABEL_COLUMNS],
                    dtype=np.float32
                )
                labels[patient_id] = label_vec
        print(f"Loaded labels for {len(labels)} patients")
        return labels
    
    def _build_sample_list(self) -> List[Dict]:
        """Build list of (patient_id, series_path) with valid labels and images."""
        samples = []
        
        if not os.path.exists(self.image_dir):
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        
        image_patients = sorted(os.listdir(self.image_dir))
        
        for patient_id in image_patients:
            if patient_id not in self.labels_dict:
                continue
            
            patient_path = os.path.join(self.image_dir, patient_id)
            if not os.path.isdir(patient_path):
                continue
            
            # Get series directories
            series_dirs = sorted([
                d for d in os.listdir(patient_path)
                if os.path.isdir(os.path.join(patient_path, d))
            ])
            
            if len(series_dirs) == 0:
                continue
            
            # Select series
            if self.series_selection == "first":
                series_id = series_dirs[0]
            else:
                series_id = series_dirs[0]
            
            series_path = os.path.join(patient_path, series_id)
            
            # Verify it has DICOM files
            dcm_files = [f for f in os.listdir(series_path) if f.endswith('.dcm')]
            if len(dcm_files) < 10:  # Skip very short series
                continue
            
            samples.append({
                'patient_id': patient_id,
                'series_path': series_path,
                'labels': self.labels_dict[patient_id],
            })
        
        print(f"Found {len(samples)} patients with both images and labels")
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def _load_dicom_volume(self, series_path: str) -> np.ndarray:
        """
        Load DICOM series into a 3D volume with HU values.
        
        Returns:
            volume: (D, H, W) float32 array in Hounsfield Units
        """
        dcm_files = sorted(
            [f for f in os.listdir(series_path) if f.endswith('.dcm')],
            key=lambda f: int(os.path.splitext(f)[0])
        )
        
        slices = []
        for dcm_file in dcm_files:
            dcm_path = os.path.join(series_path, dcm_file)
            ds = pydicom.dcmread(dcm_path)
            pixel_array = ds.pixel_array.astype(np.float32)
            
            # Convert to Hounsfield Units
            if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
                slope = float(ds.RescaleSlope)
                intercept = float(ds.RescaleIntercept)
                pixel_array = pixel_array * slope + intercept
            
            slices.append(pixel_array)
        
        volume = np.stack(slices, axis=0).astype(np.float32)  # (D, H, W)
        return volume
    
    def _normalize_volume(self, volume: np.ndarray) -> np.ndarray:
        """
        Normalize HU volume to [0, 1] using abdominal soft tissue window.
        Matches the windowing in RSNAVolumeDataset.
        """
        HU_MIN = -160
        HU_MAX = 240
        volume = np.clip(volume, HU_MIN, HU_MAX)
        volume = (volume - HU_MIN) / (HU_MAX - HU_MIN)
        return volume.astype(np.float32)
    
    def _resize_volume(self, volume: np.ndarray, target_size: Tuple[int, int, int]) -> np.ndarray:
        """Resize volume to fixed target size."""
        if volume.shape == target_size:
            return volume
        zoom_factors = np.array(target_size) / np.array(volume.shape)
        resized = ndimage.zoom(volume, zoom_factors, order=1)
        return resized.astype(np.float32)
    
    def _augment_volume(self, volume: np.ndarray) -> np.ndarray:
        """Apply data augmentation to pretrain volume (no mask needed)."""
        # Random rotation in XY plane (±15 degrees, milder than detection)
        if np.random.rand() < 0.5:
            angle = np.random.uniform(-15, 15)
            volume = ndimage.rotate(volume, angle, axes=(1, 2), reshape=False,
                                    order=1, mode='constant', cval=0)
        
        # Random scaling (crop/pad back to original size)
        if np.random.rand() < 0.5:
            scale = np.random.uniform(0.9, 1.1)
            original_shape = np.array(volume.shape)
            scaled = ndimage.zoom(volume, scale, order=1)
            actual_shape = np.array(scaled.shape)
            
            result = np.zeros(original_shape, dtype=volume.dtype)
            if scale > 1.0:
                start = (actual_shape - original_shape) // 2
                end = start + original_shape
                result = scaled[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
            else:
                start = (original_shape - actual_shape) // 2
                end = start + actual_shape
                result[start[0]:end[0], start[1]:end[1], start[2]:end[2]] = scaled
            volume = result
        
        # Intensity jittering
        if np.random.rand() < 0.5:
            noise = np.random.normal(0, 0.05, volume.shape).astype(np.float32)
            volume = np.clip(volume + noise, 0, 1)
        
        # Random depth crop & resize (simulate different scan ranges)
        if np.random.rand() < 0.3:
            D = volume.shape[0]
            crop_ratio = np.random.uniform(0.7, 1.0)
            crop_d = max(16, int(D * crop_ratio))
            start_d = np.random.randint(0, max(1, D - crop_d))
            cropped = volume[start_d:start_d + crop_d]
            # Resize back to original depth
            zoom_d = D / crop_d
            volume = ndimage.zoom(cropped, (zoom_d, 1.0, 1.0), order=1).astype(np.float32)
            # Ensure exact shape match
            if volume.shape[0] != D:
                volume = self._resize_volume(volume, (D, volume.shape[1], volume.shape[2]))
        
        return volume.astype(np.float32)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single sample.
        
        Returns:
            dict with:
                - 'volume': Tensor (1, D, H, W)
                - 'labels': Tensor (14,) - multi-label binary vector
                - 'patient_id': str
        """
        sample = self.samples[idx]
        
        # Load DICOM volume
        volume = self._load_dicom_volume(sample['series_path'])
        
        # Normalize HU to [0, 1]
        volume = self._normalize_volume(volume)
        
        # Resize to fixed target size
        volume = self._resize_volume(volume, self.target_size)
        
        # Augmentation
        if self.augment:
            volume = self._augment_volume(volume)
        
        # Convert to tensor
        volume_tensor = torch.from_numpy(volume).unsqueeze(0)  # (1, D, H, W)
        labels_tensor = torch.from_numpy(sample['labels'].copy())  # (14,)
        
        return {
            'volume': volume_tensor,
            'labels': labels_tensor,
            'patient_id': sample['patient_id'],
        }


def pretrain_collate_fn(batch: List[Dict]) -> Dict:
    """
    Collate function for pretraining.
    All volumes have the same fixed size, so simple stacking works.
    """
    volumes = torch.stack([item['volume'] for item in batch], dim=0)  # (B, 1, D, H, W)
    labels = torch.stack([item['labels'] for item in batch], dim=0)   # (B, 14)
    patient_ids = [item['patient_id'] for item in batch]
    
    return {
        'volumes': volumes,
        'labels': labels,
        'patient_ids': patient_ids,
    }
