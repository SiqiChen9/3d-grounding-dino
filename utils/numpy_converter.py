"""
NumPy Volume Converter for RSNA Dataset.

Converts DICOM images and NIfTI segmentations to compressed NumPy format (.npz)
for faster I/O during training.

Usage:
    # Convert entire dataset
    python -m utils.numpy_converter --config configs/default_config.yaml
"""

import os
import yaml
import numpy as np
import nibabel as nib
from typing import Optional, Tuple, Dict
from tqdm import tqdm

try:
    import pydicom
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def find_study_info(image_dir: str, study_id: str) -> Optional[Tuple[str, str]]:
    """
    Find the study path and patient ID for a given study ID.
    
    Args:
        image_dir: Root directory containing patient folders
        study_id: The study ID to search for
        
    Returns:
        Tuple of (study_path, patient_id) or None if not found
    """
    if not os.path.exists(image_dir):
        return None
        
    for patient_id in os.listdir(image_dir):
        patient_path = os.path.join(image_dir, patient_id)
        if os.path.isdir(patient_path):
            study_path = os.path.join(patient_path, study_id)
            if os.path.exists(study_path):
                return study_path, patient_id
    return None


def load_dicom_volume(study_path: str) -> np.ndarray:
    """
    Load DICOM volume from a study directory.
    
    Args:
        study_path: Path to directory containing DICOM files
        
    Returns:
        volume: Image volume (D, H, W) as float32 with HU values
    """
    if not PYDICOM_AVAILABLE:
        raise ImportError("pydicom is required. Install with: pip install pydicom")
    
    dcm_files = sorted(
        [f for f in os.listdir(study_path) if f.endswith('.dcm')],
        key=lambda f: int(os.path.splitext(f)[0])
    )
    
    if len(dcm_files) == 0:
        raise FileNotFoundError(f"No DICOM files found in {study_path}")
    
    slices = []
    for dcm_file in dcm_files:
        dcm_path = os.path.join(study_path, dcm_file)
        ds = pydicom.dcmread(dcm_path)
        pixel_array = ds.pixel_array.astype(np.float32)
        
        # Convert to Hounsfield Units (HU)
        if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
            pixel_array = pixel_array * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
        
        slices.append(pixel_array)
    
    return np.stack(slices, axis=0).astype(np.float32)


def load_nifti_segmentation(seg_path: str) -> np.ndarray:
    """
    Load NIfTI segmentation file (without transformations).
    
    Args:
        seg_path: Path to .nii or .nii.gz file
        
    Returns:
        mask: Segmentation mask as uint8 in original orientation
    """
    seg_nib = nib.load(seg_path)
    return seg_nib.get_fdata().astype(np.uint8)


def convert_dataset(
    config_path: str = 'configs/default_config.yaml',
    output_subdir: str = 'numpy_volumes',
    force_reconvert: bool = False,
) -> Dict[str, str]:
    """
    Convert entire dataset from DICOM + NIfTI to NumPy format.
    
    Args:
        config_path: Path to configuration file
        output_subdir: Subdirectory name for output files
        force_reconvert: If True, reconvert even if .npz already exists
        
    Returns:
        Dictionary mapping study_id to output path
    """
    config = load_config(config_path)
    data_dir = config['data']['dataset_path']
    
    seg_dir = os.path.join(data_dir, 'segmentations')
    img_dir = os.path.join(data_dir, 'train_images')
    output_dir = os.path.join(data_dir, output_subdir)
    
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    
    # Find all segmentation files and their corresponding image paths
    seg_files = sorted([
        f for f in os.listdir(seg_dir)
        if f.endswith('.nii') or f.endswith('.nii.gz')
    ])
    
    print(f"Found {len(seg_files)} segmentation files")
    
    # Build list of valid studies with all needed info
    valid_studies = []
    for seg_file in seg_files:
        study_id = seg_file.replace('.nii.gz', '').replace('.nii', '')
        study_info = find_study_info(img_dir, study_id)
        if study_info:
            study_path, patient_id = study_info
            seg_path = os.path.join(seg_dir, seg_file)
            valid_studies.append({
                'study_id': study_id,
                'patient_id': patient_id,
                'study_path': study_path,
                'seg_path': seg_path,
            })
    
    print(f"Found {len(valid_studies)} studies with corresponding images")
    
    # Convert each study
    results = {}
    os.makedirs(output_dir, exist_ok=True)
    
    for study in tqdm(valid_studies, desc="Converting"):
        study_id = study['study_id']
        output_path = os.path.join(output_dir, f"{study_id}.npz")
        
        # Skip if already exists
        if os.path.exists(output_path) and not force_reconvert:
            results[study_id] = output_path
            continue
        
        try:
            volume = load_dicom_volume(study['study_path'])
            mask = load_nifti_segmentation(study['seg_path'])
            
            np.savez_compressed(
                output_path,
                volume=volume,
                mask=mask,
                study_id=study_id,
                patient_id=study['patient_id'],
            )
            results[study_id] = output_path
            
        except Exception as e:
            print(f"Error converting {study_id}: {e}")
    
    # Summary
    total_size = sum(
        os.path.getsize(p) for p in results.values() if os.path.exists(p)
    )
    print(f"\nConverted {len(results)} studies")
    print(f"Total size: {total_size / (1024**3):.2f} GB")
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert RSNA dataset to NumPy format')
    parser.add_argument('--config', type=str, default='configs/default_config.yaml')
    parser.add_argument('--force', action='store_true', help='Force reconvert')
    
    args = parser.parse_args()
    convert_dataset(config_path=args.config, force_reconvert=args.force)
