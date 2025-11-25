"""
Standalone script for visualizing model predictions.
Loads a trained model, runs inference, and generates visualizations.
"""
import torch
import yaml
import argparse
import numpy as np
from pathlib import Path

from datasets import RSNAVolumeDataset
from models import build_model
from utils.visualization import (
    visualize_single_slice,
    visualize_multi_slice,
    save_visualization
)


def load_model_and_config(checkpoint_path: str, device: str = 'cpu'):
    """Load trained model and configuration from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    # Build model
    model = build_model(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, config


def run_inference(
    model,
    volume: torch.Tensor,
    device: str = 'cpu'
):
    """Run inference on a single volume."""
    with torch.no_grad():
        volume = volume.unsqueeze(0).to(device).float()  # (1, 1, D, H, W)
        outputs = model(volume)
        
        pred_logits = outputs['pred_logits'][0]  # (num_queries, num_classes+1)
        pred_boxes = outputs['pred_boxes'][0]    # (num_queries, 6)
        
        # Get class predictions and scores
        pred_scores = pred_logits.softmax(-1)
        pred_labels = pred_scores[:, :-1].argmax(-1)  # Exclude background class
        pred_scores = pred_scores.max(-1)[0]
        
    return pred_boxes.cpu().numpy(), pred_labels.cpu().numpy(), pred_scores.cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description='Visualize 3D Grounding-DETR predictions')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/model_best.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='./datasets',
                       help='Path to dataset directory')
    parser.add_argument('--sample_idx', type=int, default=0,
                       help='Sample index to visualize')
    parser.add_argument('--output_dir', type=str, default='./visualizations',
                       help='Output directory for visualizations')
    parser.add_argument('--score_threshold', type=float, default=0.5,
                       help='Minimum confidence score to display')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    parser.add_argument('--show_gt', action='store_true',
                       help='Show ground truth boxes')
    parser.add_argument('--multi_view', action='store_true',
                       help='Create multi-slice visualization')
    parser.add_argument('--num_slices', type=int, default=9,
                       help='Number of slices for multi-view')
    parser.add_argument('--slice_idx', type=int, default=None,
                       help='Specific slice index to visualize (for single-slice view)')
    parser.add_argument('--axis', type=str, default='axial', choices=['axial', 'sagittal', 'coronal'],
                       help='View axis')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("3D Grounding-DETR Visualization")
    print("="*60)
    
    # Load model
    print(f"\nLoading model from {args.checkpoint}...")
    device = torch.device(args.device)
    model, config = load_model_and_config(args.checkpoint, device)
    print(f"Model loaded successfully on {device}")
    
    # Load dataset
    print(f"\nLoading dataset from {args.data_dir}...")
    dataset = RSNAVolumeDataset(
        data_dir=args.data_dir,
        volume_size=tuple(config['data']['image_size']),
        train=False,
        augment=False
    )
    
    if args.sample_idx >= len(dataset):
        print(f"Error: Sample index {args.sample_idx} out of range (dataset size: {len(dataset)})")
        return
    
    # Get sample
    sample = dataset[args.sample_idx]
    volume = sample['volume'][0]  # Remove channel dimension (1, D, H, W) -> (D, H, W)
    volume_tensor = sample['volume']
    study_id = sample['study_id']
    
    print(f"\nProcessing sample {args.sample_idx} (Study ID: {study_id})")
    print(f"Volume shape: {volume.shape}")
    
    # Run inference
    print("\nRunning inference...")
    pred_boxes, pred_labels, pred_scores = run_inference(model, volume_tensor, device)
    
    # Filter by score threshold
    valid_mask = pred_scores >= args.score_threshold
    pred_boxes_filtered = pred_boxes[valid_mask]
    pred_labels_filtered = pred_labels[valid_mask]
    pred_scores_filtered = pred_scores[valid_mask]
    
    print(f"Found {len(pred_boxes_filtered)} predictions above threshold {args.score_threshold}")
    
    # Print predictions
    if len(pred_boxes_filtered) > 0:
        print("\nPredictions:")
        for i, (box, label, score) in enumerate(zip(pred_boxes_filtered, pred_labels_filtered, pred_scores_filtered)):
            print(f"  {i+1}. Class {label}, Score: {score:.3f}, Box: {box}")
    
    # Get ground truth if requested
    gt_boxes = None
    gt_labels = None
    if args.show_gt:
        gt_boxes = sample['boxes'].numpy()
        gt_labels = sample['labels'].numpy()
        print(f"\nGround truth: {len(gt_boxes)} objects")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    
    if args.multi_view:
        # Multi-slice view
        print(f"Creating multi-slice view with {args.num_slices} slices...")
        fig = visualize_multi_slice(
            volume.numpy(),
            pred_boxes=pred_boxes_filtered.tolist() if len(pred_boxes_filtered) > 0 else None,
            pred_labels=pred_labels_filtered.tolist() if len(pred_labels_filtered) > 0 else None,
            pred_scores=pred_scores_filtered.tolist() if len(pred_scores_filtered) > 0 else None,
            gt_boxes=gt_boxes.tolist() if gt_boxes is not None else None,
            gt_labels=gt_labels.tolist() if gt_labels is not None else None,
            num_slices=args.num_slices,
            axis=args.axis,
            score_threshold=args.score_threshold
        )
        
        output_path = output_dir / f"{study_id}_multi_slice_{args.axis}.png"
        save_visualization(fig, str(output_path))
    
    else:
        # Single slice view
        if args.slice_idx is None:
            # Use middle slice by default
            if args.axis == 'axial':
                slice_idx = volume.shape[0] // 2
            elif args.axis == 'sagittal':
                slice_idx = volume.shape[2] // 2
            else:  # coronal
                slice_idx = volume.shape[1] // 2
        else:
            slice_idx = args.slice_idx
        
        print(f"Creating single-slice view (slice {slice_idx}, {args.axis})...")
        fig = visualize_single_slice(
            volume.numpy(),
            slice_idx=slice_idx,
            pred_boxes=pred_boxes_filtered.tolist() if len(pred_boxes_filtered) > 0 else None,
            pred_labels=pred_labels_filtered.tolist() if len(pred_labels_filtered) > 0 else None,
            pred_scores=pred_scores_filtered.tolist() if len(pred_scores_filtered) > 0 else None,
            gt_boxes=gt_boxes.tolist() if gt_boxes is not None else None,
            gt_labels=gt_labels.tolist() if gt_labels is not None else None,
            axis=args.axis,
            score_threshold=args.score_threshold
        )
        
        output_path = output_dir / f"{study_id}_slice_{slice_idx}_{args.axis}.png"
        save_visualization(fig, str(output_path))
    
    print("\n" + "="*60)
    print("Visualization complete!")
    print("="*60)


if __name__ == '__main__':
    main()
