"""
Inference script for 3D Grounding-DETR.
Run predictions on CT volumes.
"""
import torch
import yaml
import argparse
import numpy as np
from pathlib import Path

from models import build_model
from datasets import RSNAVolumeDataset
from utils.visualization import visualize_single_slice, visualize_multi_slice, save_visualization


@torch.no_grad()
def run_inference(
    model,
    volume: torch.Tensor,
    device: torch.device,
    score_threshold: float = 0.5
):
    """
    Run inference on a single volume.
    
    Args:
        model: Trained model
        volume: (1, D, H, W) tensor
        device: Device to run on
        score_threshold: Confidence threshold
    
    Returns:
        Dict with predictions
    """
    model.eval()
    
    volume = volume.unsqueeze(0).to(device)  # (1, 1, D, H, W)
    
    outputs = model(volume)
    
    pred_logits = outputs['pred_logits'][0]  # (num_queries, num_classes+1)
    pred_boxes = outputs['pred_boxes'][0]  # (num_queries, 6)
    
    # Get scores and labels
    pred_scores = torch.softmax(pred_logits, dim=-1)
    pred_labels = pred_scores[:, :-1].max(dim=-1)[1]  # Exclude background
    pred_scores = pred_scores[:, :-1].max(dim=-1)[0]  # Max score (exclude background)
    
    # Filter by threshold
    keep = pred_scores > score_threshold
    
    pred_boxes = pred_boxes[keep].cpu().numpy()
    pred_scores = pred_scores[keep].cpu().numpy()
    pred_labels = pred_labels[keep].cpu().numpy()
    
    return {
        'boxes': pred_boxes,
        'scores': pred_scores,
        'labels': pred_labels
    }


def main():
    parser = argparse.ArgumentParser(description='Run inference with 3D Grounding-DETR')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/model_best.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='./datasets',
                       help='Path to dataset directory')
    parser.add_argument('--sample_idx', type=int, default=0,
                       help='Sample index to run inference on')
    parser.add_argument('--score_threshold', type=float, default=0.5,
                       help='Score threshold for detections')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualization')
    parser.add_argument('--output_dir', type=str, default='./inference_results',
                       help='Output directory for visualizations')
    parser.add_argument('--multi_view', action='store_true',
                       help='Create multi-slice visualization')
    parser.add_argument('--num_slices', type=int, default=9,
                       help='Number of slices for multi-view')
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint['config']
    
    # Build model
    print("Building model...")
    model = build_model(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Load dataset
    print(f"Loading dataset from {args.data_dir}")
    dataset = RSNAVolumeDataset(
        data_dir=args.data_dir,
        volume_size=tuple(config['data']['image_size']),
        train=False,
        augment=False
    )
    
    # Get sample
    sample = dataset[args.sample_idx]
    volume = sample['volume']
    study_id = sample['study_id']
    
    print(f"\nRunning inference on study {study_id}")
    print(f"Volume shape: {volume.shape}")
    
    # Run inference
    predictions = run_inference(model, volume, device, args.score_threshold)
    
    # Print results
    print(f"\nDetected {len(predictions['boxes'])} objects:")
    for i, (box, score, label) in enumerate(zip(
        predictions['boxes'],
        predictions['scores'],
        predictions['labels']
    )):
        print(f"\nObject {i+1}:")
        print(f"  Class: {label}")
        print(f"  Score: {score:.3f}")
        print(f"  Box (cx, cy, cz, w, h, d): {box}")
    
    # Ground truth
    print(f"\nGround truth:")
    gt_boxes = sample['boxes'].numpy()
    gt_labels = sample['labels'].numpy()
    print(f"  {len(gt_boxes)} objects")
    for i, (box, label) in enumerate(zip(gt_boxes, gt_labels)):
        print(f"\nObject {i+1}:")
        print(f"  Class: {label}")
        print(f"  Box: {box}")
    
    # Generate visualization if requested
    if args.visualize:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nGenerating visualization...")
        volume_np = volume[0].numpy()  # Remove channel dim
        
        pred_boxes_list = predictions['boxes'].tolist() if len(predictions['boxes']) > 0 else None
        pred_labels_list = predictions['labels'].tolist() if len(predictions['labels']) > 0 else None
        pred_scores_list = predictions['scores'].tolist() if len(predictions['scores']) > 0 else None
        
        if args.multi_view:
            fig = visualize_multi_slice(
                volume_np,
                pred_boxes=pred_boxes_list,
                pred_labels=pred_labels_list,
                pred_scores=pred_scores_list,
                gt_boxes=gt_boxes.tolist(),
                gt_labels=gt_labels.tolist(),
                num_slices=args.num_slices,
                score_threshold=args.score_threshold
            )
            output_path = output_dir / f"{study_id}_multi_slice.png"
        else:
            # Single slice (middle)
            slice_idx = volume_np.shape[0] // 2
            fig = visualize_single_slice(
                volume_np,
                slice_idx=slice_idx,
                pred_boxes=pred_boxes_list,
                pred_labels=pred_labels_list,
                pred_scores=pred_scores_list,
                gt_boxes=gt_boxes.tolist(),
                gt_labels=gt_labels.tolist(),
                score_threshold=args.score_threshold
            )
            output_path = output_dir / f"{study_id}_slice_{slice_idx}.png"
        
        save_visualization(fig, str(output_path))
        print(f"Visualization saved to {output_path}")


if __name__ == '__main__':
    main()
