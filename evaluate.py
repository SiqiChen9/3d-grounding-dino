"""
Evaluation script for 3D Grounding-DETR.
Compute metrics on validation set.
"""
import torch
import yaml
import argparse
import json
from pathlib import Path
from tqdm import tqdm

from models import build_model
from datasets import RSNAVolumeDataset, collate_fn
from utils.metrics import compute_map
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate_model(
    model,
    data_loader,
    device,
    num_classes,
    score_threshold=0.5
):
    """
    Evaluate model on dataset.
    
    Returns:
        predictions and ground_truths for mAP computation
    """
    model.eval()
    
    all_predictions = []
    all_ground_truths = []
    
    for batch in tqdm(data_loader, desc="Evaluating"):
        volumes = batch['volumes'].to(device)
        gt_labels = batch['labels']
        gt_boxes = batch['boxes']
        
        # Forward pass
        outputs = model(volumes)
        
        batch_size = volumes.shape[0]
        
        for i in range(batch_size):
            # Get predictions for this sample
            pred_logits = outputs['pred_logits'][i]  # (num_queries, num_classes+1)
            pred_boxes = outputs['pred_boxes'][i]  # (num_queries, 6)
            
            # Get scores and labels
            pred_scores = torch.softmax(pred_logits, dim=-1)
            pred_labels = pred_scores[:, :-1].max(dim=-1)[1]
            pred_scores = pred_scores[:, :-1].max(dim=-1)[0]
            
            # Filter by threshold
            keep = pred_scores > score_threshold
            
            all_predictions.append({
                'boxes': pred_boxes[keep].cpu().numpy(),
                'scores': pred_scores[keep].cpu().numpy(),
                'labels': pred_labels[keep].cpu().numpy()
            })
            
            # Ground truth
            all_ground_truths.append({
                'boxes': gt_boxes[i].cpu().numpy(),
                'labels': gt_labels[i].cpu().numpy()
            })
    
    return all_predictions, all_ground_truths


def main():
    parser = argparse.ArgumentParser(description='Evaluate 3D Grounding-DETR')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/model_best.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='./datasets',
                       help='Path to dataset directory')
    parser.add_argument('--batch_size', type=int, default=2,
                       help='Batch size for evaluation')
    parser.add_argument('--score_threshold', type=float, default=0.5,
                       help='Score threshold for detections')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    parser.add_argument('--output', type=str, default='evaluation_results.json',
                       help='Output JSON file for results')
    
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
    
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )
    
    # Evaluate
    print("\nRunning evaluation...")
    predictions, ground_truths = evaluate_model(
        model, data_loader, device,
        config['model']['num_classes'],
        args.score_threshold
    )
    
    # Compute metrics
    print("\nComputing metrics...")
    metrics = compute_map(
        predictions,
        ground_truths,
        num_classes=config['model']['num_classes'],
        iou_thresholds=[0.1, 0.3, 0.5]
    )
    
    # Print results
    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")
    print("="*60)
    
    # Save results
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
