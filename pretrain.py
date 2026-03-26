"""
Pretraining script for Swin3D backbone.
Multi-label organ injury classification on the full RSNA dataset (~2,873 patients).
Pretrained backbone weights can then be loaded into GroundingDETR3D for detection.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import yaml
import argparse
import os
import numpy as np
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score

from datasets.pretrain_dataset import RSNAPretrainDataset, pretrain_collate_fn, LABEL_COLUMNS
from models.swin3d_classifier import build_pretrain_model
from utils.logger import TrainingLogger


def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_dataloaders(config: dict, logger: TrainingLogger):
    """Create train and validation dataloaders for pretraining."""
    data_cfg = config['data']
    random_seed = data_cfg.get('random_seed', 42)
    
    target_size = tuple(data_cfg.get('target_size', [128, 128, 128]))
    
    # Create full dataset
    full_dataset = RSNAPretrainDataset(
        data_dir=data_cfg['dataset_path'],
        csv_file=data_cfg.get('csv_file', 'train_2024.csv'),
        image_dir=data_cfg.get('image_dir', 'train_images'),
        target_size=target_size,
        augment=data_cfg.get('augment', True),
        num_samples=data_cfg.get('num_samples', None),
        series_selection=data_cfg.get('series_selection', 'first'),
    )
    
    # Split into train/val
    train_ratio = data_cfg.get('train_split', 0.85)
    total_size = len(full_dataset)
    train_size = int(total_size * train_ratio)
    val_size = total_size - train_size
    
    logger.info(f"Pretrain dataset split: train={train_size}, val={val_size} (total={total_size})")
    
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(random_seed)
    )
    
    num_workers = data_cfg.get('num_workers', 4)
    persistent_workers = data_cfg.get('persistent_workers', False) and num_workers > 0
    prefetch_factor = data_cfg.get('prefetch_factor', 2) if num_workers > 0 else None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=data_cfg['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        collate_fn=pretrain_collate_fn,
        pin_memory=True,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=data_cfg['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        collate_fn=pretrain_collate_fn,
        pin_memory=True,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )
    
    return train_loader, val_loader


def compute_class_weights(dataset, device):
    """
    Compute positive class weights for BCE loss to handle class imbalance.
    pos_weight = num_negatives / num_positives for each label.
    """
    all_labels = []
    for i in range(len(dataset)):
        sample = dataset.dataset.samples[dataset.indices[i]] if hasattr(dataset, 'indices') else dataset.samples[i]
        all_labels.append(sample['labels'])
    
    labels_array = np.stack(all_labels, axis=0)  # (N, 14)
    pos_counts = labels_array.sum(axis=0)  # (14,)
    neg_counts = len(labels_array) - pos_counts
    
    # Avoid division by zero
    pos_weights = np.where(pos_counts > 0, neg_counts / pos_counts, 1.0)
    
    # Cap weights to avoid extreme values
    pos_weights = np.clip(pos_weights, 0.1, 50.0)
    
    return torch.tensor(pos_weights, dtype=torch.float32).to(device)


def train_one_epoch(model, criterion, data_loader, optimizer, device, epoch, num_epochs, config):
    """Train for one epoch."""
    model.train()
    
    total_loss = 0
    all_logits = []
    all_labels = []
    
    pbar = tqdm(data_loader, desc=f"Pretrain Epoch {epoch+1}/{num_epochs}")
    
    for batch_idx, batch in enumerate(pbar):
        volumes = batch['volumes'].to(device).float()
        labels = batch['labels'].to(device).float()
        
        # Forward pass
        outputs = model(volumes)
        logits = outputs['logits']
        
        # Loss
        loss = criterion(logits, labels)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        clip_norm = config['training'].get('clip_max_norm', 0)
        if clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        
        optimizer.step()
        
        total_loss += loss.item()
        
        # Collect predictions for metrics
        with torch.no_grad():
            all_logits.append(logits.detach().cpu())
            all_labels.append(labels.detach().cpu())
        
        pbar.set_postfix({'loss': loss.item()})
    
    # Compute epoch metrics
    num_batches = len(data_loader)
    avg_loss = total_loss / num_batches
    
    all_logits = torch.cat(all_logits, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    probs = 1.0 / (1.0 + np.exp(-all_logits))  # sigmoid
    
    metrics = {'loss_total': avg_loss}
    metrics.update(_compute_classification_metrics(all_labels, probs))
    
    return avg_loss, metrics


@torch.no_grad()
def validate(model, criterion, data_loader, device):
    """Validate model."""
    model.eval()
    
    total_loss = 0
    all_logits = []
    all_labels = []
    
    for batch in tqdm(data_loader, desc="Validation"):
        volumes = batch['volumes'].to(device).float()
        labels = batch['labels'].to(device).float()
        
        outputs = model(volumes)
        logits = outputs['logits']
        
        loss = criterion(logits, labels)
        total_loss += loss.item()
        
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())
    
    num_batches = len(data_loader)
    avg_loss = total_loss / num_batches
    
    all_logits = torch.cat(all_logits, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    probs = 1.0 / (1.0 + np.exp(-all_logits))
    
    metrics = {'loss_total': avg_loss}
    metrics.update(_compute_classification_metrics(all_labels, probs))
    
    return avg_loss, metrics


def _compute_classification_metrics(labels: np.ndarray, probs: np.ndarray) -> dict:
    """
    Compute classification metrics for multi-label task.
    
    Args:
        labels: (N, 14) ground truth binary labels
        probs: (N, 14) predicted probabilities
    
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Per-label accuracy at threshold 0.5
    preds = (probs > 0.5).astype(float)
    accuracy = (preds == labels).mean()
    metrics['accuracy'] = accuracy
    
    # Mean AUC-ROC (skip labels with only one class present)
    aucs = []
    for i in range(labels.shape[1]):
        if len(np.unique(labels[:, i])) > 1:
            try:
                auc = roc_auc_score(labels[:, i], probs[:, i])
                aucs.append(auc)
            except ValueError:
                pass
    if aucs:
        metrics['mean_auroc'] = np.mean(aucs)
    
    # Mean Average Precision
    aps = []
    for i in range(labels.shape[1]):
        if labels[:, i].sum() > 0:
            try:
                ap = average_precision_score(labels[:, i], probs[:, i])
                aps.append(ap)
            except ValueError:
                pass
    if aps:
        metrics['mean_ap'] = np.mean(aps)
    
    # Per-organ accuracy (5 organs)
    organ_groups = {
        'bowel': [0, 1],       # bowel_healthy, bowel_injury
        'extravasation': [2, 3],  # extravasation_healthy, extravasation_injury
        'kidney': [4, 5, 6],   # kidney_healthy, kidney_low, kidney_high
        'liver': [7, 8, 9],    # liver_healthy, liver_low, liver_high
        'spleen': [10, 11, 12],  # spleen_healthy, spleen_low, spleen_high
    }
    for organ, indices in organ_groups.items():
        organ_acc = (preds[:, indices] == labels[:, indices]).mean()
        metrics[f'acc_{organ}'] = organ_acc
    
    return metrics


def save_checkpoint(model, optimizer, scheduler, epoch, config, filename, logger=None, is_best=False):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'backbone_state_dict': model.get_backbone_state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'config': config,
    }
    
    checkpoint_dir = Path(config['paths']['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    filepath = checkpoint_dir / filename
    torch.save(checkpoint, filepath)
    
    if logger:
        checkpoint_type = "BEST" if is_best else "CHECKPOINT"
        logger.info(f"{checkpoint_type} saved at epoch {epoch+1}: {filepath}")
    else:
        print(f"Checkpoint saved: {filepath}")


def main():
    parser = argparse.ArgumentParser(description='Pretrain Swin3D backbone')
    parser.add_argument('--config', type=str, default='configs/pretrain_config.yaml',
                        help='Path to pretrain config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, 
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    parser.add_argument('--debug', action='store_true',
                        help='Debug mode (single batch)')
    parser.add_argument('--run-name', type=str, default=None,
                        help='Name for this pretraining run')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Initialize logger
    logger = TrainingLogger(
        log_dir=config['paths']['log_dir'],
        config=config,
        run_name=args.run_name,
        config_path=args.config,
    )
    
    logger.info("=" * 60)
    logger.info("Swin3D Backbone Pretraining")
    logger.info("=" * 60)
    logger.info(f"Config loaded from {args.config}")
    for line in yaml.dump(config, default_flow_style=False).split('\n'):
        if line:
            logger.info(f"  {line}")
    
    # Device
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(config, logger)
    logger.log_dataset_info(train_loader, val_loader)
    
    # Create model
    logger.info("Building pretrain model...")
    model = build_pretrain_model(config)
    model = model.to(device)
    logger.log_model_info(model)
    
    # Loss function with class weights
    logger.info("Computing class weights for imbalanced labels...")
    pos_weight = compute_class_weights(train_loader.dataset, device)
    logger.info(f"Positive weights: {pos_weight.cpu().numpy().round(2).tolist()}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Optimizer and scheduler
    train_cfg = config['training']
    optimizer = AdamW(
        model.parameters(),
        lr=train_cfg['lr'],
        weight_decay=train_cfg['weight_decay'],
    )
    
    warmup_epochs = train_cfg.get('warmup_epochs', 5)
    total_epochs = train_cfg['epochs']
    
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.01,
        total_iters=warmup_epochs,
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_epochs - warmup_epochs,
        eta_min=train_cfg['lr'] * 0.01,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )
    
    # Resume from checkpoint
    start_epoch = 0
    if args.resume:
        logger.info(f"Loading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        logger.info(f"Resumed from epoch {start_epoch}")
    
    # Training loop
    logger.info("=" * 60)
    logger.info("Starting pretraining...")
    logger.info("=" * 60)
    
    best_val_loss = float('inf')
    best_val_auroc = 0.0
    num_epochs = 1 if args.debug else total_epochs
    
    for epoch in range(start_epoch, num_epochs):
        # Train
        train_loss, train_metrics = train_one_epoch(
            model, criterion, train_loader, optimizer, device, epoch, num_epochs, config
        )
        
        # Log training metrics
        logger.log_metrics(
            epoch=epoch,
            phase='train',
            metrics=train_metrics,
            lr=optimizer.param_groups[0]['lr'],
        )
        
        # Validate
        val_interval = config['training'].get('val_interval', 5)
        if (epoch + 1) % val_interval == 0 or epoch == num_epochs - 1:
            val_loss, val_metrics = validate(model, criterion, val_loader, device)
            
            logger.log_metrics(
                epoch=epoch,
                phase='val',
                metrics=val_metrics,
            )
            
            # Save best model (by validation AUROC or loss)
            val_auroc = val_metrics.get('mean_auroc', 0)
            if val_auroc > best_val_auroc:
                best_val_auroc = val_auroc
                save_checkpoint(
                    model, optimizer, scheduler, epoch, config,
                    'pretrain_best.pth', logger, is_best=True,
                )
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
        
        # Save checkpoint periodically
        save_interval = config['training'].get('save_interval', 10)
        if (epoch + 1) % save_interval == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, config,
                f'pretrain_epoch_{epoch+1}.pth', logger,
            )
        
        scheduler.step()
        
        if args.debug:
            logger.info("Debug mode: stopping after 1 epoch")
            break
    
    # Save final model
    save_checkpoint(
        model, optimizer, scheduler, num_epochs - 1, config,
        'pretrain_final.pth', logger,
    )
    
    # Also save backbone-only weights for easy loading
    backbone_path = Path(config['paths']['checkpoint_dir']) / 'pretrained_backbone.pth'
    torch.save({
        'backbone_state_dict': model.get_backbone_state_dict(),
        'config': config,
        'epoch': num_epochs - 1,
        'best_val_auroc': best_val_auroc,
    }, backbone_path)
    logger.info(f"Backbone-only weights saved to: {backbone_path}")
    
    logger.info("=" * 60)
    logger.info(f"Pretraining complete! Best val AUROC: {best_val_auroc:.4f}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
