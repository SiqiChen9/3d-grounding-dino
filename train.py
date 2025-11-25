"""
Training script for 3D Grounding-DETR.
Main entry point for model training.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import yaml
import argparse
import os
from tqdm import tqdm
from pathlib import Path

from datasets import RSNAVolumeDataset, collate_fn
from models import build_model
from models.losses import HungarianMatcher, SetCriterion


def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_dataloaders(config: dict):
    """Create train and validation dataloaders."""
    data_cfg = config['data']
    
    # Create full dataset
    full_dataset = RSNAVolumeDataset(
        data_dir=data_cfg['dataset_path'],
        volume_size=tuple(data_cfg['image_size']),
        train=True,
        augment=True
    )
    
    # Split into train/val
    train_size = int(len(full_dataset) * data_cfg.get('train_split', 0.8))
    val_size = len(full_dataset) - train_size
    
    if val_size > 0:
        train_dataset, val_dataset = random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
    else:
        train_dataset = full_dataset
        val_dataset = None
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=data_cfg['batch_size'],
        shuffle=True,
        num_workers=data_cfg.get('num_workers', 0),
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=data_cfg['batch_size'],
            shuffle=False,
            num_workers=data_cfg.get('num_workers', 0),
            collate_fn=collate_fn,
            pin_memory=True
        )
    
    return train_loader, val_loader


def create_optimizer_and_scheduler(model, config):
    """Create optimizer and learning rate scheduler."""
    train_cfg = config['training']
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=train_cfg['lr'],
        weight_decay=train_cfg['weight_decay']
    )
    
    # Scheduler: warmup + cosine annealing
    warmup_epochs = train_cfg.get('warmup_epochs', 5)
    total_epochs = train_cfg['epochs']
    
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        total_iters=warmup_epochs
    )
    
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_epochs - warmup_epochs,
        eta_min=train_cfg['lr'] * 0.01
    )
    
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs]
    )
    
    return optimizer, scheduler


def train_one_epoch(
    model,
    criterion,
    data_loader,
    optimizer,
    device,
    epoch,
    config
):
    """Train for one epoch."""
    model.train()
    criterion.train()
    
    total_loss = 0
    losses_dict = {}
    
    pbar = tqdm(data_loader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(pbar):
        # Move to device and ensure float32
        volumes = batch['volumes'].to(device).float()  # Ensure float32
        target_labels = [labels.to(device) for labels in batch['labels']]
        target_boxes = [boxes.to(device) for boxes in batch['boxes']]
        
        # Forward pass
        outputs = model(volumes)
        
        # Compute loss
        losses = criterion(
            outputs['pred_logits'],
            outputs['pred_boxes'],
            target_labels,
            target_boxes
        )
        
        loss = losses['loss_total']
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if config['training'].get('clip_max_norm', 0) > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config['training']['clip_max_norm']
            )
        
        optimizer.step()
        
        # Accumulate losses
        total_loss += loss.item()
        for k, v in losses.items():
            if k not in losses_dict:
                losses_dict[k] = 0
            losses_dict[k] += v.item()
        
        # Update progress bar
        pbar.set_postfix({'loss': loss.item()})
    
    # Average losses
    num_batches = len(data_loader)
    avg_loss = total_loss / num_batches
    avg_losses_dict = {k: v / num_batches for k, v in losses_dict.items()}
    
    return avg_loss, avg_losses_dict


@torch.no_grad()
def validate(model, criterion, data_loader, device):
    """Validate model."""
    model.eval()
    criterion.eval()
    
    total_loss = 0
    losses_dict = {}
    
    for batch in tqdm(data_loader, desc="Validation"):
        # Move to device and ensure float32
        volumes = batch['volumes'].to(device).float()  # Ensure float32
        target_labels = [labels.to(device) for labels in batch['labels']]
        target_boxes = [boxes.to(device) for boxes in batch['boxes']]
        
        # Forward pass
        outputs = model(volumes)
        
        # Compute loss
        losses = criterion(
            outputs['pred_logits'],
            outputs['pred_boxes'],
            target_labels,
            target_boxes
        )
        
        total_loss += losses['loss_total'].item()
        for k, v in losses.items():
            if k not in losses_dict:
                losses_dict[k] = 0
            losses_dict[k] += v.item()
    
    # Average losses
    num_batches = len(data_loader)
    avg_loss = total_loss / num_batches
    avg_losses_dict = {k: v / num_batches for k, v in losses_dict.items()}
    
    return avg_loss, avg_losses_dict


def save_checkpoint(model, optimizer, scheduler, epoch, config, filename):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'config': config
    }
    
    checkpoint_dir = Path(config['paths']['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    filepath = checkpoint_dir / filename
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")


def main():
    parser = argparse.ArgumentParser(description='Train 3D Grounding-DETR')
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                       help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    parser.add_argument('--debug', action='store_true',
                       help='Debug mode (single batch)')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    print(f"Config loaded from {args.config}")
    print(yaml.dump(config, default_flow_style=False))
    
    # Device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader = create_dataloaders(config)
    print(f"Train batches: {len(train_loader)}")
    if val_loader:
        print(f"Val batches: {len(val_loader)}")
    
    # Create model
    print("\nBuilding model...")
    model = build_model(config)
    model = model.to(device)
    
    # Create criterion
    loss_cfg = config['loss']
    matcher = HungarianMatcher(
        cost_class=loss_cfg['cost_class'],
        cost_bbox=loss_cfg['cost_bbox'],
        cost_giou=loss_cfg['cost_giou']
    )
    
    weight_dict = {
        'loss_ce': loss_cfg['weight_ce'],
        'loss_l1': loss_cfg['weight_l1'],
        'loss_giou': loss_cfg['weight_giou']
    }
    
    criterion = SetCriterion(
        num_classes=config['model']['num_classes'],
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=loss_cfg['eos_coef']
    )
    criterion = criterion.to(device)
    
    # Create optimizer and scheduler
    optimizer, scheduler = create_optimizer_and_scheduler(model, config)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        print(f"\nLoading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from epoch {start_epoch}")
    
    # Training loop
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60 + "\n")
    
    best_val_loss = float('inf')
    
    num_epochs = 1 if args.debug else config['training']['epochs']
    
    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Train
        train_loss, train_losses = train_one_epoch(
            model, criterion, train_loader, optimizer, device, epoch, config
        )
        
        print(f"Train Loss: {train_loss:.4f}")
        for k, v in train_losses.items():
            print(f"  {k}: {v:.4f}")
        
        # Validate
        if val_loader and (epoch + 1) % config['training']['val_interval'] == 0:
            val_loss, val_losses = validate(model, criterion, val_loader, device)
            print(f"Val Loss: {val_loss:.4f}")
            for k, v in val_losses.items():
                print(f"  {k}: {v:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    model, optimizer, scheduler, epoch, config,
                    'model_best.pth'
                )
        
        # Save checkpoint periodically
        if (epoch + 1) % config['training']['save_interval'] == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, config,
                f'checkpoint_epoch_{epoch+1}.pth'
            )
        
        # Step scheduler
        scheduler.step()
        
        if args.debug:
            print("\nDebug mode: stopping after 1 epoch")
            break
    
    # Save final model
    save_checkpoint(
        model, optimizer, scheduler, num_epochs-1, config,
        'model_final.pth'
    )
    
    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)


if __name__ == '__main__':
    main()
