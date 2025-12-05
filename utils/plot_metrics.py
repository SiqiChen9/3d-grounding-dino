"""
Utility to plot training metrics from log files.
"""
import matplotlib.pyplot as plt
import json
from pathlib import Path
import argparse


def plot_metrics(log_dir: str, run_name: str, save_path: str = None):
    """
    Plot training metrics from a log directory.
    
    Args:
        log_dir: Base log directory
        run_name: Name of the training run
        save_path: Optional path to save the plot
    """
    log_dir = Path(log_dir)
    metrics_file = log_dir / run_name / "metrics.jsonl"
    
    if not metrics_file.exists():
        print(f"Metrics file not found: {metrics_file}")
        return
    
    # Load metrics
    train_metrics = []
    val_metrics = []
    
    with open(metrics_file, 'r') as f:
        for line in f:
            metric = json.loads(line)
            if metric['phase'] == 'train':
                train_metrics.append(metric)
            else:
                val_metrics.append(metric)
    
    if not train_metrics:
        print("No training metrics found")
        return
    
    # Extract data
    train_epochs = [m['epoch'] for m in train_metrics]
    train_loss = [m['loss_total'] for m in train_metrics]
    train_ce = [m.get('loss_ce', 0) for m in train_metrics]
    train_l1 = [m.get('loss_l1', 0) for m in train_metrics]
    train_giou = [m.get('loss_giou', 0) for m in train_metrics]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Training Metrics - {run_name}', fontsize=16)
    
    # Plot 1: Total Loss
    ax = axes[0, 0]
    ax.plot(train_epochs, train_loss, 'b-', label='Train', linewidth=2)
    if val_metrics:
        val_epochs = [m['epoch'] for m in val_metrics]
        val_loss = [m['loss_total'] for m in val_metrics]
        ax.plot(val_epochs, val_loss, 'r-', label='Val', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Total Loss')
    ax.set_yscale('log')
    ax.set_title('Total Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Loss Components
    ax = axes[0, 1]
    ax.plot(train_epochs, train_ce, label='CE Loss', linewidth=2)
    ax.plot(train_epochs, train_l1, label='L1 Loss', linewidth=2)
    ax.plot(train_epochs, train_giou, label='GIoU Loss', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_yscale('log')
    ax.set_title('Loss Components')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Learning Rate
    ax = axes[1, 0]
    if 'lr' in train_metrics[0]:
        train_lr = [m['lr'] for m in train_metrics]
        ax.plot(train_epochs, train_lr, 'g-', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No LR data', ha='center', va='center')
        ax.axis('off')
    
    # Plot 4: Gradient Norm
    ax = axes[1, 1]
    if 'grad_norm' in train_metrics[0]:
        train_grad = [m['grad_norm'] for m in train_metrics]
        ax.plot(train_epochs, train_grad, 'purple', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Gradient Norm')
        ax.set_title('Gradient Norm')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No gradient data', ha='center', va='center')
        ax.axis('off')
    
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()


def list_runs(log_dir: str):
    """List all available training runs."""
    log_dir = Path(log_dir)
    
    if not log_dir.exists():
        print(f"Log directory not found: {log_dir}")
        return
    
    runs = [d.name for d in log_dir.iterdir() if d.is_dir()]
    
    if not runs:
        print(f"No training runs found in {log_dir}")
        return
    
    print(f"Available training runs in {log_dir}:")
    for run in sorted(runs, reverse=True):
        run_dir = log_dir / run
        config_file = run_dir / "config.yaml"
        metrics_file = run_dir / "metrics.jsonl"
        
        # Count epochs
        num_epochs = 0
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                for line in f:
                    metric = json.loads(line)
                    if metric['phase'] == 'train':
                        num_epochs += 1
        
        status = "✓" if config_file.exists() and metrics_file.exists() else "✗"
        print(f"  {status} {run} ({num_epochs} epochs)")


def main():
    parser = argparse.ArgumentParser(description='Plot training metrics')
    parser.add_argument('--log-dir', type=str, default='./logs',
                       help='Base log directory')
    parser.add_argument('--run-name', type=str, default=None,
                       help='Name of the run to plot')
    parser.add_argument('--list', action='store_true',
                       help='List all available runs')
    parser.add_argument('--save', type=str, default=None,
                       help='Save plot to this path instead of showing')
    
    args = parser.parse_args()
    
    if args.list:
        list_runs(args.log_dir)
    elif args.run_name:
        plot_metrics(args.log_dir, args.run_name, args.save)
    else:
        print("Please specify --run-name or use --list to see available runs")


if __name__ == '__main__':
    main()
