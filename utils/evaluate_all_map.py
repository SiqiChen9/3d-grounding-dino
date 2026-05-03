import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from datasets import RSNAVolumeDataset
from models import build_model
from utils.metrics import compute_map


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate mAP on test split and run train/validation overfitting check."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="report\\pretrain\\default\\pretrain_best.pth",
        help="Path to model checkpoint (.pth).",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/home/woody/iwi5/iwi5378h/rsna2023",
        help="Dataset root directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/map_eval",
        help="Directory to save output figures.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on.",
    )
    parser.add_argument(
        "--iou-thresholds",
        type=str,
        default="0.1,0.2,0.3,0.4,0.5,0.6,0.7",
        help="Comma-separated IoU thresholds for mAP evaluation.",
    )
    return parser.parse_args()


def evaluate_map_on_split(split_dataset, split_name, model, device, num_classes, iou_thresholds):
    num_samples = len(split_dataset)
    all_predictions = []
    all_ground_truths = []

    loader = DataLoader(split_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    print(f"Evaluating {split_name} split ({num_samples} samples)...")
    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"{split_name} inference"):
            volume_input = batch["volume"].to(device).float()
            outputs = model(volume_input)

            pred_logits = outputs["pred_logits"][0]
            pred_boxes = outputs["pred_boxes"][0]

            pred_probs = pred_logits.softmax(-1)[:, :-1]
            pred_scores = pred_probs.max(-1)[0]
            pred_labels = pred_probs.argmax(-1)

            all_predictions.append(
                {
                    "boxes": pred_boxes.cpu().numpy(),
                    "scores": pred_scores.cpu().numpy(),
                    "labels": pred_labels.cpu().numpy(),
                }
            )

            all_ground_truths.append(
                {
                    "boxes": batch["boxes"][0].cpu().numpy(),
                    "labels": batch["labels"][0].cpu().numpy(),
                }
            )

    return compute_map(
        predictions=all_predictions,
        ground_truths=all_ground_truths,
        num_classes=num_classes,
        iou_thresholds=iou_thresholds,
    )


def plot_test_map(test_map_results, output_path):
    metrics = [k for k in test_map_results.keys() if k != "mAP"]
    values = [test_map_results[k] for k in metrics]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(metrics, values, color="steelblue")
    ax.axhline(
        y=test_map_results["mAP"],
        color="#f39c12",
        linestyle="--",
        linewidth=2,
        label=f"Overall mAP: {test_map_results['mAP']:.4f}",
    )

    ax.set_ylabel("mAP Score", fontsize=12)
    ax.set_title("Test Set: Mean Average Precision at Different IoU Thresholds", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.01,
            f"{height:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_overfitting_check(train_map_results, val_map_results, output_path):
    metrics = [k for k in train_map_results.keys() if k != "mAP"]
    train_values = [train_map_results[k] for k in metrics]
    val_values = [val_map_results[k] for k in metrics]

    x = np.arange(len(metrics))
    bar_width = 0.36
    gap_values = np.array(val_values) - np.array(train_values)
    overall_gap = val_map_results["mAP"] - train_map_results["mAP"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    bars_train = ax1.bar(
        x - bar_width / 2,
        train_values,
        width=bar_width,
        color="#4C78A8",
        label="Train mAP",
    )
    bars_val = ax1.bar(
        x + bar_width / 2,
        val_values,
        width=bar_width,
        color="#54A24B",
        label="Validation mAP",
    )

    ax1.axhline(
        y=train_map_results["mAP"],
        color="#2A5C8A",
        linestyle="--",
        linewidth=2,
        label=f"Train overall mAP: {train_map_results['mAP']:.4f}",
    )
    ax1.axhline(
        y=val_map_results["mAP"],
        color="#2E7D32",
        linestyle="--",
        linewidth=2,
        label=f"Validation overall mAP: {val_map_results['mAP']:.4f}",
    )

    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics, rotation=25)
    ax1.set_ylim(0, 1.0)
    ax1.set_ylabel("mAP Score")
    ax1.set_title("Train vs Validation mAP Across IoU Thresholds")
    ax1.grid(axis="y", alpha=0.3)
    ax1.legend(fontsize=9)

    for bar in list(bars_train) + list(bars_val):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.01,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    dataset_names = ["Train", "Validation"]
    dataset_map_values = [train_map_results["mAP"], val_map_results["mAP"]]
    dataset_colors = ["#4C78A8", "#54A24B"]

    bars_split = ax2.bar(dataset_names, dataset_map_values, color=dataset_colors)
    ax2.set_ylim(0, 1.0)
    ax2.set_ylabel("Overall mAP")
    ax2.set_title("Overall mAP by Split (Overfitting View)")
    ax2.grid(axis="y", alpha=0.3)

    for bar in bars_split:
        h = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.01,
            f"{h:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    summary_text = f"VAL - TRAIN: {overall_gap:+.4f}"
    ax2.text(
        0.05,
        0.95,
        summary_text,
        transform=ax2.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    )

    plt.suptitle(
        f"Overfitting Check: overall VAL-TRAIN mAP gap = {overall_gap:+.4f}",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    print("Per-IoU VAL-TRAIN gaps:")
    for metric_name, gap in zip(metrics, gap_values):
        print(f"  {metric_name}: {gap:+.4f}")
    print(f"Overall mAP VAL-TRAIN gap: {overall_gap:+.4f}")


def main():
    args = parse_args()
    iou_thresholds = [float(x.strip()) for x in args.iou_thresholds.split(",") if x.strip()]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    print(f"Using device: {device}")

    checkpoint_path = Path(args.checkpoint)
    data_dir = Path(args.data_dir)

    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    config = checkpoint["config"]

    print("Building model...")
    model = build_model(config)
    missing, unexpected = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    print(f"Loading with strict=False")
    print(f"Unexpected keys (skipped): {len(unexpected)}")
    print(f"Missing keys: {len(missing)}")
    model = model.to(device)
    model.eval()

    print(f"Loading dataset from {data_dir}...")
    full_dataset = RSNAVolumeDataset(
        data_dir=str(data_dir),
        target_width=config["data"]["target_width"],
        train=False,
        augment=False,
        image_format=config["data"].get("image_format", "dcm"),
    )

    train_ratio = config["data"].get("train_split", 0.7)
    val_ratio = config["data"].get("val_split", 0.15)
    test_ratio = config["data"].get("test_split", 0.15)
    random_seed = config["data"].get("random_seed", 42)

    total_ratio = train_ratio + val_ratio + test_ratio
    train_ratio /= total_ratio
    val_ratio /= total_ratio
    test_ratio /= total_ratio

    total_size = len(full_dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(random_seed),
    )

    print(f"Split sizes: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")

    num_classes = config["model"]["num_classes"]
    test_map_results = evaluate_map_on_split(
        test_dataset, "TEST", model, device, num_classes, iou_thresholds
    )
    train_map_results = evaluate_map_on_split(
        train_dataset, "TRAIN", model, device, num_classes, iou_thresholds
    )
    val_map_results = evaluate_map_on_split(
        val_dataset, "VAL", model, device, num_classes, iou_thresholds
    )

    print("\n" + "=" * 68)
    print("Split mAP Summary")
    print("=" * 68)
    print(f"TRAIN mAP: {train_map_results['mAP']:.4f}")
    print(f"VAL   mAP: {val_map_results['mAP']:.4f}")
    print(f"TEST  mAP: {test_map_results['mAP']:.4f}")
    print("-" * 68)
    print(f"VAL - TRAIN gap (overfitting gap): {(val_map_results['mAP'] - train_map_results['mAP']):+.4f}")
    print(f"TEST - VAL gap: {(test_map_results['mAP'] - val_map_results['mAP']):+.4f}")
    print(f"TEST - TRAIN gap: {(test_map_results['mAP'] - train_map_results['mAP']):+.4f}")
    print("=" * 68)

    test_fig_path = output_dir / "test_map.png"
    overfit_fig_path = output_dir / "overfitting_check.png"

    plot_test_map(test_map_results, test_fig_path)
    plot_overfitting_check(train_map_results, val_map_results, overfit_fig_path)

    print(f"Saved test mAP figure to: {test_fig_path}")
    print(f"Saved overfitting check figure to: {overfit_fig_path}")


if __name__ == "__main__":
    main()
