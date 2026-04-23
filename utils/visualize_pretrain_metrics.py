
"""
Visualize pretraining JSONL logs with minimal outputs.

Outputs only:
1. metrics_curves.png
   - 4 line charts: loss_total, accuracy, mean_auroc, mean_ap
   - train/val are distinguished by blue/red
   - no markers
2. metrics_task_accuracy.png
   - task-wise validation accuracy bar chart (same logic as before)
3. runs_summary.json
   - keep the summary JSON output

Notes:
- Recursively search *.jsonl under the log directory.
- Parse JSONL files only.
- By default, the summary includes all discovered runs.
- The two PNGs are generated for a single representative run:
  * if multiple runs exist, the best run by composite_score is used;
  * otherwise, the only run is used.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_LOG_DIR = 'report\\pretrain\\default\\20260422_215918'
DEFAULT_OUT_DIR = './pretrain_visualization'


def find_jsonl_files(log_dir: str) -> List[Path]:
    """Recursively find all .jsonl files under the given directory."""
    base = Path(log_dir)
    if not base.exists():
        raise FileNotFoundError(f'Log directory does not exist: {base}')
    return sorted([p for p in base.rglob('*.jsonl') if p.is_file()])


def parse_metrics_file(path: Path) -> pd.DataFrame:
    """Parse a single JSONL metrics file into a DataFrame."""
    records = []
    with path.open('r', encoding='utf-8', errors='ignore') as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f'File {path} line {line_no} is not valid JSON: {e}') from e
            if isinstance(obj, dict):
                records.append(obj)

    if not records:
        raise ValueError(f'No valid records were parsed from file: {path}')

    df = pd.DataFrame(records)
    sort_cols = [c for c in ['epoch', 'phase', 'timestamp'] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)
    return df


def build_recommendation(summary: Dict) -> str:
    """Generate a short automatic conclusion from the summary."""
    parts = []

    if summary.get('best_val_accuracy') is not None:
        parts.append(
            f"Best validation accuracy at epoch {summary.get('best_val_accuracy_epoch')}, "
            f"accuracy={summary.get('best_val_accuracy'):.4f}."
        )

    if summary.get('best_val_mean_auroc') is not None:
        parts.append(
            f"Best validation AUROC at epoch {summary.get('best_val_mean_auroc_epoch')}, "
            f"mean_auroc={summary.get('best_val_mean_auroc'):.4f}."
        )

    if summary.get('best_val_mean_ap') is not None:
        parts.append(
            f"Best validation AP at epoch {summary.get('best_val_mean_ap_epoch')}, "
            f"mean_ap={summary.get('best_val_mean_ap'):.4f}."
        )

    gap = summary.get('gap_train_minus_val_accuracy@latest_val')
    if gap is not None:
        if gap > 0.10:
            parts.append(f"Train-val accuracy gap at the last validation is {gap:.4f}, suggesting some overfitting risk.")
        else:
            parts.append(f"Train-val accuracy gap at the last validation is {gap:.4f}, showing train/val are relatively close.")

    if not parts:
        return 'Not enough validation metrics were found to produce an automatic conclusion.'
    return ' '.join(parts)


def summarize_run(df: pd.DataFrame, run_name: str, source_file: str) -> Dict:
    """Summarize the key information from one run."""
    train_df = df[df['phase'] == 'train'].copy() if 'phase' in df.columns else pd.DataFrame()
    val_df = df[df['phase'] == 'val'].copy() if 'phase' in df.columns else pd.DataFrame()

    result: Dict = {
        'run': run_name,
        'source_file': source_file,
        'num_records': int(len(df)),
        'num_train_records': int(len(train_df)),
        'num_val_records': int(len(val_df)),
        'max_epoch_seen': int(df['epoch'].max()) if 'epoch' in df.columns and not df.empty else None,
    }

    def add_best(metric: str, mode: str = 'max'):
        if val_df.empty or metric not in val_df.columns:
            return
        row = val_df.loc[val_df[metric].idxmax()] if mode == 'max' else val_df.loc[val_df[metric].idxmin()]
        result[f'best_val_{metric}'] = float(row[metric])
        result[f'best_val_{metric}_epoch'] = int(row['epoch']) if 'epoch' in row else None

    for metric in ['accuracy', 'mean_auroc', 'mean_ap']:
        add_best(metric, 'max')
    add_best('loss_total', 'min')

    if not train_df.empty and not val_df.empty and 'epoch' in train_df.columns and 'epoch' in val_df.columns:
        latest_val_epoch = int(val_df['epoch'].max())
        result['latest_val_epoch'] = latest_val_epoch
        t = train_df[train_df['epoch'] == latest_val_epoch]
        v = val_df[val_df['epoch'] == latest_val_epoch]
        if not t.empty and not v.empty:
            for metric in ['accuracy', 'loss_total', 'mean_auroc', 'mean_ap']:
                if metric in t.columns and metric in v.columns:
                    result[f'gap_train_minus_val_{metric}@latest_val'] = float(t.iloc[-1][metric] - v.iloc[-1][metric])

    if not val_df.empty and len(val_df) >= 3:
        val_tail = val_df.sort_values('epoch').tail(3)
        for metric in ['accuracy', 'mean_auroc', 'mean_ap', 'loss_total']:
            if metric in val_tail.columns:
                result[f'last3_val_std_{metric}'] = float(val_tail[metric].std())

    score = 0.0
    have = False
    if 'best_val_accuracy' in result:
        score += 0.35 * result['best_val_accuracy']
        have = True
    if 'best_val_mean_auroc' in result:
        score += 0.35 * result['best_val_mean_auroc']
        have = True
    if 'best_val_mean_ap' in result:
        score += 0.30 * result['best_val_mean_ap']
        have = True
    result['composite_score'] = score if have else None
    result['composite_score_note'] = '0.35*best_val_accuracy + 0.35*best_val_mean_auroc + 0.30*best_val_mean_ap' if have else None
    result['recommendation'] = build_recommendation(result)
    return result


def _safe_run_name(jsonl_path: Path, root_dir: Path) -> str:
    """Generate a stable and readable run name from a path."""
    try:
        rel = jsonl_path.relative_to(root_dir)
        parts = list(rel.parts)
        parts[-1] = Path(parts[-1]).stem
        return '__'.join(parts)
    except Exception:
        return jsonl_path.stem


def plot_metrics_curves(df: pd.DataFrame, out_dir: Path):
    """Save a single PNG with 4 line charts: loss, acc, mean_auroc, mean_ap."""
    train_df = df[df['phase'] == 'train'].copy() if 'phase' in df.columns else pd.DataFrame()
    val_df = df[df['phase'] == 'val'].copy() if 'phase' in df.columns else pd.DataFrame()

    metrics = ['loss_total', 'accuracy', 'mean_auroc', 'mean_ap']
    titles = ['Total Loss', 'Accuracy', 'Mean AUROC', 'Mean AP']
    ylabels = ['Loss', 'Accuracy', 'AUROC', 'AP']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    axes = axes.flatten()

    for ax, metric, title, ylabel in zip(axes, metrics, titles, ylabels):
        if not train_df.empty and metric in train_df.columns:
            ax.plot(train_df['epoch'], train_df[metric], color='blue', linewidth=2.0, label='train')
        if not val_df.empty and metric in val_df.columns:
            ax.plot(val_df['epoch'], val_df[metric], color='red', linewidth=2.0, label='val')
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        if metric == 'loss_total':
            values = []
            if not train_df.empty and metric in train_df.columns:
                values.extend(train_df[metric].dropna().tolist())
            if not val_df.empty and metric in val_df.columns:
                values.extend(val_df[metric].dropna().tolist())
            if values and min(values) > 0:
                ax.set_yscale('log')
        ax.legend()

    axes[2].set_xlabel('Epoch')
    axes[3].set_xlabel('Epoch')

    fig.suptitle('Pretraining Metrics', fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_dir / 'metrics_curves.png', dpi=160)
    plt.close(fig)


def plot_task_accuracy(df: pd.DataFrame, out_dir: Path):
    """Save the task accuracy plot, keeping the previous logic unchanged."""
    val_df = df[df['phase'] == 'val'].copy() if 'phase' in df.columns else pd.DataFrame()
    organ_cols = [
        c for c in ['acc_bowel', 'acc_extravasation', 'acc_kidney', 'acc_liver', 'acc_spleen']
        if c in df.columns
    ]
    if val_df.empty or not organ_cols:
        return

    if 'accuracy' in val_df.columns:
        chosen = val_df.loc[val_df['accuracy'].idxmax()]
        title_suffix = f"best_val_acc_epoch_{int(chosen['epoch'])}"
    else:
        chosen = val_df.sort_values('epoch').iloc[-1]
        title_suffix = f"last_val_epoch_{int(chosen['epoch'])}"

    fig, ax = plt.subplots(figsize=(9, 4))
    vals = [float(chosen[c]) for c in organ_cols]
    ax.bar(organ_cols, vals)
    ax.set_ylim(0, 1)
    ax.set_ylabel('accuracy')
    ax.set_title(f'Task - wise Accuracy ({title_suffix})')
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / 'metrics_task_accuracy.png', dpi=160)
    plt.close(fig)


def list_jsonl_runs(log_dir: str):
    """List all JSONL files under the directory."""
    files = find_jsonl_files(log_dir)
    if not files:
        print(f'No JSONL files were found under {Path(log_dir).resolve()}')
        return
    print(f'Found the following JSONL files under {Path(log_dir).resolve()}:')
    for fp in files:
        print(f' - {fp}')


def choose_plot_run(run_items: List[Tuple[str, Path, pd.DataFrame, Dict]]) -> Tuple[str, Path, pd.DataFrame, Dict]:
    """Choose the representative run for the two PNGs.

    Priority:
    1. highest composite_score if available
    2. highest best_val_accuracy if available
    3. first successfully parsed run
    """
    if len(run_items) == 1:
        return run_items[0]

    def sort_key(item):
        summary = item[3]
        composite = summary.get('composite_score')
        acc = summary.get('best_val_accuracy')
        return (
            composite if composite is not None else -1.0,
            acc if acc is not None else -1.0,
        )

    return sorted(run_items, key=sort_key, reverse=True)[0]


def plot_metrics(log_dir: str = DEFAULT_LOG_DIR, out_dir: str = DEFAULT_OUT_DIR):
    """Parse JSONL logs, save only two PNGs and runs_summary.json."""
    root_dir = Path(log_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    jsonl_files = find_jsonl_files(log_dir)
    if not jsonl_files:
        print(f'No JSONL files were found under {root_dir.resolve()}')
        return

    summaries: List[Dict] = []
    run_items: List[Tuple[str, Path, pd.DataFrame, Dict]] = []

    print(f'Found {len(jsonl_files)} JSONL file(s):')
    for fp in jsonl_files:
        print(f'  - {fp}')

    for fp in jsonl_files:
        run_name = _safe_run_name(fp, root_dir)
        try:
            df = parse_metrics_file(fp)
        except Exception as e:
            print(f'[Skip] Failed to parse {fp}: {e}')
            continue

        summary = summarize_run(df, run_name, str(fp))
        summaries.append(summary)
        run_items.append((run_name, fp, df, summary))

    if not summaries:
        print('No JSONL file was parsed successfully. Nothing was generated.')
        return

    summary_df = pd.DataFrame(summaries)
    if 'composite_score' in summary_df.columns and summary_df['composite_score'].notna().any():
        summary_df = summary_df.sort_values('composite_score', ascending=False)
    elif 'best_val_accuracy' in summary_df.columns and summary_df['best_val_accuracy'].notna().any():
        summary_df = summary_df.sort_values('best_val_accuracy', ascending=False)

    # Keep runs_summary.json output.
    (out_dir / 'runs_summary.json').write_text(
        summary_df.to_json(orient='records', force_ascii=False, indent=2),
        encoding='utf-8'
    )

    # Save exactly two PNGs for a representative run.
    selected_run_name, selected_fp, selected_df, selected_summary = choose_plot_run(run_items)
    print(f'Selected run for plotting: {selected_run_name} ({selected_fp})')

    plot_metrics_curves(selected_df, out_dir)
    plot_task_accuracy(selected_df, out_dir)

    print(f'Output directory: {out_dir.resolve()}')
    print('Generated files: metrics_curves.png, metrics_task_accuracy.png, runs_summary.json')


def main():
    parser = argparse.ArgumentParser(description='Visualize pretraining JSONL logs with minimal outputs')
    parser.add_argument('--log-dir', type=str, default=DEFAULT_LOG_DIR,
                        help='Log directory to search recursively for *.jsonl')
    parser.add_argument('--outdir', type=str, default=DEFAULT_OUT_DIR,
                        help='Output directory')
    parser.add_argument('--list', action='store_true',
                        help='List all JSONL files in the directory')
    args = parser.parse_args()

    if args.list:
        list_jsonl_runs(args.log_dir)
    else:
        plot_metrics(args.log_dir, args.outdir)


if __name__ == '__main__':
    main()
