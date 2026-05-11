"""
Generate training curve plots from log files.
"""
import re
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def parse_log(log_path):
    """Parse ultralytics training log - extract val metrics from 'all' lines."""
    with open(log_path) as f:
        content = f.read()

    # Find all "all  N  N  P  R  mAP50  mAP50-95" patterns
    pattern = r'all\s+\d+\s+\d+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)'
    matches = re.findall(pattern, content)

    epochs = []
    for i, (p, r, map50, map50_95) in enumerate(matches):
        epochs.append({
            'epoch': i + 1,
            'precision': float(p),
            'recall': float(r),
            'mAP50': float(map50),
            'mAP50-95': float(map50_95),
        })
    return epochs


def parse_csv(csv_path):
    """Parse results.csv."""
    import csv
    epochs = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append({
                'epoch': int(float(row['epoch'])),
                'precision': float(row['metrics/precision(B)']),
                'recall': float(row['metrics/recall(B)']),
                'mAP50': float(row['metrics/mAP50(B)']),
                'mAP50-95': float(row['metrics/mAP50-95(B)']),
            })
    return epochs


def main():
    base = Path("/workspace/tly/uav-detection")
    experiments = {}

    # Exp1
    csv1 = base / "results/exp1_baseline_n-2/results.csv"
    if csv1.exists():
        experiments["Exp1: YOLOv8n"] = parse_csv(csv1)

    # Exp2
    csv2 = base / "results/exp2_baseline_s-2/results.csv"
    if csv2.exists():
        experiments["Exp2: YOLOv8s"] = parse_csv(csv2)

    # Exp3
    csv3 = base / "results/exp3_cbam_v3/results.csv"
    if csv3.exists():
        experiments["Exp3: YOLOv8n+CBAM"] = parse_csv(csv3)

    # Exp5 - from log
    log5 = base / "logs/exp5_v2.log"
    if log5.exists():
        data5 = parse_log(log5)
        if data5:
            experiments["Exp5: YOLOv8n-P2+CBAM"] = data5

    # Print summary
    print(f"\n{'Experiment':<25} {'Epochs':>6} {'Best mAP50':>10} {'Best mAP50-95':>13}")
    print("-" * 60)
    for name, data in experiments.items():
        if data:
            best_map50 = max(d['mAP50'] for d in data)
            best_map50_95 = max(d['mAP50-95'] for d in data)
            print(f"{name:<25} {len(data):>6} {best_map50:>10.4f} {best_map50_95:>13.4f}")

    # Comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    metrics = [('mAP50', 'mAP@0.5'), ('mAP50-95', 'mAP@0.5:0.95'),
               ('precision', 'Precision'), ('recall', 'Recall')]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for ax, (key, title) in zip(axes.flat, metrics):
        for i, (name, data) in enumerate(experiments.items()):
            if not data or key not in data[0]:
                continue
            epochs = [d['epoch'] for d in data]
            values = [d[key] for d in data]
            ax.plot(epochs, values, label=name, color=colors[i % len(colors)], linewidth=1.5)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(base / "results/comparison_curves.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {base / 'results/comparison_curves.png'}")

    # Exp5 standalone plot
    if "Exp5: YOLOv8n-P2+CBAM" in experiments:
        data = experiments["Exp5: YOLOv8n-P2+CBAM"]
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        for ax, (key, title) in zip(axes, metrics):
            epochs = [d['epoch'] for d in data]
            values = [d[key] for d in data]
            ax.plot(epochs, values, color='#d62728', linewidth=1.5)
            ax.set_title(title)
            ax.set_xlabel('Epoch')
            ax.grid(True, alpha=0.3)
        plt.suptitle('Exp5: YOLOv8n-P2+CBAM Training Curves', fontsize=13)
        plt.tight_layout()
        plt.savefig(base / "results/exp5_cbam_p2/training_curves.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {base / 'results/exp5_cbam_p2/training_curves.png'}")


if __name__ == "__main__":
    main()
