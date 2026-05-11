"""
Compare results across all experiments.

Usage:
    python visualize.py
"""
import csv
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

EXPERIMENTS = {
    "Exp1: YOLOv8n": "exp1_baseline_n-2",
    "Exp2: YOLOv8s": "exp2_baseline_s-2",
    "Exp3: YOLOv8n+CBAM": "exp3_cbam_v3",
    "Exp5: YOLOv8n-P2+CBAM": "exp5_cbam_p2",
}


def read_best_metrics(exp_dir):
    csv_path = exp_dir / "results.csv"
    if not csv_path.exists():
        return None
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return None
    best_map50 = -1
    best_row = None
    for row in rows:
        map50 = float(row["metrics/mAP50(B)"])
        if map50 > best_map50:
            best_map50 = map50
            best_row = row
    return {
        "mAP@0.5": float(best_row["metrics/mAP50(B)"]),
        "mAP@0.5:0.95": float(best_row["metrics/mAP50-95(B)"]),
        "Precision": float(best_row["metrics/precision(B)"]),
        "Recall": float(best_row["metrics/recall(B)"]),
        "epoch": int(float(best_row["epoch"])),
    }


def main():
    print(f"\n{'Experiment':<25} {'mAP@0.5':>10} {'mAP@0.5:0.95':>14} {'Precision':>10} {'Recall':>8} {'Best Ep':>8}")
    print("-" * 80)
    for name, dirname in EXPERIMENTS.items():
        exp_dir = RESULTS_DIR / dirname
        metrics = read_best_metrics(exp_dir)
        if metrics:
            print(f"{name:<25} {metrics['mAP@0.5']:>10.4f} {metrics['mAP@0.5:0.95']:>14.4f} {metrics['Precision']:>10.4f} {metrics['Recall']:>8.4f} {metrics['epoch']:>8d}")
        else:
            print(f"{name:<25} {'N/A':>10} {'N/A':>14} {'N/A':>10} {'N/A':>8} {'N/A':>8}")

    print(f"\nAvailable presentation materials per experiment:")
    for name, dirname in EXPERIMENTS.items():
        exp_dir = RESULTS_DIR / dirname
        if not exp_dir.exists():
            print(f"  {name}: directory not found")
            continue
        files = list(exp_dir.glob("*.png")) + list(exp_dir.glob("*.jpg"))
        print(f"  {name}: {len(files)} images ({', '.join(sorted(f.name for f in files[:8]))})")


if __name__ == "__main__":
    main()
