"""
Unified training script for UAV detection experiments.

Exp1: YOLOv8n baseline
Exp2: YOLOv8s baseline
Exp3: YOLOv8n + CBAM (backbone)
Exp4: YOLOv8n + SAHI (inference only, use eval_sahi.py)
Exp5: YOLOv8n-P2 + CBAM (backbone) + SAHI (inference)

Usage:
    python train.py --exp 1 --epochs 100
    python train.py --exp 3 --epochs 100 --device 0
    python train.py --exp 5 --epochs 100 --device 0
"""
import sys
import argparse
from pathlib import Path
from copy import deepcopy

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "scripts"))

from ultralytics import YOLO

DATA_YAML = str(PROJECT / "configs" / "data.yaml")
RESULTS_DIR = str(PROJECT / "results")


def _patch_trainer(custom_model):
    """Patch DetectionTrainer to use custom model and save it properly."""
    from ultralytics.models.yolo.detect import DetectionTrainer
    from datetime import datetime
    from ultralytics import __version__
    from ultralytics.utils import GIT
    import io, torch
    from ultralytics.engine.trainer import unwrap_model

    original_setup = DetectionTrainer.setup_model
    original_save = DetectionTrainer.save_model

    def patched_setup(self):
        if isinstance(self.model, type(custom_model)):
            return
        self.model = custom_model
        return

    def patched_save(self):
        """Save model with full model object (not None) so CBAM is preserved."""
        import io
        ema = deepcopy(unwrap_model(self.ema.ema)).half()
        if not all(torch.isfinite(v).all() for v in ema.state_dict().values() if isinstance(v, torch.Tensor)):
            return False

        # Also save the full model (not just EMA) so CBAM modules are preserved
        full_model = deepcopy(unwrap_model(self.model)).half()

        buffer = io.BytesIO()
        torch.save(
            {
                "epoch": self.epoch,
                "best_fitness": self.best_fitness,
                "model": full_model,  # Save full model with CBAM, not None
                "ema": ema,
                "updates": self.ema.updates,
                "optimizer": None,  # Skip optimizer to reduce size
                "scaler": self.scaler.state_dict() if self.scaler else None,
                "train_args": vars(self.args),
                "train_metrics": {**self.metrics, **{"fitness": self.fitness}},
                "train_results": self.read_results_csv(),
                "date": datetime.now().isoformat(),
                "version": __version__,
                "git": {
                    "root": str(GIT.root),
                    "branch": GIT.branch,
                    "commit": GIT.commit,
                    "origin": GIT.origin,
                },
                "license": "AGPL-3.0 (https://ultralytics.com/license)",
                "docs": "https://docs.ultralytics.com",
            },
            buffer,
        )
        serialized_ckpt = buffer.getvalue()
        self.wdir.mkdir(parents=True, exist_ok=True)
        self.last.write_bytes(serialized_ckpt)
        if self.best_fitness == self.fitness:
            self.best.write_bytes(serialized_ckpt)
        if (self.save_period > 0) and (self.epoch % self.save_period == 0):
            (self.wdir / f"epoch{self.epoch}.pt").write_bytes(serialized_ckpt)
        return True

    DetectionTrainer.setup_model = patched_setup
    DetectionTrainer.save_model = patched_save
    return original_setup, original_save


def _restore_trainer(originals):
    """Restore original methods."""
    from ultralytics.models.yolo.detect import DetectionTrainer
    DetectionTrainer.setup_model = originals[0]
    DetectionTrainer.save_model = originals[1]


def train(exp, epochs=100, imgsz=640, batch=None, device="0"):
    if exp == 1:
        model = YOLO("yolov8n.pt")
        name = "exp1_baseline_n"
        bs = batch or 32
    elif exp == 2:
        model = YOLO("yolov8s.pt")
        name = "exp2_baseline_s"
        bs = batch or 16
    elif exp == 3:
        import custom_modules
        model = custom_modules.build_cbam_model("yolov8n.pt")
        name = "exp3_cbam_v4"
        bs = batch or 32
    elif exp == 4:
        print("Exp4 is SAHI inference only. Run: python eval_sahi.py")
        return
    elif exp == 5:
        import custom_modules
        model = custom_modules.build_cbam_p2_model("yolov8n-p2.yaml", "yolov8n.pt")
        name = "exp5_cbam_p2_v2"
        bs = batch or 16
    else:
        raise ValueError(f"Unknown exp {exp}")

    print(f"\n{'='*50}")
    print(f"Experiment {exp}: {name}")
    print(f"  epochs={epochs}, imgsz={imgsz}, batch={bs}, device={device}")
    print(f"  params={sum(p.numel() for p in model.model.parameters())/1e6:.2f}M")
    print(f"{'='*50}\n")

    has_custom = False
    for i, m in enumerate(model.model.model):
        t = type(m).__name__
        if 'CBAM' in t:
            print(f"  Layer {i}: {t} (CBAM confirmed)")
            has_custom = True

    if has_custom:
        originals = _patch_trainer(model.model)
        model.ckpt = {}

    try:
        results = model.train(
            data=DATA_YAML,
            epochs=epochs,
            imgsz=imgsz,
            batch=bs,
            device=device,
            project=RESULTS_DIR,
            name=name,
            patience=20,
            save=True,
            plots=True,
            mosaic=1.0,
            mixup=0.15,
            copy_paste=0.3,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=10.0,
            translate=0.1,
            shear=2.0,
            fliplr=0.5,
        )

        metrics = model.val()
        print(f"\n{'='*50}")
        print(f"Results for {name}:")
        print(f"  mAP@0.5:      {metrics.box.map50:.4f}")
        print(f"  mAP@0.5:0.95: {metrics.box.map:.4f}")
        print(f"  Precision:    {metrics.box.mp:.4f}")
        print(f"  Recall:       {metrics.box.mr:.4f}")
        print(f"{'='*50}")
    finally:
        if has_custom:
            _restore_trainer(originals)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=int, required=True, choices=[1, 2, 3, 4, 5])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--device", type=str, default="0")
    args = parser.parse_args()
    train(args.exp, args.epochs, args.imgsz, args.batch, args.device)
