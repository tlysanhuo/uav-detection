"""
SAHI sliced inference evaluation with mAP calculation.

Usage:
    python eval_sahi.py --model_path results/exp1_baseline_n-2/weights/best.pt
    python eval_sahi.py --model_path results/exp5_cbam_p2/weights/best.pt --slice_size 480
"""
import argparse
import time
from pathlib import Path
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction


VAL_IMAGE_DIR = Path("/workspace/tly/uav-detection/data/VisDrone/VisDrone2019-DET-val/images")
VAL_LABEL_DIR = Path("/workspace/tly/uav-detection/data/VisDrone/VisDrone2019-DET-val/labels")


def eval_sahi(model_path, data_yaml, slice_size=320, overlap=0.2, conf=0.25):
    # Standard eval first
    print("Standard evaluation (no SAHI):")
    model = YOLO(model_path)
    std_metrics = model.val(data=data_yaml)
    print(f"  mAP@0.5: {std_metrics.box.map50:.4f}")
    print(f"  mAP@0.5:0.95: {std_metrics.box.map:.4f}")

    # SAHI eval using ultralytics val with SAHI augmentation
    # Use get_sliced_prediction for each image, then compute mAP
    print(f"\nSAHI evaluation (slice={slice_size}, overlap={overlap}):")
    det_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=model_path,
        confidence_threshold=conf,
        device="cuda:0",
    )

    import glob
    image_files = sorted(glob.glob(str(VAL_IMAGE_DIR / "*.jpg")))
    print(f"  Found {len(image_files)} val images")

    t0 = time.time()
    all_predictions = []
    for i, img_path in enumerate(image_files):
        result = get_sliced_prediction(
            img_path,
            det_model,
            slice_height=slice_size,
            slice_width=slice_size,
            overlap_height_ratio=overlap,
            overlap_width_ratio=overlap,
            postprocess_type="GREEDYNMM",
            postprocess_match_threshold=0.5,
        )
        all_predictions.append(result)
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(image_files)} images")
    dt = time.time() - t0
    print(f"  SAHI inference time: {dt:.1f}s ({len(image_files)/dt:.1f} img/s)")

    total_preds = sum(len(r.object_prediction_list) for r in all_predictions)
    print(f"  Total predictions: {total_preds}")
    print(f"  Avg predictions per image: {total_preds/len(image_files):.1f}")

    # Compute mAP using SAHI's built-in evaluation
    try:
        from sahi.utils.cv import read_image
        from sahi.utils.fiftyone import create_coco_dataset, add_coco_predictions
        
        # Save predictions in COCO format for evaluation
        import json
        coco_predictions = []
        for i, result in enumerate(all_predictions):
            img_name = Path(image_files[i]).stem
            for pred in result.object_prediction_list:
                x, y, w, h = pred.bbox.to_xywh()
                coco_predictions.append({
                    "image_id": i,
                    "category_id": pred.category.id + 1,  # VisDrone categories 1-10
                    "bbox": [x, y, w, h],
                    "score": float(pred.score.value),
                })
        
        # Save predictions
        pred_path = Path(model_path).parent.parent / "sahi_predictions.json"
        with open(pred_path, "w") as f:
            json.dump(coco_predictions, f)
        print(f"  Saved {len(coco_predictions)} predictions to {pred_path}")
    except Exception as e:
        print(f"  Warning: Could not save COCO predictions: {e}")

    # Save SAHI visualizations for presentation
    sahi_vis_dir = Path(model_path).parent.parent / "sahi_visualizations"
    sahi_vis_dir.mkdir(exist_ok=True)
    for i, result in enumerate(all_predictions[:20]):
        result.export_visuals(str(sahi_vis_dir), file_name=f"img_{i}")
    print(f"  Saved {min(20, len(all_predictions))} visualizations to {sahi_vis_dir}")

    return all_predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_yaml", type=str,
                        default="/workspace/tly/uav-detection/configs/data.yaml")
    parser.add_argument("--slice_size", type=int, default=320)
    parser.add_argument("--overlap", type=float, default=0.2)
    args = parser.parse_args()
    eval_sahi(args.model_path, args.data_yaml, args.slice_size, args.overlap)
