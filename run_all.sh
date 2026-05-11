#!/bin/bash
# One-click: convert data -> train all -> evaluate
set -e

PROJ="/workspace/tly/uav-detection"
source /workspace/tly/miniconda3/bin/activate uav_detect

echo "===== Step 1: Convert VisDrone annotations ====="
python $PROJ/scripts/convert_visdrone.py --src $PROJ/data/VisDrone

echo "===== Step 2: Exp1 - YOLOv8n baseline ====="
python $PROJ/scripts/train.py --exp 1 --epochs 100 --device 0

echo "===== Step 3: Exp2 - YOLOv8s baseline ====="
python $PROJ/scripts/train.py --exp 2 --epochs 100 --device 0

echo "===== Step 4: Exp3 - YOLOv8n + CBAM ====="
python $PROJ/scripts/train.py --exp 3 --epochs 100 --device 0

echo "===== Step 5: Exp4 - SAHI on Exp1 model ====="
python $PROJ/scripts/eval_sahi.py --model_path $PROJ/results/exp1_baseline_n/weights/best.pt

echo "===== Step 6: Exp5 - CBAM + P2 + SAHI ====="
python $PROJ/scripts/train.py --exp 5 --epochs 100 --device 0
python $PROJ/scripts/eval_sahi.py --model_path $PROJ/results/exp5_cbam_p2_sahi/weights/best.pt

echo "===== Compare all results ====="
python $PROJ/scripts/visualize.py
echo "===== ALL DONE ====="
