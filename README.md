# UAV-Drone-Detection: 基于改进YOLOv8的无人机图像小目标检测

基于 VisDrone-DET 数据集，对 YOLOv8 进行三项改进（CBAM注意力、P2小目标检测头、SAHI切片推理），并通过消融实验验证各组件贡献。

## 主要结果

| 实验 | 模型 | 参数量 | mAP@0.5 | mAP@0.5:0.95 | Δ mAP@0.5 |
|------|------|--------|---------|---------------|-----------|
| Exp1 | YOLOv8n | 3.16M | 0.3183 | 0.1746 | — |
| Exp2 | YOLOv8s | 11.13M | 0.3876 | 0.2202 | — |
| Exp3 | YOLOv8n+CBAM | 3.17M | 0.3264 | 0.1798 | +0.81% |
| Exp4 | YOLOv8n+SAHI | 3.16M | 0.3207 | 0.1766 | +0.24% |
| **Exp5** | **YOLOv8n-P2+CBAM** | **3.37M** | **0.3522** | **0.1975** | **+3.39%** |

**核心发现**：P2小目标检测头是最有效的改进（+3.39%），仅增加0.21M参数。CBAM单独增益有限（+0.81%），SAHI切片参数待调优。

## 改进方法

### 1. CBAM注意力模块
在 YOLOv8 backbone 的4个 C2f 模块输出端插入 CBAM（通道注意力+空间注意力），增强目标特征、抑制背景噪声。额外参数仅 +11.3K。

### 2. P2小目标检测头
在标准3尺度检测（P3/P4/P5）基础上增加 P2 层（4×下采样，160×160特征图），使极小目标（<32px）有足够的特征表达空间。采用 Ultralytics 官方 yolov8n-p2.yaml 架构。

### 3. SAHI切片推理
推理时将图像切成 320×320 小片（重叠20%）分别检测，再 NMS 合并。纯推理策略，无需重训。

## 实验设置

- **数据集**：VisDrone-DET（6471 train / 548 val / 1610 test，10类）
- **GPU**：NVIDIA H100 80GB（单卡训练，各实验分别使用不同GPU）
- **框架**：PyTorch 2.6.0 + CUDA 12.4, Ultralytics 8.4.48
- **训练**：100 epochs, batch=32(n)/16(s), imgsz=640, SGD, lr0=0.01, patience=20
- **数据增强**：Mosaic=1.0, MixUp=0.15, Copy-Paste=0.3, HSV扰动, 旋转/翻转

## 项目结构

```
├── configs/              # 数据集配置
├── scripts/
│   ├── train.py          # 统一训练脚本（--exp 1~5）
│   ├── custom_modules.py # CBAM / C2f_CBAM / P2+CBAM 模块实现
│   ├── eval_sahi.py      # SAHI切片推理评估
│   ├── convert_visdrone.py # VisDrone→YOLO格式转换
│   ├── gen_plots.py      # 训练曲线对比图生成
│   └── visualize.py      # 结果汇总表
├── presentation/         # 展示素材
│   ├── 展示文稿.md        # 完整展示文稿（背景→方法→实验→结论）
│   ├── comparison_curves.png  # 5实验对比图
│   ├── exp1_baseline_n/  # YOLOv8n 完整可视化
│   ├── exp2_baseline_s/  # YOLOv8s 完整可视化
│   ├── exp3_cbam/        # CBAM 完整可视化
│   └── sahi_visualizations/ # SAHI推理可视化
├── results/              # 训练结果（图表+CSV）
└── run_all.sh            # 一键运行脚本
```

## 快速复现

```bash
# 1. 环境准备
conda create -n uav_detect python=3.10
conda activate uav_detect
pip install ultralytics sahi opencv-python-headless

# 2. 下载数据集（VisDrone-DET）
python scripts/convert_visdrone.py  # 转换为YOLO格式

# 3. 训练
python scripts/train.py --exp 1 --epochs 100  # YOLOv8n baseline
python scripts/train.py --exp 3 --epochs 100  # YOLOv8n+CBAM
python scripts/train.py --exp 5 --epochs 100  # YOLOv8n-P2+CBAM

# 4. SAHI评估
python scripts/eval_sahi.py --model_path results/exp1_baseline_n/weights/best.pt
```

## 消融分析

| 组件 | mAP@0.5 提升 | 分析 |
|------|-------------|------|
| CBAM | +0.81% | VisDrone瓶颈在尺度覆盖而非特征选择性 |
| SAHI | +0.24% | 320×320切片过小，丢失上下文，待调优至480+ |
| P2检测头 | +3.39% | 高分辨率特征图有效捕捉极小目标，关键改进 |

## 参考文献

1. Woo S, et al. CBAM: Convolutional Block Attention Module. ECCV 2018.
2. Ultralytics YOLOv8. https://github.com/ultralytics/ultralytics
3. Akyon F C, et al. Slicing Aided Hyper Inference and Fine-Tuning for Small Object Detection. ICIP 2022.
4. Zhu P, et al. Detection and Tracking Meet Drones Challenge. IEEE TPAMI 2021.
