UAV无人机视觉目标检测 - 课程项目展示素材
============================================

一、实验结果汇总表
----------------------------------------------
实验      模型                   mAP@0.5   mAP@0.5:0.95  提升(vs Exp1)
Exp1      YOLOv8n baseline       0.3183    0.1746         --
Exp2      YOLOv8s baseline       0.3876    0.2202         --
Exp3      YOLOv8n+CBAM           0.3264    0.1798         +0.8%
Exp4      YOLOv8n+SAHI           0.3207    0.1766         +0.2%
Exp5      YOLOv8n-P2+CBAM        0.3522    0.1975         +3.4%

二、消融分析
----------------------------------------------
- CBAM注意力模块(Exp3): 单独加在backbone上提升有限(+0.8%)
  说明：VisDrone瓶颈不在特征表达，而在尺度覆盖
- SAHI切片推理(Exp4): 对baseline提升微小(+0.2%)
  说明：slice_size=320可能太小，丢失上下文信息
- P2小目标检测头(Exp5): 最有效改进(+3.4%)
  说明：VisDrone大量极小目标(<32px)，P2在4x下采样尺度检测是关键

三、展示素材清单
----------------------------------------------
exp1_baseline_n-2/  - YOLOv8n基线完整素材
  - results.png          训练曲线(官方)
  - confusion_matrix.png 混淆矩阵
  - confusion_matrix_normalized.png 归一化混淆矩阵
  - BoxPR_curve.png      PR曲线
  - BoxP_curve.png       Precision曲线
  - BoxR_curve.png       Recall曲线
  - BoxF1_curve.png      F1曲线
  - val_batch*_labels.jpg  验证集GT标注
  - val_batch*_pred.jpg    验证集预测结果
  - train_batch*.jpg       训练集样本
  - labels.jpg             标签分布
  - results.csv            逐epoch指标
  - sahi_visualizations/   SAHI切片推理可视化

exp2_baseline_s-2/  - YOLOv8s基线(同上结构)

exp3_cbam_v3/       - YOLOv8n+CBAM(同上结构)

exp5_cbam_p2/       - YOLOv8n-P2+CBAM
  - training_curves.png   训练曲线(从日志生成)
  - train_batch*.jpg      训练集样本
  - labels.jpg            标签分布

comparison_curves.png - 5个实验mAP/P/R/Recall对比图

四、关键数字(用于PPT)
----------------------------------------------
- VisDrone-DET: 6471 train / 548 val / 1610 test, 10类
- YOLOv8n: 3.16M params, YOLOv8s: 11.13M params
- CBAM额外参数: +11.3K (3.17M total)
- P2+CBAM: 3.37M params
- 最佳改进: P2+CBAM mAP@0.5提升3.4个百分点
