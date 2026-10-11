# 公式与源码映射

以下名称对应最新主文的 LaTeX equation label。

| 文稿公式/模块 | 源码位置 |
|---|---|
| eq:quality：独立头、联合输入、点支持门控 | `model.py / SegmentationPath.forward` |
| eq:quality_targets：退化质量代理 | `data.py / degrade` |
| eq:fusion：相对可靠性融合 | `model.py / SegmentationPath.forward` |
| eq:ring：D25\D11 环带 | `supervision.py / build_states` |
| eq:negative_admission：两教师稳定背景及双模态 AND | 同上 |
| eq:positive_admission：稳定正候选及模态 OR | 同上 |
| eq:valid_set：K 与 detached YK | 同上，`States` |
| eq:diffusion：余弦前向噪声 | `model.py / ReconstructionBranch.forward` |
| eq:reconstruction：噪声预测与 x0 恢复 | 同上 |
| eq:residual：置信度门控有界修正 | `model.py / SegmentationPath.forward` |
| eq:loss：总目标与 batch gate | `losses.py / total_loss`，`engine.py / train_step` |
| class-balanced BCE + Dice | `losses.py / balanced_segmentation` |
| 有效边界、有效骨架 | `losses.py / structural_losses`、`skeleton` |
| 预热与 EMA 切换 | `engine.py / train`，`model.py / EMATeacher` |
| 同几何双视图 | `data.py / aligned_geometry`、`weak_view` |
| 深度可见性与 4×4 聚合 | `geometry.py / project`、`aggregate` |
| 固定配额筛选原语 | `supervision.py / fixed_quota_selection` |

边界/骨架损失的具体选择，以及文稿不充分确定的细节，统一以 `IMPLEMENTATION_NOTES.md` 为准。训练参数和权重保存在 checkpoint 内，推理从 checkpoint 恢复配置，不依赖另一个可能变动的 YAML。
