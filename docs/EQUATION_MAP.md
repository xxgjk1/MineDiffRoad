# 公式与源码映射

对应所附40页论文，不使用旧稿的环带定义；监督集合汇总见表B.6。

| 文稿标签／模块 | 当前源码 |
| --- | --- |
| full SegFormer-B0 | backbones.py：MiTB0.forward_stages，四阶段均参与 |
| eq:quality | model.py：SegmentationPath.forward，联合输入及点支持门控 |
| eq:quality_targets | data.py：degrade，不读取完整道路标注 |
| eq:fusion | model.py：加权后拼接，再经fusion |
| eq:ring | supervision.py：build_states，D25(M⁺)减去M⁺及B_out |
| eq:negative_admission | build_states：stable_background、admitted、provisional_unknown |
| eq:unknown_area_cap | supervision.py：cap_unknown；data.py：original_area_weights |
| eq:complete_negative_set | build_states：exterior ∪ admitted ∪ restored |
| eq:positive_admission | build_states：面积限制后在unknown中选candidates |
| eq:valid_set | build_states：valid及停止梯度的target |
| eq:diffusion / eq:reconstruction | model.py：ReconstructionBranch |
| eq:residual | model.py：有界残差及末层零初始化 |
| eq:loss | losses.py：total_loss，engine.py：train_step |
| 保比例几何与标定 | data.py：aligned_geometry |
| 最近深度及4×4聚合 | geometry.py：project、aggregate |
| 固定配额 | supervision.py：fixed_quota_selection |
| MR / EP / FNS / BR | metrics.py：recovery_metrics、selection_diagnostics |
| 配对t及交叉bootstrap | statistics.py：paired_iou_intervals |

监督像素计数在训练steps.csv中分别记录准入环带、外部背景、面积恢复区域与未知区域。[实现选择](IMPLEMENTATION_NOTES.md)说明论文未充分指定的部分。
