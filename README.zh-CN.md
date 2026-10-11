# MineDiffRoad

### 面向保守标注矿区道路分割的可靠性引导监督准入方法

**论文题目：** *MineDiffRoad: Supervision Admission Guided by Reliability for Mine Road Segmentation with Conservative Annotations*

[English](README.md) · [论文完整结果表](docs/PAPER_RESULTS.md) · [全部论文配图](docs/FIGURES.md) · [数据格式](docs/DATA_FORMAT.md) · [实现细节](docs/IMPLEMENTATION_NOTES.md)

![MineDiffRoad 总体架构](docs/figures/architecture.png)

*论文图 2：RGB／LiDAR 编码、可靠性引导融合、监督准入与道路恢复。重建分支和 EMA 教师用于训练，部署时保留分割路径与有界残差，实现单次前向预测。点击图片可查看原尺寸。*

## 研究问题

保守道路标注只标记可以确认的道路核心，未标注区域同时包含背景和遗漏道路。若将所有未标注像素视为背景，会抑制真实道路；若全部忽略，则缺少限制错误扩张的负监督。

MineDiffRoad 将局部观测可靠性用于监督位置选择。稳定的类别预测仍可能来自模糊图像或稀疏点云，因此需要区分“模型确信某个类别”和“当前位置有足够的观测证据”。

![RGB、保守核心、遗漏道路与完整参考](docs/figures/problem.png)

*论文图 1：完整参考用于评价和离线展示，不作为保守标注训练的完整监督。*

## 方法要点

1. **多模态对齐与可靠性估计。** 通过真实标定投影，将 RGB 和 LiDAR 特征对齐至 128 × 128 网格；利用受控退化代理学习各模态的局部观测可靠性。无点支持位置的有效点云可靠性为零。
2. **严格负监督准入。** 在核心周围的几何环带内，仅接纳双视图教师稳定预测为背景、且两种模态均可靠的位置，其余位置保持未知。
3. **非对称正候选恢复。** 在未知区域内，教师稳定预测为道路且至少一种模态可靠时，形成软正候选，不直接当作可信标注。
4. **有效域重建与有界细化。** 仅在可信监督和候选构成的有效域内施加时间条件重建及结构损失，通过有界残差约束预测修正。

| 监督类型 | 预热后的准入条件 | 目标 |
|---|---|---|
| 可信正样本 $M^+$ | 保守标注核心 | 1 |
| 准入负样本 $M^-$ | 环带内；两教师道路概率均 < 0.10；两模态可靠性均 > 0.50 | 0 |
| 恢复候选 $Q^+$ | 未知域内；两教师道路概率均 > 0.90；差值 < 0.05；至少一模态可靠性 > 0.50 | 两教师概率的停止梯度均值 |
| 剩余未知区域 | 未满足准入条件 | 不提供目标监督 |

环带使用 25 与 11 的膨胀核宽度，二者不是半径。有效目标集合为 $K=M^+\cup M^-\cup Q^+$；集合外的 0.5 仅为输入填充值，不成为训练标签。

![三态监督与离线错误审计](docs/figures/supervision_states.png)

*论文图 3：可信核心、候选环带、稳定背景、准入负样本与未知区域。完整参考仅用于最后一栏的离线审计。代码通过输入的 valid 掩膜定义有效图像区域。*

有界残差为：

$$
s_f=s_b+0.1(1-|2P_b-1|)\tanh(h(F_c)),\qquad P_f=\sigma(s_f).
$$

单次前向的 logit 修正不超过 0.1，概率修正不超过 0.025；这并非对累计训练影响的限制。推理无需迭代扩散采样。

## 论文核心结果

以下数值来自论文。IoU 为道路交并比，MR 为遗漏道路召回率，EP 为核心外扩张精度；单位均为 %，↑ 表示越高越好。主实验按五个种子汇报，先在各个种子内汇总测试像素计数计算 IoU，再跨种子平均。

| 方法 | IoU ↑ | MR ↑ | EP ↑ |
|---|---:|---:|---:|
| UniMatch V2，协议匹配 | 65.62 | 60.48 | 51.83 |
| CSL 自适应选择 | 65.94 | 63.56 | 52.42 |
| CSL，SegFormer-B5 较大图像骨干 | 66.73 | 67.84 | 53.16 |
| Concat，相同架构 | 63.58 | 71.42 | 46.93 |
| Conf，相同架构 | 64.94 | 73.68 | 50.87 |
| 仅可靠性选择 | 67.61 | 76.84 | 53.73 |
| **MineDiffRoad** | **68.57** | **79.13** | **55.12** |

此处节选论文表 1。MineDiffRoad 相比相同架构 Conf 的 IoU 提高 **3.63 个百分点**。完整标注参考模型的 IoU／MR／EP 分别为 69.21／80.42／55.79，其监督和辅助更新日程与保守标注实验不同。完整方法列表见 [英文首页](README.md#overall-comparison--table-1) 和 [完整结果表](docs/PAPER_RESULTS.md)。

### 固定负样本配额的选择机制

固定教师输出、候选域、正候选和负样本数量，每种方法选择 60% 的候选。FNS 是将遗漏道路选作负样本的比例，↓ 表示越低越好。

| 负样本排序 | FNS ↓ | IoU ↑ | MR ↑ | EP ↑ |
|---|---:|---:|---:|---:|
| 随机 | 4.62 | 65.44 | 70.83 | 47.92 |
| 置信度 | 3.81 | 66.14 | 72.61 | 49.84 |
| MC 不确定性，8 次前向 | 3.18 | 66.91 | 74.83 | 51.42 |
| **可靠性** | **2.91** | **68.76** | **78.21** | **54.13** |

此处为论文表 2(a)。固定轨迹实验的 **68.76%** 与在线训练主实验的 **68.57%** 对应不同协议。固定配额下，可靠性相对置信度排序提高 2.62 个百分点 IoU。析因实验报告的监督选择效应为 2.81 个百分点，特征融合效应为 1.03 个百分点。

![固定配额下的负样本选择](docs/figures/negative_selection.png)

*论文图 7：每个选择器从 682 个候选中选择 409 个。紫色为通过完整参考离线识别的错误负监督。*

### 恢复模块消融

| 恢复规则 | IoU ↑ | EP ↑ |
|---|---:|---:|
| 硬候选直接监督 | 67.09 | 50.67 |
| 软候选直接监督 | 67.37 | 52.27 |
| **有效域重建 + 有界残差** | **68.76** | **54.13** |
| 有效域重建，残差无界 | 67.59 | 51.47 |

论文表 3(a) 固定教师输出、候选、辅助更新步数与容量。更多时间条件、噪声层级、跨矿区、RELLIS-3D、自然退化、连通性及效率结果见 [论文完整表格](docs/PAPER_RESULTS.md)，对应 11 份 CSV 位于 [docs/results](docs/results/)。

## 定性结果与失败案例

![遗漏道路恢复与共享局部区域对比](docs/figures/road_recovery.png)

*论文图 4：深绿为可信／正确道路，浅绿为恢复道路，黄色为遗漏道路，紫色为错误扩张。各列依次为 RGB、完整参考、Concat、Conf、MineDiffRoad、完整标注参考模型。*

<details>
<summary><b>展开查看与 CSL（SegFormer-B5）的 8 组对比</b></summary>

![样本 1–4 对比](docs/figures/comparison_samples_1_4.png)

*论文图 5：RGB／完整参考／CSL（SegFormer-B5）／MineDiffRoad。*

![样本 5–8 对比](docs/figures/comparison_samples_5_8.png)

*论文图 6：列顺序与显示规则相同。*

</details>

![相似地形导致的错误道路扩张](docs/figures/failure_case.png)

*论文图 8：与道路外观相似的地形仍可能导致错误扩张。观测可靠性不能完全解决语义歧义。论文还报告了较大范围端点连通任务上的错误，具体定义与取舍见完整结果表。*

## 安装与运行

### 1. 环境

建议 Linux、Python ≥ 3.10、支持 CUDA 的 NVIDIA GPU。在项目根目录执行，先按本机驱动／CUDA 配置安装合适的 PyTorch，再安装其余依赖。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# 此处先安装与本机兼容的 PyTorch。
python -m pip install -r requirements-cuda.txt
python -m pip install -e . --no-deps
python -m minediffroad --help
```

[requirements-tested.txt](requirements-tested.txt) 记录检查环境。CUDA 依赖文件使用 `spconv-cu124`、`cumm-cu124`；更换环境时使用相容构建，避免同时安装多种 spconv。CPU 可运行数据处理、监督和损失等检查，训练与推理使用真实点云骨干。

### 2. 数据

依据 [DATA_FORMAT.md](docs/DATA_FORMAT.md) 和 [清单示例](examples/manifest.example.jsonl) 准备真实 RGB、N×4 LiDAR、标定矩阵及掩膜。数据集和预训练权重单独准备。

- 训练记录提供保守 `core`，不能含完整 `reference`。
- 验证／测试评价使用独立完整 `reference`。
- `group` 不能跨 train／val／test，同一采集组只归属一个集合。
- 点云 XYZ 单位为米，强度范围为 [0, 1]；真实标定包括 `K` 和 `T_lidar_to_camera`。

```bash
python -m minediffroad verify-data --manifest /absolute/path/manifest.jsonl

python -m minediffroad prepare-fps \
  --manifest /absolute/path/manifest.jsonl \
  --output /absolute/path/fps_cache_new --num-points 16384 --device cuda
```

FPS 缓存步骤可选，会输出新清单、索引和元数据；后续可使用新的 `manifest.jsonl`。

### 3. 训练与续训

```bash
cp configs/paper.yaml configs/my_run.yaml
```

修改 `configs/my_run.yaml` 的 `imagenet_checkpoint`，填写 [NVIDIA SegFormer](https://github.com/NVlabs/SegFormer) 发布的 MiT-B0 ImageNet 权重路径，可同时填写 `imagenet_sha256`。默认 512×512、16,384 点、80 epoch、batch size 8，前 10 epoch 环带预热，第 11 epoch 初始化 EMA=0.99 的教师。默认学习率和权重衰减为论文搜索空间中的一个候选组合，最终设置应由验证集选择并记录。

```bash
python -m minediffroad train \
  --config configs/my_run.yaml \
  --manifest /absolute/path/fps_cache_new/manifest.jsonl \
  --output runs/seed0_new

python -m minediffroad train \
  --config runs/seed0_new/config.yaml \
  --manifest /absolute/path/fps_cache_new/manifest.jsonl \
  --resume runs/seed0_new/epoch_040.pt \
  --output runs/seed0_resume_new
```

输出包括配置、环境、训练日志、`steps.csv`、每个 epoch 的 checkpoint、`checkpoint_index.json`，以及存在验证集时的逐图和汇总指标。通过 `checkpoint_index.json` 读取验证集选出的 epoch。续训要求配置和清单一致，并使用新输出目录。

显存不足时可降低 `batch_size`，或启用 `empty_cache_each_epoch`。输入分辨率／点数的变化需要在实验比较时注明。

### 4. 评估、推理与导出

下列 epoch 编号仅为命令示例，实际使用验证集选出的 checkpoint。

```bash
python -m minediffroad evaluate \
  --checkpoint runs/seed0_new/epoch_080.pt \
  --manifest /absolute/path/manifest.jsonl --split test \
  --output runs/evaluation_new --device cuda

python -m minediffroad predict \
  --checkpoint runs/seed0_new/epoch_080.pt \
  --manifest /absolute/path/manifest.jsonl --split test \
  --output runs/predictions_new --device cuda --save-probabilities

python -m minediffroad export \
  --checkpoint runs/seed0_new/epoch_080.pt \
  --output checkpoints/deployment_new.pt
```

| 操作 | 输出 |
|---|---|
| 评估 | `per_image.csv`、`metrics.json`；分别记录全局混淆矩阵指标与逐图均值 |
| 推理 | 二值 PNG、可选概率 `.npy`、帧 ID 映射 `index.json` |
| 导出 | 仅含分割路径的 PyTorch 权重，可作为 `predict --checkpoint` 输入 |

CLI 评价指标包括 Road IoU、二类 mIoU、Dice、Precision、Recall、Accuracy、Boundary F1。论文的 MR／EP、FNS／BR、连通性审计与固定轨迹实验使用各自的评价记录和流程，不属于当前 CLI 的直接输出。默认阈值 0.50，移除小于 50 像素的 8 连通分量。预测位于配置的输入尺寸；无标注预测可使用 `--split predict`。

### 5. 软件检查

```bash
python -m unittest discover -s tests -v

MINEDIFFROAD_TEST_CUDA=1 MINEDIFFROAD_TEST_ROUNDTRIP=1 \
  python -m unittest discover -s tests -v
```

检查环境与覆盖范围见 [SOFTWARE_VALIDATION.md](docs/SOFTWARE_VALIDATION.md)。

## 文件导航

| 路径 | 内容 |
|---|---|
| `minediffroad/` | 编码、投影、可靠性、监督、重建、训练与推理 |
| `configs/paper.yaml` | 方法配置 |
| `docs/EQUATION_MAP.md` | 论文公式与代码对应 |
| `docs/IMPLEMENTATION_NOTES.md` | 模块接口与实现选择 |
| `docs/PAPER_RESULTS.md` | 论文表 1–6 及实验协议说明 |
| `docs/figures/` | 论文 8 张图 |
| `docs/results/` | 11 份论文表格 CSV |
| `docs/ASSET_PROVENANCE.json` | 图表来源、PDF 页码与文件哈希 |
| `tests/` | 软件检查 |
| `third_party/` | LitePT 来源及许可证 |

源码采用 [MIT 许可证](LICENSE)。LitePT 的上游版权与许可证保留在 [third_party](third_party/README.md)。图表来自所提供论文，数据集和外部权重遵循各自发布方条款。
