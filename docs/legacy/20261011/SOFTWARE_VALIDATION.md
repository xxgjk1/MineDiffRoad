# 软件验证记录

记录日期：2026-10-11。以下为源码包随附的软件检查记录，检查使用合成输入或临时文件。

环境：Python 3.10、PyTorch 2.11.0+cu128、RTX 3090、spconv-cu124 2.3.8、cumm-cu124 0.7.11。其他依赖见 `requirements-tested.txt`。

执行命令：

```bash
MINEDIFFROAD_TEST_CUDA=1 MINEDIFFROAD_TEST_ROUNDTRIP=1 \
  python -m unittest discover -s tests -v
```

21 项检查通过，覆盖：

- 环带核宽度、监督状态互斥、阈值严格比较和标签 detach。
- 未知区域对分割/结构损失的隔离，以及空集合处理。
- 标定投影、前后深度过滤、均值/最大值聚合和梯度。
- 共享裁剪/缩放/翻转后的点像素对齐。
- 固定起点 FPS、固定配额和排序并列处理。
- MiT-B0 所用阶段尺寸，以及 checkpoint 键匹配/哈希接口。
- LitePT 兼容算子的输出和梯度。
- 残差上界、无点支持门控、EMA 排除重建分支。
- 重建终点有限性、迁移目标停止梯度、条件特征可接收重建梯度。
- 训练标签访问限制、采集组隔离、随机状态保存恢复。
- 真实 LitePT-S CUDA 前后向、混合精度优化步、eval 重复前向。
- 临时合成数据上的训练文件 IO、EMA 切换、checkpoint 保存/恢复、分割路径加载与部署推理。

另外完成一次实际配置尺寸的真实骨干前向检查：512×512 RGB、16,384 个合成 LiDAR 点，输出为有限的 512×512 概率张量，融合支持尺寸为 128×128。

曾在未统一 cuDNN 确定性设置的测试中观察到重复推理数值差异；训练与部署入口现均设置相同随机种子和 cuDNN 确定性选项。仍不能承诺不同 GPU、CUDA 或 sparse kernel 构建之间逐位相同。


## GitHub 文档版检查（2026-10-11）

使用 Python 3.10 / PyTorch 2.11.0+cu128 执行 CPU 检查：

```bash
python -m unittest discover -s tests -v
```

21 项中 18 项通过；2 项真实 CUDA 骨干检查和 1 项训练／checkpoint 集成检查需显式启用，本次未启用。CLI 帮助命令通过。
