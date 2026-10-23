# 数据与坐标约定

使用一个 JSONL 清单显式列出 train/val/test（或 predict）集合。每行一个帧，路径相对于清单所在目录解析，也接受绝对路径。先复制 `examples/manifest.example.jsonl` 再替换占位文件路径。

所附论文使用的 SegMine 工作子集包含 12 个采集组、26 个场景、749 个 RGB–LiDAR 配对帧。其清单预期规模如下，不能把这些数字解释为 SegMine 其他数据产品的规模。另有 400 帧原始人工保守标注数据，属于独立设置，不计入下表。

| 划分 | 采集组 | 场景 | 帧 |
| --- | ---: | ---: | ---: |
| 训练 | 8 | 17 | 524 |
| 验证 | 2 | 3 | 75 |
| 测试 | 2 | 6 | 150 |
| 合计 | 12 | 26 | 749 |

`python -m minediffroad verify-data --manifest data/manifest.jsonl --paper-segmine` 对实际清单检查上述数量及场景、采集组跨划分泄漏。该选项只适用于这份 SegMine 工作子集；通用的 `verify-data` 不强制其他数据集使用这些数量。仓库不随附该数据清单，因此文档数值本身不是对实际文件的核验结果。

如需核实论文中 $Y^e=Y^c$ 的前提，验证/测试记录应同时提供 `core`、`evaluation_core` 和 `reference`，然后运行 `python -m minediffroad verify-data --manifest data/manifest.jsonl --paper-segmine --paper-core-equality --output core_audit.json`。命令在原图尺寸上逐像素比较两份二值核心，记录每帧掩膜摘要；若有缺失或不一致则报错。它不读取训练记录中的完整标注，也不把评价标注用于训练。尚未提供实际清单时，不能仅凭字段定义声称该一致性已得到验证。

自然退化实验的40个帧ID可保存为JSON字符串数组，再运行 `python -m minediffroad verify-data --manifest data/manifest.jsonl --natural-degradation-ids data/natural40.json --output natural40_audit.json`。检查要求40帧全部出现在同一清单的独立测试划分，并记录场景、采集组与ID集合摘要；若包含训练或验证帧则报错。实际帧清单必须来自实验记录，不能用文档中的数量替代。

| 字段 | 要求 |
|---|---|
| id | 字符串，全清单唯一，与输出 index.json 对应 |
| group | 字符串，采集序列/场景/路段分组标识；同组只能属于一个 split |
| scene | 预先定义的统计重采样场景标识，不默认复用group；场景与采集组不强制一一对应 |
| split | train、val、test、predict 之一 |
| rgb | RGB 图片路径 |
| points | N×4 `.npy`；XYZ 米，反射强度 [0,1]；无 NaN/Inf |
| calibration | JSON 标定路径，包含 K、T_lidar_to_camera |
| core | train 必须提供，保守二值道路标注；如要核对论文的评价掩膜等价条件，val/test 也需提供同源核心 |
| reference | 仅验证/测试评价使用，train 记录禁止出现 |
| evaluation_core | 可选，冻结的评价核心，仅用于val/test计算MR/EP；论文核心一致性审计要求其与同帧 core 逐像素相同；train禁止提供，不能在训练时从reference生成 |
| valid | 可选，图像有效域；缺省为原图全有效 |
| fps_indices | 可选，原始点云的无重复索引，长度 min(N,num_points) |
| points_sha256 | prepare-fps 自动生成，用于检查缓存对应的原始点云是否变化 |

所有掩膜必须与原始 RGB 同尺寸，使用单通道 0/1 或 0/255；多类语义图必须在独立数据准备步骤中映射为二值道路图。`verify-data` 检查清单字段、重复 ID、文件存在及分组隔离；实际加载还检查掩膜尺寸/取值、点云形状/有限性和标定矩阵尺寸。

标定 JSON 的真实数值需来自数据集。可参照 `calibration.schema.json` 的形状约定；该文件是 schema，不能直接作为相机参数。

```text
[X_camera, Y_camera, Z_camera, 1]^T
    = T_lidar_to_camera @ [X_lidar, Y_lidar, Z_lidar, 1]^T

[u', v', w']^T = K @ [X_camera, Y_camera, Z_camera]^T
u = u'/w', v = v'/w'
```

相机轴方向：x 向右、y 向下、z 向前；图像 u 对应列、v 对应行。输入 RGB 应已去畸变，或者点云与 RGB 已处于同一相机模型；本代码不猜测畸变参数，不估算外参，不从图像推造点云。图像变换同步左乘 K，不对物理 LiDAR 坐标做不匹配的水平翻转。

训练仅访问原始 RGB、真实 LiDAR、标定、保守 core、可选 valid。验证/测试完整标注只能在各自记录中出现，不能把 val/test 的 core/reference 作为训练记录输入。采集组隔离优先于逐帧随机切分。

predict split 可以省略 core 和 reference，只需传感器输入与标定。同一评价批次的reference/evaluation_core字段必须齐全一致，不能混入缺失参考的记录。缺少evaluation_core时不报告MR/EP，缺少scene时不允许计算场景bootstrap。

默认将原图保比例缩放后中心补边至1024×1024，resize_mode=crop则保比例缩放后中心裁剪。训练使用共享尺度扰动、裁剪／补边和翻转，RGB与所有掩膜一致。K同步乘像素中心变换H，不拉伸非方形原图。补边Omega=0，先排除补边再进行连通分量后处理。预测PNG处于1024网格，不可直接解释为原图坐标。

训练的area_weights记录原始有效像素中心在工作网格中的归属。30%未知区域上限以原始有效图像面积为分母，不是环带或补边后的正方形面积。裁掉的原图像素保留在分母中，但不贡献未知区域面积。掩膜按最近邻映射，RGB按双线性插值。
