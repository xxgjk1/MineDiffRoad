# 数据与坐标约定

使用一个 JSONL 清单显式列出 train/val/test（或 predict）集合。每行一个帧，路径相对于清单所在目录解析，也接受绝对路径。先复制 `examples/manifest.example.jsonl` 再替换占位文件路径。

| 字段 | 要求 |
|---|---|
| id | 字符串，全清单唯一，与输出 index.json 对应 |
| group | 字符串，采集序列/场景/路段分组标识；同组只能属于一个 split |
| split | train、val、test、predict 之一 |
| rgb | RGB 图片路径 |
| points | N×4 `.npy`；XYZ 米，反射强度 [0,1]；无 NaN/Inf |
| calibration | JSON 标定路径，包含 K、T_lidar_to_camera |
| core | train 必须提供，保守二值道路标注 |
| reference | 仅验证/测试评价使用，train 记录禁止出现 |
| valid | 可选，图像有效域；缺省为原图全有效 |
| fps_indices | 可选，原始点云的无重复索引，长度 min(N,num_points) |
| points_sha256 | prepare-fps 自动生成，用于检查缓存对应的原始点云是否变化 |

所有掩膜必须与原始 RGB 同尺寸，使用单通道 0/1 或 0/255；多类语义图必须在独立数据准备步骤中映射为二值道路图。`verify-data` 检查清单字段、重复 ID、文件存在及分组隔离；实际加载还检查掩膜尺寸/取值、点云形状/有限性和标定矩阵尺寸。清单无法证明标注本身是否正确，原始数据仍需数据负责人核对。

标定 JSON 的真实数值需来自数据集。可参照 `calibration.schema.json` 的形状约定；该文件是 schema，不能直接作为相机参数。

```text
[X_camera, Y_camera, Z_camera, 1]^T
    = T_lidar_to_camera @ [X_lidar, Y_lidar, Z_lidar, 1]^T

[u', v', w']^T = K @ [X_camera, Y_camera, Z_camera]^T
u = u'/w', v = v'/w'
```

相机轴方向：x 向右、y 向下、z 向前；图像 u 对应列、v 对应行。输入 RGB 应已去畸变，或者点云与 RGB 已处于同一相机模型；本代码不猜测畸变参数，不估算外参，不从图像推造点云。图像变换同步左乘 K，不对物理 LiDAR 坐标做不匹配的水平翻转。

训练仅访问原始 RGB、真实 LiDAR、标定、保守 core、可选 valid。验证/测试完整标注只能在各自记录中出现，不能把 val/test 的 core/reference 作为训练记录输入。采集组隔离优先于逐帧随机切分。

predict split 可以省略 core 和 reference，只需传感器输入与标定。eval 模式统一缩放至配置的正方形输入尺寸，预测 PNG 也在该尺寸；如需要回到原图，应另行保存和逆应用变换，不能直接按原始像素坐标解释 512×512 输出。
