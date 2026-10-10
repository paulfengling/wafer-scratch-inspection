# Wafer scratch inspection · YOLO11 + OpenCV

面向 1096 × 1096 wafer 图片与已有 YOLO detect 矩形框标注：先由 OpenCV 检测白色圆划痕／圆弧，命中图片保存到 `rule-based/`；未命中图片再用 YOLO11 检测整图及重叠切图。数据前处理、训练、推理、报废 wafer 级评估均提供命令行入口。

当前是可运行的算法基线。已验证合成几何、标签转换、路由和训练调用；尚未接入真实数据、训练权重或生产召回率测试。圆弧阈值设为 30°，实际能否检出仍取决于对比度、曲率、背景结构和参数，不能承诺所有 ≥30° 的圆弧均检出。

## 1. 安装

建议 Python 3.11。在项目根目录执行以下 PowerShell 命令。训练环境也能完成前处理、规则检测和全部测试：

```powershell
py -3.11 -m venv .venv-train
.\.venv-train\Scripts\python.exe -m pip install --upgrade pip
.\.venv-train\Scripts\python.exe -m pip install -r requirements-train.txt
```

GPU 训练需要适合本机 CUDA 的 PyTorch；安装方法见 [PyTorch 官方安装器](https://pytorch.org/get-started/locally/)。可检查设备：

```powershell
.\.venv-train\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

仅运行前处理、规则检测和测试时，可建立 `.venv` 并安装 `requirements.txt`，无需 PyTorch／Ultralytics。两个 requirements 分别使用 `opencv-python` 与 `opencv-python-headless`，请使用独立环境，避免两个 OpenCV 包混装。下文统一使用训练环境的 Python；Linux 可替换为 `python`。

## 2. 输入数据与分组

支持此目录及其子目录，图片与 `.txt` 文件相对路径、文件名主体一致：

```text
original_dataset/
  data.yaml             # names 按原标注类别 ID 排列，可保留现有 8–9 类
  images/
    wafer_001.png
    wafer_002.png
  labels/
    wafer_001.txt
    wafer_002.txt
```

每个标签行是 `class_id x_center y_center width height`，后四项为相对于原图尺寸的归一化值。1096 分辨率上的已有归一化标注直接输入，无需先缩放或重标。正常图片需要同名空标签文件；缺失标签默认报错。原图不得是 16 bit；如有这类数据，先明确灰度转换规则，再将图像和标注一起校验。

先划分原始 wafer，再切图。同一 wafer 的多次拍摄、视角、派生图片应使用同一个 `group_id`；如果希望评估新批次泛化，可把整个生产批次设为一个 group。分组数量较少时比例只能近似满足。

建立 `groups.csv`，`path` 相对于 `images/`，覆盖所有输入图片：

```csv
path,group_id
wafer_001.png,lot_001
wafer_002.png,lot_002
```

默认按 group 随机划分约 70% train、20% val、10% test，seed=42。未提供 CSV 时只能按图片名称主体分组，无法自动识别同一 wafer 的不同拍摄，务必检查。字节完全相同的图片即使被改名，也不能跨 split；不同编码、相近画面的重复图片仍需通过 group 处理。

已经有可信 train／val／test 划分时，使用 `splits.csv` 保留原划分。比如输入图片相对路径是 `train/a.png`：

```csv
path,split,group_id
train/a.png,train,wafer_a
val/b.png,val,wafer_b
test/c.png,test,wafer_c
```

## 3. 数据前处理

将下面的 `D:\wafer\original_dataset` 替换为实际数据目录。输出目录必须为空或不存在：

```powershell
.\.venv-train\Scripts\python.exe -m wafer_inspection.prepare_dataset --source "D:\wafer\original_dataset" --output .\datasets\prepared --group-csv "D:\wafer\groups.csv" --tile-size 640 --overlap 0.25
```

保留已有划分时，替换分组参数：

```powershell
.\.venv-train\Scripts\python.exe -m wafer_inspection.prepare_dataset --source "D:\wafer\original_dataset" --output .\datasets\prepared --split-manifest "D:\wafer\splits.csv" --tile-size 640 --overlap 0.25
```

也可以修改 `dataset_config.example.json` 后传 `--config`。找不到源 `data.yaml` 时用 `--data-yaml` 指定，或者 `--names` 指定原类别名称 JSON 文件。程序不猜类别 ID。

默认保留所有原类别。若只训练两种报废划痕，读取自己的 `data.yaml` 后用 `--classes` 指定原 ID，例如 `--classes 2,5` **仅在原 ID 确实为 2、5 时使用**。输出会将它们重映射为 0、1；旧的其他类别图片可成为此任务背景，前提是两种划痕没有漏标。映射记录在 `summary.json`。

前处理规则：

| 项目 | 行为 |
| --- | --- |
| 整图 | 保留原始图像字节与尺寸；标签转成输出类别 ID |
| 切图 | 仅 train 切图；640 × 640，名义重叠率 25% |
| 1096 边缘 | 起点为 0、456，形成 4 个窗口；末端对齐增加实际重叠 |
| val／test | 只保留原图，避免切图泄漏和重复计数 |
| 完整落在切图内的框 | 平移、归一化后保留 |
| 与切图边界相交的任务框 | 默认跳过整张切图，避免产生错误正样本或漏标背景 |
| 极细框 | 默认最小宽／高 1 px；小于阈值的相交目标使整切图跳过并记录 |
| 背景切图 | 默认确定性保留 25%；完整背景图保留 |

**大圆弧矩形框不能直接裁框当成正确标签。** 框内部包含大量黑色区域，裁后的矩形可能没有划痕。`--tile-label-mode clip` 会产生裁剪候选，但全部隔离在 `review/`，不会进入 `data.yaml`。需要人工根据真实划痕重新确认、修正标签，再作为独立训练样本导入；不要直接批量复制。大圆框默认仍通过整图参加训练。检查 `skipped_tiles`，若大量圆弧只能用整图训练，应逐渐补充更贴合局部圆弧的框或像素标注。

输出：

```text
datasets/prepared/
  data.yaml              # 绝对 path；JSON 语法也是合法 YAML
  images/{train,val,test}/
  labels/{train,val,test}/
  manifest.csv           # 每张图对应原图、group、split、窗口坐标
  audit.json             # 被跳过／需复核的框、重复源图片、孤立标签
  summary.json           # 类别映射、各 split 数量、参数与源数据哈希
  review/                # 仅 clip 模式，始终排除在训练外
```

移动 prepared 目录后需修改 `data.yaml` 中的绝对 `path`。前处理会拒绝非空输出，使用新的目录保留各版数据。

## 4. YOLO11 训练

先审计类别、归一化框、缺失标签、重复图片与数据内容哈希，打印实际训练参数，不下载权重、不启动训练：

```powershell
.\.venv-train\Scripts\python.exe -m wafer_inspection.train --data .\datasets\prepared\data.yaml --config .\training_config.json --dry-run
```

正式训练，验证集用于选择 best.pt；`--evaluate-test` 在训练结束后评估一次保留的测试集：

```powershell
.\.venv-train\Scripts\python.exe -m wafer_inspection.train --data .\datasets\prepared\data.yaml --config .\training_config.json --device 0 --workers 0 --name wafer-yolo11-v1 --evaluate-test
```

| 默认参数 | 用途 |
| --- | --- |
| yolo11s.pt | 初始 detect 模型；首次实际训练可能下载预训练权重 |
| imgsz=1120 | 32 的倍数，接近原图 1096；整图与切图统一模型输入尺寸 |
| epochs=150，patience=40 | 起始训练预算，按验证集决定是否调整 |
| batch=-1 | 单 GPU 自动 batch；显存不足可显式 `--batch 2` |
| mosaic=0.2，最后 20 epochs 关闭 | 降低大圆被强拼接破坏的程度；可在验证集比较 `--mosaic 0` |
| mixup=0，perspective=0，rotate／flip=0 | 保守起点，按采集条件再增加增强 |
| hsv_v=0.1，scale=0.1 | 小幅亮度／尺度增强，需验证不淹没弱划痕 |

这是**一个模型**同时学习完整 wafer 与局部切图。640 裁片在训练时放大到 1120，并非按 640 输入训练；推理切图输入也默认 1120。若需要区分两者贡献，可在相同原始 split 上比较整图训练、整图＋切图训练，再测最终级联 wafer 召回率。

Windows 示例使用 `--workers 0` 避免 DataLoader 子进程问题。CPU 可设 `--device cpu --batch 1`，但正式训练速度取决于硬件。当前训练封装仅支持单设备，避免多进程同时写数据快照。仅做短暂流程试跑时可用 `--epochs 3 --close-mosaic 0`。已有标准 YOLO11 detect 权重可用 `--model D:\models\best.pt` 开始新微调。

训练输出位于 `runs/train/<name>/`，包括 `weights/best.pt`、`weights/last.pt` 和训练图表。`dataset_snapshot.json` 在第一次训练迭代前保存图像／标签内容哈希；完整调用和结果保存到 `runs/train/_metadata/.../training_run.json`，其中记录实际 Ultralytics 版本。

中断续训：

```powershell
.\.venv-train\Scripts\python.exe -m wafer_inspection.train --data .\datasets\prepared\data.yaml --resume .\runs\train\wafer-yolo11-v1\weights\last.pt --device 0
```

续训恢复 checkpoint 的训练配置、优化器和 epoch，只允许设备覆盖，不能附加 `--config`、`--epochs`、`--batch` 等新训练参数。同路径的数据内容变化也会被拒绝。需保留原 run 下的 `dataset_snapshot.json`；外部权重没有此文件或训练已完成／优化器已移除时，用 `--model` 开始新微调。

训练 API 依据 [YOLO11 官方文档](https://docs.ultralytics.com/models/yolo11/) 和 [训练参数文档](https://docs.ultralytics.com/modes/train/)。本项目尚未进行真实 GPU 训练；CPU 自动测试用 fake model 检查调用和错误处理。

## 5. 规则检测 → YOLO 二次检测

接入训练好的本地权重：

```powershell
.\.venv-train\Scripts\python.exe -m wafer_inspection --source .\images --output .\outputs\run01 --weights .\runs\train\wafer-yolo11-v1\weights\best.pt --rule-config .\rule_config.json --tile-size 640 --tile-overlap 0.25 --tile-imgsz 1120 --confidence 0.15
```

这保留了原有 `--weights .\best.pt` 用法，也可将 weights 换成自己的现有模型。YOLO 对规则阴性图片执行 1 次整图＋4 次局部切图检测，恢复到原图坐标后按类别 NMS 合并。低置信度 0.15 是召回优先的起点，要用验证集同时观察误报。缺陷名严格、唯一匹配 `circle scratch`／`line scratch`（忽略大小写、空格、下划线、连字符），自动读取实际模型 ID；其他命名用 `--critical-classes` 指定输出模型 ID。

| 目录 | 意义 |
| --- | --- |
| rule-based | 规则命中 circle scratch，已按报废目标检出 |
| yolo-detected | 规则未命中，YOLO 检出任意类别；是否报废看 decision／critical_reject |
| undetected | 本次 YOLO 无检出；不等同于质量保证 |
| yolo-pending | 无权重时规则阴性，等待 YOLO 检测 |
| errors | 读取／处理异常，需复核 |

结果提供原图和 JSON records，成功读取的图片有 overlay；规则命中的圆划痕另外生成 crops，YOLO 框当前在 overlay／records 中保存。读取失败无法生成 overlay。`summary.json` 汇总路由。输出必须为新目录。无需 YOLO 时不传 `--weights`，但阴性图片进入 pending，不能视为正常。

默认命中 circle 的图片直接跳过 YOLO，符合“先规则、后模型”的报废筛选流程。因此同一图的 line scratch 可能不单独记录。若需要完整分类统计，增加 `--yolo-on-rule-hit`。

规则检测自动定位 wafer，结合亮细纹增强、候选圆拟合、圆心局部搜索、极坐标角度／半径支持和几何复核，区分弧、直线、散点与 wafer 边缘。`rule_config.json` 的主要参数：

| 参数 | 默认 | 调整依据 |
| --- | --- | --- |
| min_arc_deg | 30 | 每段连续圆弧的最小角度；远隔的短弧不会累加成 30° |
| max_center_offset_px | 48 | 相对估计 wafer 圆心的搜索／偏移上限；偏移未知时先校准 |
| angular_step_deg | 0.5 | 极坐标角分辨率 |
| radial_tolerance_px | 2.5 | 支持点半径容差 |
| max_radial_residual_px | 2 | 拟合残差上限 |
| max_gap_deg | 2 | 可容忍的内部小断裂 |
| min_arc_coverage | 0.72 | 候选圆弧上的有效支持比例 |
| edge_margin_px | 8 | 边缘排除宽度，需检查靠近 wafer 边缘的真实划痕 |

偏移较大、短且曲率很小的圆弧、极弱反差或复杂纹理可能仍漏检，交由 YOLO 兜底。规则分数是几何评分而非概率。圆心搜索、边缘排除与亮纹阈值需通过真实样本检查后确定，不要只凭合成结果放行。

## 6. 评估与验证

最终指标应对**保留原图**运行完整级联，再按 wafer 计数；一条大圆分成几个标注框仍只算一个报废 wafer。例：

```powershell
.\.venv-train\Scripts\python.exe -m wafer_inspection --source .\datasets\prepared\images\test --output .\outputs\test-v1 --weights .\runs\train\wafer-yolo11-v1\weights\best.pt --rule-config .\rule_config.json --tile-size 640
.\.venv-train\Scripts\python.exe -m wafer_inspection.evaluate --run .\outputs\test-v1 --labels .\datasets\prepared\labels\test
```

有模型名称时自动解析关键类别；否则传 `--circle-id`、`--line-id`，与**这份输出标签**的 ID 一致。只训练两类后不能沿用原 8–9 类的 ID。输出 `wafer_metrics.json` 包含报废 wafer 召回率、无关键缺陷 wafer 的误拒率、circle-only／line-only／mixed 子组和漏检清单。YOLO 单独的 mAP／框召回率用于辅助诊断，不替代 wafer 级指标。

圆弧角度、真实曲率及同一 circle 的实例归属无法从现有矩形框可靠恢复；要评估“≥30° 圆弧召回”，需另外记录角度／圆心／半径或像素级标注。调规则阈值、置信度与增强使用 train／val，确定配置后再使用 test。

运行 CPU 自动验证（合成场景＋fake YOLO，无权重下载）：

```powershell
.\.venv-train\Scripts\python.exe -m unittest discover -s tests -v
```

可生成 10 张 1096 合成 wafer 场景并试规则路由：

```powershell
.\.venv-train\Scripts\python.exe -m tests.fixtures .\synthetic_demo\input
.\.venv-train\Scripts\python.exe -m wafer_inspection --source .\synthetic_demo\input --output .\outputs\synthetic-v1 --rule-config .\rule_config.json
```

仓库只包含代码、配置与小型测试生成器。`.gitignore` 排除 `.venv*`、`datasets/`、`runs/`、`outputs/`、`synthetic_demo/` 和模型权重。真实数据、权重和训练结果按自己的存储方式管理。
