# LingBot-Depth / LingBot-Map 2：大厅数据离线实验完整计划

日期：2026-11-28（Asia/Shanghai）。状态：**计划已制定，尚无本轮模型运行结果**。

执行入口：[完整 Codex 提示词](../prompts/lingbot_depth_map_20261128.md)。机器配置：[实验协议](../configs/lingbot_depth_map_protocol.json)。

## 1. 本轮目标与交付边界

用已有 Gemini 335 大厅记录完成两项独立实验，并回答其是否值得用于后续 DN-Splatter 改进：

1. **Depth**：RGB + 已配准的传感器 Z 深度 → 补全/修正深度与单帧点云。检查补全、保留测量、细结构和跨视角一致性。
2. **Map**：按原始时间顺序的 RGB → 相机位姿、内参、预测深度、置信分数及场景点云。检查轨迹、场景一致性、尺度变化和已知困难区域。

第一轮使用预训练权重做离线推理。两项任务互不依赖，不将 Depth 输出作为 Map 的默认输入。主数据为大厅 Gemini 335，不把腕部 DC1、VLA、VA、机械臂动作或新数据采集列为前置条件。

完整交付包括：可复用运行脚本、来源与环境、固定输入清单、数值输出、逐帧指标、可读对照图、三维回放/视频、失败分析、明确的后续采用建议。仅安装、下载或运行 30 帧不等于本轮完成。30 帧和短序列是正式任务的首批，不是反复小测试；接口正确后持续推进计划中的全量离线推理。

**本轮到离线实验和候选接口为止，不自动启动新的 DN 训练，不修改已发布 B2/B3 的输入、相机和结果。**后续“只换深度”和“只换位姿”的训练对照分别设计，不能在本轮直接把两个改动一起塞入基线。

## 2. 已核实的仓库起点

本计划依据仓库提交 `1f2b48744feefe95f879bbf43c7bc58a9871d2d8` 编写。执行时再检查更新和服务器实态，但不因过时入口重做已完成工作。

- B2、B3 均已有完整 240k 结果。现有冻结选择保留 B2 210k 渲染模型，采用 B3 210k 的 supported Poisson 网格。
- B3 没有接受新的位姿改动；主要启用深度权重。RGB 小幅提升与深度平均误差改善不是同一件事。
- 主序列为 1846 帧，train/val/development-test 为 1478/184/184；另有 100 张辅助建图图像，不混入本轮单相机主序列。
- 已有 1280×720 去畸变 RGB 和对齐到彩色光轴的米制 float32 Z 深度。原始传感器深度是 848×480、每计数 1 mm，0/65535 是原始无效码。
- 原始传感器、已有 SfM/优化位姿、TSDF、Poisson 均不能自动作为独立几何真值。
- 仓库根 `reports/latest_status.json` 是历史 VR Room 状态，不能用它判断大厅 B3 或本轮 LingBot 完成。

依据：[B3 总报告](../reports/capture-full-quality-v3/20261122-v1/overview.md)、[B3 中文完整对照](../reports/capture-full-quality-v3/20261122-v1/B3_完整视频与指标对比.md)、[B3 冻结选择](../reports/capture-full-quality-v3/20261122-v1/selection_freeze.json)、[D2C 复核](../reports/capture-bag-review/2026-11-19/overview.md)、[配准实现](../scripts/capture_rectify.py)。

## 3. 模型、代码与环境

| 实验 | 固定对象 | 本次计划核查的官方代码 |
| --- | --- | --- |
| Depth | `robbyant/lingbot-depth-pretrain-vitl-14-v0.5` | `Robbyant/lingbot-depth@f3a237e434ae987bc38281476d6cfb5df3e4d739` |
| Map | `robbyant/lingbot-map-v2-base-streaming`，pure causal、无 GCR | `Robbyant/lingbot-map-v2@078b54485830c74ab1d3797667073e3e8353f3b9` |

开始时记录上游 commit、模型下载 revision、权重 SHA-256、配置 SHA-256。以表中代码作为首轮起点；若因确切兼容问题采用更新官方修复，保存前后 SHA、原因和差异，不能悄悄漂移到另一模型。

Map 发布配置中权重 SHA-256 为 `d3ae62a09d3caaeb0947c4372f6fb23a4c0e85304867fa9702948ba030310e27`。使用配套 `model_weights.pt` 和 `config.json`，保留官方下载器的校验；不能用 `causal.example.json` 或 Map 1 权重代替。Depth 代码/权重 Apache-2.0；Map 2 原创代码和发布权重按其 CC BY-NC 4.0 使用，本轮是学术实验。[O1–O7]

### 3.1 隔离与 Blackwell 适配

主机使用现有 AutoDL RTX PRO 6000 96GB / 25 vCPU / 120GB RAM，以实际 GPU 型号、compute capability、容器配额和余量为准。复用已验证驱动/Toolkit，不重装系统、不修改共享 CUDA 链接。

分别使用项目专属前缀：

- `.local/conda/lingbot-depth-bw`
- `.local/conda/lingbot-map2-bw`

Depth 当前 `pyproject.toml` 锁定 `torch==2.6.0`、`xformers==v0.0.29.post2`；Map 要求 `torch>=2.8` 和 FlashInfer，官方安装示例为 CUDA 12.8。不能把两个仓库依次安装进同一个环境，也不能降级现有 DN 训练环境。[O2, O6]

Depth 需要实际验证默认深度 masking/nested-token 路径。虽然模型提供 native SDPA 包装方法，默认 nested tokens 仍依赖 xformers，不能仅卸载 xformers 或设置 `XFORMERS_DISABLED` 就宣称完成适配；不能关闭 `enable_depth_mask` 来偷偷改变方法。优先选择实际支持当前 GPU 的 torch/xformers 配对；需要改包约束或内核时，使用独立环境、保存最小补丁，并以同权重、同输入核查数值与语义。若没有可运行参照，不声称“数值等价已验证”。[O3, O4]

Depth 跑官方例 0 一次作为入口核查，随后进入本数据；Map 的正式短序列同时承担入口核查。只在异常无法定位时追加官方示例对照，不下载整套训练数据或堆叠冒烟实验。没有依赖兼容证据前，不承诺任何特定 GPU 帧率。

## 4. 阶段 P0：一次完成输入与运行身份准备

### 4.1 定位已有资料

先确认现有仓库远端、分支、未提交工作和实际项目根目录。历史建议位置 `/root/autodl-tmp/DN-Splatter` 只是定位线索，不能覆盖同名目录。

优先读取：

1. 本计划、配套协议、当前 `AGENTS.md`。
2. B3 总报告、冻结选择、`artifacts.json`，以及 B2 的 `latest.json`。
3. `configs/capture_quality_protocol.json`、`configs/capture_surface_quality_protocol.json`、`configs/capture_full_test_protocol.json`。
4. 当前服务器 B2 的 `inputs/transforms.json`、`inputs/splits.json`、`inputs/training_manifest.json`；通过其真实字段定位 RGB、米制深度。
5. `configs/capture_full_test_protocol.json` 指向的 geometry_run/transfer 中的 `selection.json`、`calibration.json`，以及配准产物的 `grids.json`。
6. `scripts/capture_rectify.py`、`scripts/capture_quality_paths.py`、`scripts/capture_surface_submaps.py`，确认数据与像素中心约定。

已知 B2 run 为 `capture-full-quality-v2-20261121-v1`，B3 run 为 `capture-full-quality-v3-20261122-v1`。从实际 manifest 解出路径，不按猜测重提取 6GB bag、不重跑 SfM、不重做历史训练。

### 4.2 固定一个公共输入清单

建立本轮 `input_manifest.json`，每帧至少含：原始 frame ID、原始时间戳、主/辅相机身份、split、RGB 路径/哈希、深度路径/哈希、宽高、深度单位与定义、内参、像素中心约定、已有位姿及其来源哈希。保存 split、calibration、selection、baseline camera 文件身份。

以下属于一次必要的接口检查：

- 主帧数/划分符合 1846 与 1478/184/184；记录真实缺失而不静默删帧。
- D2C 深度是彩色相机坐标的 Z、单位 m；原始毫米仅在实际读取原始数据时转换。不能通过猜量级自动缩放。
- 已配准图不再二次 D2C。RGB、D、有效域、K 对应同一去畸变画布。
- 排除原始 0/65535、非有限/非正值；保留原文件。评测主距离区间继承 0.3–6 m，区间外保留并另报，不用截断伪造覆盖。
- 读取实际 `K_cv` 与 `K_colmap_ns`。已有处理代码的主点转换为 +0.5；明确下游 UV 约定，只转换一次。不能从字段名猜测需不需要再次加减 0.5。
- 在一帧真实数据上检查反投影/再投影和 RGB—D 边缘对应；错误先修输入接口，不能把错误输入产生的结果归咎模型。

本轮 RGB/深度只读引用。派生结果单独保存，不通过可写硬链接覆盖旧输入。

### 4.3 在看模型输出前确定样例

Depth 选择 30 个 train 帧：10 个较完整普通区域、10 个明显孔洞区域、10 个细杆/桌沿/遮挡边界区域；按整个时间线分布，记录每个 ID、理由、区域 ROI 和是否属于已知困难段。类别不足时记录实际数量，不编造覆盖；不能跑完后只挑好看的 30 帧。

Map 从同一主相机 train 中按原始时间顺序选择 256 帧短序列。尽量覆盖平移、转弯和重复观察；排除 val/test 后产生的时间间隔原样记录，不把它说成完整连续采集。若已知困难帧 `frame_01304`、`frame_01305`、`frame_01514` 不在此段，作为全 train 结果中的固定诊断点，先核对其真实 split 和证据，不强行改 split 纳入。

预先固定几何校准帧：在短序列前部选择最多 32 个具备可靠原始深度、无明显动态干扰的训练帧；选择依据只看原始数据，不看 Map 输出。不足时记录可用数量与不能可靠定尺度的原因。

## 5. 两线并行与执行顺序

P0 之后允许 Depth、Map 两条独立任务并行。各自环境、模型实例、日志、状态和 tmux 会话独立；一个分支遇到依赖问题，另一分支继续。一个 Map 模型实例只持有一个活动 stream。

| 阶段 | Depth 线 | Map 线 | 实际交付 |
| --- | --- | --- | --- |
| P0 | 共同输入与帧表 | 共同输入与帧表 | 来源、数据身份、样例/校准清单 |
| P1 | 官方例 0 + 自采首帧/30 帧 | 256 帧 train 短序列 | 数值输出、完整性、首批可视化 |
| P2 | 30 帧固定 mask 恢复与几何诊断；全 1478 train 无人工 mask 推理 | 独立初始化运行全部 1478 train | 完整 train 产物、逐帧指标/轨迹 |
| P3 | 固定姿态下的原始/修复深度比较 | 独立初始化运行 1846 全帧探索序列；同约定分析 | 长序列现象、对照视频、明确数据用途 |
| P4 | 汇总 Depth 结论 | 汇总 Map 结论 | 中文报告、采用建议、可恢复入口、GitHub 提交 |

首批接口正确后连续推进，不设阶段人工审批。效果差是正常实验结果，不是停止或删帧的理由。确切的运行错误先最小修复；无有效输出或资源/访问阻塞时记录真实边界，交付另一条线及可恢复工作，不能写“全部完成”。

允许使用全部已分配算力。双 GPU 进程并行前检查实际显存、RAM 和磁盘余量；资源紧张时调度 GPU 重任务，CPU 解码/统计/报告仍可并行。不要用过量 worker×内部线程制造争用，不结束其他项目进程。

## 6. Depth 执行细则

### 6.1 读取适配器

实现独立 wrapper，直接读取现有米制 NPY：

- `image`：RGB 顺序、float32 `[1,3,H,W]`、范围 `[0,1]`。不在外部重复加 ImageNet 标准化。
- `depth_in`：float32 `[1,H,W]`、米制，相同画布，无效输入为 0。
- `intrinsics`：使用当前画布对应的 `K_cv`，形状 `[1,3,3]`，水平参数按 W、垂直参数按 H 归一化一次。官方 `depth_to_pointcloud` 使用整数像素中心 `u=0…W−1, v=0…H−1`；若读取的是本仓库 DN transforms / `K_colmap_ns`，先对 `cx,cy` 各减 0.5 恢复 `K_cv`。来源已是 `K_cv` 时不再减。内参在该接口中用于点云反投影，不能代替 RGB-D 对齐。
- 三个张量显式放同一 device；加载后显式 `.eval()`，使用 inference mode。首轮 batch size 1，记录实际 token/resolution 设置，保留官方默认深度 masking。
- 调用 `MDMModel.from_pretrained(...).to(device).eval()` 和 `model.infer(image, depth_in=depth, intrinsics=K)`；用固定模型权重与默认有效性行为。[O1, O3]

示例脚本只支持 `examples/<name>/rgb.*`、毫米 `raw_depth.png`、像素内参 `intrinsics.txt`。它不是本项目 NPY 批处理接口；不要给它编造 `--depth-npy` 等不存在的参数。

保留模型输出米制深度、模型布尔 mask、有限正深度 mask。默认屏蔽区域可能为 `inf`；统计时正确处理，原始输出和原始分母仍保留。模型 mask 不解释成准确率或概率。

### 6.2 三种问题分别评测

**A. 原始有效区域的修改程度。**在预先固定的原始参考像素上，统计预测覆盖、相对原值的绝对变化/有符号变化、P50/P95；按距离、边界/内部和帧等权汇总。这里是“传感器观测保留/改变”，不是独立几何精度。

**B. 隐藏已知测量的恢复能力。**在 30 帧内固定两个实验，seed=42：

- 散点：隐藏参考有效像素的 10%。
- 连续块：64×64 像素块，累计覆盖约 20% 参考有效像素；保存精确集合与实际比例。

可靠参考集合使用原始有限 Z、0.3–6 m、有效成像域，主恢复指标排除既有深度跳变附近 2 像素，边界现象另列。prepare 阶段固定边界检测：复用 `capture_rectify.edges` 的相邻有效 Z 差阈值 `max(0.05 m, 0.03×较近Z)`，对边缘掩码扩展 2 像素，保存方法和最终集合。**连续块内全部深度输入置为无效；计分 mask 是矩形与可靠参考集合的交集。**边缘不计分也不能继续留在块内输入。散点实验的输入/计分 mask 相同。两种 mask 分别保存，且在任何自定义缩放、补洞、归一化或尺度估计之前作用于输入。目标值只进入评测端，不能泄漏回输入或其他统计。

固定一个简洁基线：只从同一个被遮挡输入的剩余有效值做二维最近邻补洞；输出规则与适用范围记录。原始缺测的 0 不是预测，不能算成巨大误差来衬托模型。设 M 为冻结计分 mask，V 为该方法输出有限、严格大于 0 且符合其实际有效性规则的预测集合：预测覆盖为 `|M∩V|/|M|`；MAE/RMSE/AbsRel/P95 仅在 `M∩V` 计算并报告支持数；5/10/20 cm 正确恢复覆盖的分母始终为全部 `|M|`，无效输出计为未正确恢复。0.3–6 m 限制用于参考集合，不能按预测值范围额外删除离群误差。另给两方法共同有效像素误差和共同支持量。`M∩V` 为空时误差为 null、覆盖为 0；帧等权汇总记录有定义的帧数与全部失败帧数。不能只用模型自己的 mask 删除困难样本后宣称整体精度提高。

**C. 真实自然孔洞与跨帧一致性。**报告补全率、预测范围、非有限比例、桌沿/椅腿是否变厚或误连接。只有独立其他视角的真实传感器测量可提供支持；处理遮挡、重复近邻和动态区域，记录所依赖的位姿。没有可靠参照的孔洞只标“未证实”，不能用模型自身结果证明自己。

用同一套冻结 B2 相机分别将 raw 和 refined 投影到相邻/重叠训练视角，记录残差、可见支持与冲突。这里受原位姿误差影响，结论叫几何一致性诊断，不标真值精度。若某段位姿已可疑，该段保留结果并加标记，不能静默去掉。

### 6.3 全 train 与显示

在 1478 train 上保存未经人工 mask 的完整推理结果，每帧有身份、状态和输出哈希。30 帧诊断是这批固定输入的子集，可复用相同配置的无 mask 输出，避免重复推理。

显示采用统一 0.3–6 m 色标，并标出区间外/无效预测；保存 RGB/raw/refined 三联图、差值图、原始/修复点云同视角对照。PLY 显示采样与数值评测分开，点数不是准确性证据。视频选固定片段、保留时间戳，同时包含普通和失败样例。伪彩色 PNG 不用作深度输入。

可额外导出“保留原始有效值、仅用模型填孔”的候选数组及 source mask，明确这是我们设计的融合候选，不冒称官方输出；本轮不因此自动进入 DN 训练。

## 7. Map 执行细则

### 7.1 三个独立序列

分别启动新的 stream：

1. `short_train_256`：固定 train 短片段。
2. `full_train_1478`：全部训练主帧按原始时间排序。用于可追溯的候选训练几何与诊断。
3. `full_all_1846_exploratory`：全部主帧，清楚标记包含 val/test 的完整采集探索。不能把其中 train 子集导出后冒充不含留出信息的训练先验。

每个输入 JSON 必须为按顺序的图片路径字符串数组；另存 frame ID、timestamp、split 与序列位置映射。使用明确输入清单，不按目录顺序猜时序。每个序列保持默认首 8 帧初始化。

官方调用形态如下，路径由执行者按实际位置生成：[O5–O7]

```bash
python download_model.py --source hf --output checkpoints/base-streaming
python infer.py \
  --checkpoint checkpoints/base-streaming/model_weights.pt \
  --config checkpoints/base-streaming/config.json \
  --image-list /absolute/path/to/ordered_frames.json \
  --output /absolute/path/to/new_prediction_directory
python demo.py --predictions /absolute/path/to/new_prediction_directory
```

以上在官方 Map 源码目录内运行，`/absolute/path/...` 是待替换说明，不是已有文件。每次显式传 `--config`；省略时加载器寻找 `<checkpoint>.json`，不保证寻找旁边的 `config.json`。输入列表相对路径按列表文件所在目录解析。

### 7.2 帧数、缓存与恢复

首轮保留官方 CLI 默认关键帧策略、512×288 处理画布、64 帧窗口、8 帧初始化，不加 `--keyframe-interval 1`。CLI 的 Uniform320 是约 320 帧预算，不是所有输入只预测 320 帧；记录实际 persistent IDs。默认最大持久帧容量为 1024。`demo.py` 的选关键帧实现与 CLI 不完全相同，所以统一先 `infer.py` 落盘、再 `demo.py --predictions` 查看，避免看图时又产生另一套重建。

前 7 次 API 返回 None，第 8 次返回 8 帧，之后通常返回单帧。使用 manifest 输出列表和 NPZ `raw_indices` 联合原始 frame ID 聚合；`frame_indices` 是可复用的提交时钟，不能作原始帧唯一键。不能用 NPZ 文件数量直接当预测帧数。

完成时核对每个输入恰有一份输出、无重复/无漏帧；失败条目明确留在统计里。保存原始 `poses_c2w`、`intrinsics`、`depth`、`depth_conf` 和输入映射。

Map 是有状态的流式模型。`progress.json` 和已保存 NPZ 不等于完整 KV/相机状态 checkpoint；官方无可用状态恢复时，首选用原始完整 image-list 重跑该序列，写新 attempt 目录。若适配器从头重放到中断位置，必须使用原始完整序列长度对应的已冻结 persistent raw IDs，不能对截短列表重新计算 Uniform320 而改变缓存历史。输入顺序、初始化前缀、关键帧选择和模型配置全部保持。不能从中间帧直接开始并宣称保持完整历史。切片独立运行会产生新坐标系/尺度，不能直接拼接成一条全序列轨迹。

### 7.3 几何、内参与尺度

保留未经对齐的原始预测。Map 输出为 OpenCV 风格 C2W（x 右、y 下、z 前），深度与平移为任意场景尺度。输出 K 对应自动覆盖缩放/中心裁剪后的画布。[O5]

与传感器比较前复用/复现官方 resize/crop 映射，保存尺寸、比例、裁剪与像素中心变换。深度/有效性参考使用正确投影或有遮挡处理的采样；不能用双线性跨孔洞或前后景边界混合出不存在的参考深度。已有 DN OpenGL C2W 与 OpenCV C2W 的轴变换应从既有实现核对，不能盲目取逆或反复翻轴。

在 P0 固定校准训练帧上，使用预定可靠、静态、非边界参考像素估计一个尺度：先求每帧 `sensor_Z / map_Z` 的中位数，再在有效校准帧间取中位数。**单一尺度的作用域是每个独立序列/attempt。**三个 stream 使用同一组冻结的校准 frame ID，但分别估计各自的尺度和刚体对齐；不能把 short 的数值倍率/坐标变换直接套到 full_train 或 full_all。记录具体支持数、排除理由、比值分布、尺度参数及不确定性线索。只用这些校准观测，不能在同一个序列内逐帧或逐段各乘新倍率美化结果。若支持不足或不一致，保留任意尺度结果并明确米制未验证。

为显示与旧轨迹叠加，可在同一校准帧上拟合一次旋转/平移；该序列全程使用同一变换。某个 run 缺少校准帧输出时，按该 run 报告支持不足，不临时替换帧以美化结果。相机中心近静止、近共线或过度集中造成对齐退化时，明确其不确定性；米制尺度可确定与轨迹刚体对齐可确定分别报告。可另报全 train 最优 Sim(3) 对齐差异作为附加诊断，但与固定校准结果分列，不用其掩盖沿程漂移。尺度同时作用于深度、点云和相机平移，不作用于旋转。

### 7.4 指标与可视化

必须报告：输出覆盖与异常、轨迹连续性和跳变位置、与旧轨迹的差异、固定重叠帧对的重投影/深度一致性、固定全局尺度下的分段残差/尺度比值、已知困难位置的三维对照。

已有 B2/B3 位姿并非独立 GT，不能把与其的差异叫真实 ATE/RPE。没有真实重访或独立依据时闭环指标标 not applicable，不能把终点拉向起点当效果改善。若使用原生深度、已有 TSDF 或网格作为参照，说明同源性和位姿依赖。

保留 raw confidence；默认查看阈值和全体有限点结果同时留证。`depth_conf` 是分数，不是概率；阈值改变时报告覆盖变化，不能只显示过滤后的漂亮点云。点云、相机轨迹、按时间增长的回放和至少一个失败/困难区域视角都要交付。没有失败的判断也要基于实际检查。

查看器在预测目录对应的原图仍可访问时运行。服务器上可用项目专属端口转发查看，不为展示关闭其他服务；视频/GLB/点云输出按清单保存。室内首轮不必启用额外天空分割下载。此版本没有 GCR，不实现一个假 GCR 阶段，也不把点云称为完成的 3DGS/可交互物理资产。

## 8. 数据使用规则与结论强度

### 8.1 固定划分

1478 train 用于本轮主体诊断、mask 设计、尺度校准和可用于未来训练的候选结果。val/test 不进入 `full_train_1478` 的前缀、记忆、尺度拟合或融合。只把全序列输出筛出 train 子集不能消除上下文信息。

完整 1846 另作探索重建，允许看全序列，但完整披露其用途。两种运行的上下文、输入数量和时间间隔不同，不宣称性能差异只由一个算法因素导致。

本轮不强行为 DN 生成全部 val/test 位姿。后续严格接入需要明确固定 train 地图下的定位方法；Map 的 `persistent=False` 本身不证明现成完成了无泄漏定位流程，不能用插值姿态充数。现有 test 已用于开发，报告继续叫固定开发 test。

### 8.2 采用建议

不临时发明一个百分数阈值来强行通过。分别给每条线写证据支持的结论：

- `candidate_supported`：在当前数据与明确参考下，有具体、可复查的收益，且局部损伤/覆盖损失已披露；可设计后续单因素 DN 实验。
- `local_use_only`：只在部分距离/场景/片段有帮助，限定适用范围。
- `insufficient_evidence`：能运行或补全，但没有足够几何证据。
- `not_recommended_current_setting`：出现有证据的恶化或与当前目标不符。
- `blocked`：实际执行受阻，明确未获得的证据与继续入口。

这些是技术判断，不是程序返回码。允许负结果，不要求一定超过旧模型。Depth 的候选应保留原始有效测量和模型来源 mask；Map 的候选要保留原始与对齐坐标、帧身份、内参、尺度证据。任何一项可用都不自动触发 DN 重训。

## 9. 待实现的最小脚本结构

以下为本轮 Codex 要实现的入口，**本计划提交时尚不存在，也未运行**。允许在职责不变的前提下简化模块，避免重复造框架：

| 建议路径 | 职责 |
| --- | --- |
| `scripts/lingbot_prepare.py` | 复用输入、身份清单、固定样例/mask、三个 Map 序列与画布映射 |
| `scripts/lingbot_depth_run.py` | 米制 NPY/内参适配、Depth 推理、逐帧恢复与状态 |
| `scripts/lingbot_map_run.py` | 调用固定官方 CLI、独立序列、原始索引聚合和恢复策略 |
| `scripts/lingbot_evaluate.py` | 三类 Depth 评测、Map 尺度/一致性、固定分母、逐帧汇总 |
| `scripts/lingbot_report.py` | 图表、点云/轨迹、视频与中文结论、产物清单 |
| `scripts/launch_lingbot_depth_map.sh` | 共用准备后启动两线 tmux；可分别 resume；汇总实际退出状态 |

依赖锁、必要适配补丁放版本控制目录；实际环境、上游工作副本、权重、原始预测和大产物放 `.local/`。使用既有可复用函数时核对是否包含全局 RUN 副作用，不能导入旧脚本就误写 B3。

## 10. 运行、日志与结果保存

建议 run ID：`lingbot-depth-map-20261128-v1`；完整数据位于 `.local/runs/<run-id>/`。公开小报告目录：`reports/lingbot-depth-map/20261128-v1/`，分支状态入口：`reports/lingbot-depth-map/latest.json`。本轮不覆盖历史根 `latest_status.json`。

状态分为 planned / running / completed / failed / blocked，按阶段和分支分别记录；质量判断单独存。记录时间（带时区）、命令、环境、上游与适配器 commit、权重、数据身份、输入/输出数量、退出码、错误、恢复入口。用户看到的计划状态不能被写成实测状态。

计时分模型加载、编译/预热、稳定 GPU 推理、数据拷贝、保存及端到端；GPU 使用同步或 CUDA event，报告 P50/P95、分辨率/batch/token、峰值显存/RAM。并行争用下的速度与单任务速度区分，不为了测峰值吞吐追加整套性能研究。

使用持久 tmux 或等效方式，无训练/推理完成 deadline、无 `timeout` 强杀、无自动关机。普通命令的网络故障超时与整轮实验期限不同。恢复先识别实际进程、输出和输入身份，避免重复协调；有阻塞时保留可继续工作。

## 11. 最低完整交付清单

- 中文 `README.md`：两项实验各回答了什么、哪些有帮助/无帮助、证据局限、后续建议。
- `sources.json`、`environment.json`、`input_manifest.json`、`selection.json`：实际来源、运行版本与输入身份。
- Depth：30 帧分组、两个固定 mask 清单、每帧 CSV/JSON、全 1478 train 输出清单、原始/修复/差异图、同视角点云、困难结构解释。
- Map：三个序列的输入/输出/持久帧计数、原始预测清单、尺度/对齐参数、轨迹图、可视化入口/视频、固定困难区域和未完成项。
- `metrics.json`：指标定义、单位、分母、覆盖、各分组/逐帧结果；缺测为 null 并附原因，不写虚构 0。
- `decision.json`：两条线分别判断，列证据和反例；是否值得后续 DN 单因素试验。
- `artifacts.json`：大产物实际路径、字节数、SHA-256、查看方式；所有大文件保留服务器。
- `execution.md`：实际执行、关键修复、失败、运行/恢复命令、各阶段完成状态。

普通 Git 提交计划、脚本、协议、必要补丁、小报告、精简指标与可读预览。模型权重、全量 NPY/NPZ、环境、完整原始图像、巨型点云和视频不进普通 Git；若已有授权的大文件发布方式可复用。**仅推用户仓库，不做额外 Windows/本机归档，不删除服务器数据。**

推送前核对 diff，保留并发工作；只提交本轮文件，不 force push、不重置用户改动。推送后读回分支/提交及入口链接。最终回复列实际完成数量、两个结论、视频/查看入口、失败与下一步，不能只说“已启动”。

## 12. 一手实现依据

- [O1] [Depth README / v0.5 / 官方入口](https://github.com/Robbyant/lingbot-depth/tree/f3a237e434ae987bc38281476d6cfb5df3e4d739)；[example.py](https://github.com/Robbyant/lingbot-depth/blob/f3a237e434ae987bc38281476d6cfb5df3e4d739/example.py)。
- [O2] [Depth 依赖声明](https://github.com/Robbyant/lingbot-depth/blob/f3a237e434ae987bc38281476d6cfb5df3e4d739/pyproject.toml)。
- [O3] [Depth 推理实现](https://github.com/Robbyant/lingbot-depth/blob/f3a237e434ae987bc38281476d6cfb5df3e4d739/mdm/model/v2.py)、[RGB-D 编码器](https://github.com/Robbyant/lingbot-depth/blob/f3a237e434ae987bc38281476d6cfb5df3e4d739/mdm/model/modules_rgbd_encoder.py)、[整数像素中心反投影](https://github.com/Robbyant/lingbot-depth/blob/f3a237e434ae987bc38281476d6cfb5df3e4d739/mdm/utils/geo.py)。
- [O4] [Depth nested-token 实现](https://github.com/Robbyant/lingbot-depth/blob/f3a237e434ae987bc38281476d6cfb5df3e4d739/mdm/model/dinov2_rgbd/layers/block.py)。
- [O5] [Map 2 README / 输入输出 / 基础版边界](https://github.com/Robbyant/lingbot-map-v2/tree/078b54485830c74ab1d3797667073e3e8353f3b9)。
- [O6] [Map 2 依赖](https://github.com/Robbyant/lingbot-map-v2/blob/078b54485830c74ab1d3797667073e3e8353f3b9/pyproject.toml)、[发布配置](https://github.com/Robbyant/lingbot-map-v2/blob/078b54485830c74ab1d3797667073e3e8353f3b9/configs/base-streaming.json)。
- [O7] [Map 官方下载器](https://github.com/Robbyant/lingbot-map-v2/blob/078b54485830c74ab1d3797667073e3e8353f3b9/download_model.py)、[CLI](https://github.com/Robbyant/lingbot-map-v2/blob/078b54485830c74ab1d3797667073e3e8353f3b9/lingbot_map2/utils/cli.py)、[输入和关键帧](https://github.com/Robbyant/lingbot-map-v2/blob/078b54485830c74ab1d3797667073e3e8353f3b9/lingbot_map2/utils/load_fn.py)。

本次提交只完成研究计划与启动说明。模型性能、兼容性和对自采数据的收益均以未来真实运行结果为准。
