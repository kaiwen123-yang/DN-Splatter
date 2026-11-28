# 每日研究与执行记录

本文件是 `kaiwen123-yang/DN-Splatter` 的长期研究日志入口。后续按 **Asia/Shanghai** 日期在同一个文件中追加条目；同一天补充新阶段，不另造多个互相矛盾的“最终日志”。长期目标是：从真实场景建立有来源、可接受动作、能随观测和交互校准、并能生成经验证训练数据的世界。

入口：[当前完整计划](docs/17_interactive_world_plan.md) · [完整 Codex 执行提示词](prompts/interactive_world_20261128.md) · [机器可读协议](configs/interactive_world_protocol.json) · [2026-11-28 现状复核](reports/interactive-world-review/2026-11-28/README.md) · [文献与借鉴范围](reports/interactive-world-review/2026-11-28/literature.md)。

## 记录规则

1. **区分计划、已启动、已完成、已验证。** 只有计划就记录计划；只有启动日志不能写完成；完成推理不等于质量通过；本日零训练、零仿真也如实填写。历史结果另标“历史回执复核”，不得冒充当日新结果。
2. **单一协调者写入。** 各 lane/worker 将事件、PID、指标和产物写到自己的 run/lane 日志或结构化 `events.jsonl`；协调者汇总后更新本文件并统一提交，避免多个 writer 同时覆盖 Markdown 或 push。跨日长任务按实际事件时间归档，保留原始时间戳及其时区。
3. **每个阶段都有身份与证据。** 记录源码 commit、run ID、输入/划分/模型/补丁身份、实际入口、累计样本或更新步数、状态、退出码、关键产物路径/字节数/SHA-256。未取得的值写“未取得”或 `null`，不填估计成功指标；日志不包含凭据、完整 SSH 参数或其他敏感信息。
4. **指标必须带分母和比较口径。** 明确单位、有效参考 mask、无效预测处理、逐帧/逐组/像素池化、拟合/校准/评价角色；记录负结果、被拒候选、未覆盖样本和未知区域。相同来源派生出的两个结果不能自动算独立证据。
5. **资源与偏差同样是结果。** 记录当前实际 quota/affinity、RSS、CUDA allocated/reserved、整卡占用、CPU 线程和 I/O，并注明哪个组合正在并行。实现或协议改变时保留原版本，说明原因及是否影响可比性。不要以“显存占满”代替有效实验吞吐。
6. **保留持续执行入口。** 每天结尾列仍在运行的任务、PID/锁、最近合法 checkpoint、剩余更新量和可恢复命令。尚未实现的命令只能标为待实现接口，不能伪装成已存在入口。某 lane 真实阻塞不应让其他独立 lane 一起停下。
7. **没有墙钟完成 deadline。** 不为训练、推理、网格或仿真设 timeout/watchdog 强杀，不自动关机；实验更新数、数据集长度和 episode 模拟时长是任务定义，不是墙钟 deadline。只读连接检查可有短连接超时。
8. **发布与保存边界。** 小代码、协议、脱敏报告和预览推本仓库；大模型、数据、完整视频和轨迹留项目服务器并登记清单。允许按哈希读取既有电脑/外置盘归档恢复原资产；不新建用户电脑归档，不删除服务器数据，不 force push。提交哈希以真实 Git 历史为准，不为日志自引用循环提交。

### 后续每日条目的固定内容

每一天至少填写以下各项，可按实际工作合并段落，但不能略去失败和未完成事项：

- 当日目标、授权变化和任务边界。
- 输入 commit、run、资产/模型 revision 与当前恢复状态。
- 实际执行的操作、入口、开始/结束与累计计数。
- 解决了什么，依据哪些代码、数值、日志、图片或视频。
- 未解决问题、负结果、拒绝原因、缺失证据和真实阻塞。
- 指标及完整分母、单位、划分、与基线的可比条件。
- 实际资源、并行组合、等待/OOM/编译/I/O 情况。
- 偏离原协议的内容、原因和修复验证范围。
- 产物路径、字节数、SHA-256、报告/视频/固定帧链接。
- GitHub 提交与远端核验结果；未推送就明确未确认。
- 仍在运行的任务及恢复入口、下一步和待新采集的数据。

---

## 2026-11-28｜建立新研究阶段的计划、证据复核与每日记录

**条目性质：已发布历史结果的复核、本地只读诊断、新执行计划的制定。此条目没有新增 GPU 模型推理、DN 训练、物理仿真或实机动作。** 下文 A 是已有结果，B 是本次实际复核，C 是尚待执行的新计划，三者不能混写。

### 1. 当日目标与范围

用户希望下一步不只展示静态重建和模型 demo，而要解释并实施有实质增量的算法实验，增加并行工作、日志与视频评估，最终走向可校准的交互世界。本次据此复核现状，形成新的完整计划和 Codex 提示词，并建立本文件。

研究主线仍为 LingBot-Depth 的有测量条件几何推断，以及 LingBot-Map 的几何上下文与记忆更新启示。新的候选质量校准、对象证据状态和物理桥梁属于本项目拟实施内容；不能写成官方已交付的功能。本阶段继续暂缓 VLA/VA 与 LingBot-World 大模型训练。

### 2. 输入身份与现有资源的证据边界

本次主要算法审阅固定在 [`ffc157fe30b01796f12e41361c6b2d3ace33521e`](https://github.com/kaiwen123-yang/DN-Splatter/commit/ffc157fe30b01796f12e41361c6b2d3ace33521e)。发布前另发现并读取同日并发执行器已推送的 [`94d17e968ec65854d616f2215bca5135b9d5d905`](https://github.com/kaiwen123-yang/DN-Splatter/commit/94d17e968ec65854d616f2215bca5135b9d5d905) 录屏回执；其新增录屏脚本/清单和 execution 补充保留，原 B2/B3、Depth、Map 指标没有改变。本次计划发布以该更新为基础，不覆盖并发成果。

主要历史 run 为 `lingbot-depth-map-20261128-v1`，其执行清单中的任务 commit 为 `aa979aea18d4e7ac492ef60b6c8449b4b5516f49`。大厅划分保持 **1,478 train / 184 val / 184 development-test**；该 test 已用于开发，不改称盲测。

固定官方源码与权重身份来自 [sources.json](reports/lingbot-depth-map/20261128-v1/sources.json)：

| 模型 | 已执行版本 | 固定官方源码 | 已执行权重 SHA-256 |
|---|---|---|---|
| LingBot-Depth | `lingbot-depth-pretrain-vitl-14-v0.5` | `f3a237e434ae987bc38281476d6cfb5df3e4d739` | `b60cf27ddbd0e51e9b59b03475c0d39d02d2e48ecf8dbb5866f04d46802b3c23` |
| LingBot-Map 2 | `lingbot-map-v2-base-streaming`，公开 base，GCR=false | `078b54485830c74ab1d3797667073e3e8353f3b9` | `d3ae62a09d3caaeb0947c4372f6fb23a4c0e85304867fa9702948ba030310e27` |

[上轮执行记录](reports/lingbot-depth-map/20261128-v1/execution.md)明确说明新实例数据盘原为空：从 11 月 25 日 H 盘既有归档恢复了 B2 相机、划分、标定、1,846 张 RGB 和 1,846 张全距离配准 Z，重建两个 LingBot 专属环境。按 B2 历史规则生成的 1,846 个区间深度文件与旧 SHA-256 全部一致。

本轮恢复白名单不含 native depth packet、B2/B3 checkpoint、DN 训练环境或 VR/Faro 数据。它们有些具备历史 receipt，但**不能据历史路径就宣称现在都在服务器上**。新的执行入口必须先做实际库存与哈希核验；本条目不代替该远程库存检查。需要时读取已有归档恢复，不创建新的电脑归档。

### 3. A｜已有回执支持的历史结果，今日没有重跑

#### A1. B2/B3 已完整完成，不能退回“只有早期 30k”的叙述

来源：[B3 总报告](reports/capture-full-quality-v3/20261122-v1/overview.md)、[完整视频与指标对比](reports/capture-full-quality-v3/20261122-v1/B3_完整视频与指标对比.md)。两组已训练至 240k，验证选择为 210k；冻结推荐为 B2 210k 渲染、B3 210k supported Poisson 网格。

| 历史 210k 节点，development-test | B2 | B3 |
|---|---:|---:|
| PSNR | 24.6807 dB | 24.8184 dB |
| 模型深度 MAE | 12.36 cm | 13.06 cm |
| 模型深度 P95 | 44.13 cm | 41.61 cm |
| supported Poisson 的 10 cm 正确覆盖 | 61.95% | 62.84% |
| supported Poisson 的前方冲突 | 22.26% | 20.73% |

正确覆盖以全部 **29,100,252 个有效参考像素**为分母，未命中同样进入分母。参考来自同次采集的 sensor Z，不是大厅独立 GT。B3 已有独立观测组支持/冲突权重 `(1+S)/(1+S+C)`；已接受位姿改动为 0。新计划不能把重复实现这条权重公式当成创新。

#### A2. 已执行 LingBot-Depth 的全量计数与负结果

已有 **1,478 张无人工遮挡推理 + 60 次固定人工遮挡推理，共 1,538 次正式调用**。来源：[LingBot 总报告](reports/lingbot-depth-map/20261128-v1/README.md)、[metrics.json](reports/lingbot-depth-map/20261128-v1/metrics.json)。

| 固定隐藏任务 | 全部计分像素 | LingBot 10 cm 正确覆盖 | 同输入 NN | LingBot / NN 条件 MAE |
|---|---:|---:|---:|---:|
| 10% 散点 | 488,533 | 76.03% | 99.34% | 25.46 / 1.02 cm |
| 64×64 连续块 | 991,427 | 74.92% | 78.77% | 30.15 / 17.44 cm |

覆盖率分母固定，无效预测是 miss；MAE 则在正有限预测上计算。散点逐帧为 NN 30/30 胜出；块遮挡 LingBot 的 10 cm 正确覆盖在 8/30 帧较高。因此目前不支持整图替换，局部选择有待后续对照验证。

全训练原孔洞 **901,836,249**；其中模型产生正有限值 **896,873,869**，即 **99.4497%**。这些已填值里 **462,513,769** 落在 0.3–6 m 外，占已填值 **51.5695%**；在主评测区间内的正预测占原孔洞 **48.1640%**。区间外不一定错，但不能把正数覆盖称为准确几何覆盖。

64 个诊断目标帧共有 **39,277,863** 个孔洞，其中 **737,442** 个有两个其他视角的传感器共同支持，占 **1.8775%**；其中 **737,428** 个有有效预测，**627,139** 个与支持表面相符，即 **85.0441%**。其余缺少两视角证据的位置不是已判错；64 帧里也只有 53 帧取得两个选定邻居。这些证据仍依赖 B2 位姿和当前投影方式。

#### A3. 已执行 Map，当前只把它当独立候选

已完成 256、1,478、1,846 帧三个独立序列；完整训练序列保留 302 个持久关键帧。1,846 帧序列含 val/test RGB，只作探索，不得回流为严格 train-only 先验。

在原 32 帧定尺度/刚体对齐协议下，1,478 帧训练序列相对 B2 的相机中心差异中位/P95 为 **0.82654 / 1.62161 m**，旋转差中位/P95 为 **0.56153 / 1.47527°**；固定尺度深度与 sensor Z 的像素池化 MAE 为 **0.47332 m**。这些是与估计参考的差异，不能称为独立 GT 的轨迹误差。

#### A4. 同日并发任务新增完整录屏回执，本次只读未完整观看

北京时间 16:34 发布的 `94d17e9…` 新增 [hall_recording.json](reports/lingbot-depth-map/20261128-v1/hall_recording.json) 与 [录屏脚本](scripts/lingbot_record_hall.py)，并追加执行记录。其回执记录 **1,846 个探索序列时间位置 + 250 个 orbit 帧 = 2,096 个解码帧**，逐帧 PTS 核对 2,096 项；分辨率 1920×1080，时长 **379.074973 秒**。沿用 30 个既有遮挡位置，因此 `source_frames_displayed=1816`；显示点数 **4,266,403**，置信阈值 1.5，采样 stride 为 6。

此完整视频使用 `full_all_1846_exploratory`，含 train/val/development-test RGB，不能倒流成严格训练先验。PyAV 15.1.0 已在该录屏任务成功使用 H.264/NVENC；这是新增录屏链的结果，不能被旧 FFmpeg 编码限制掩盖。按用户要求，16:33 已关闭当时的 viewer/replay 服务和端口转发，不能继续把旧服务写成当前运行中。

**本次仅阅读该回执，没有下载解码或完整观看这个 379 秒视频；实际画面复核仍仅限后述原 12/16 秒预览及静态图。** 录屏与任何既有用户电脑副本属于另一个并发执行任务，本次计划/审阅没有创建或验证 Windows 副本，也没有开展新的电脑归档。

### 4. B｜本次实际完成的复核与本地分析

#### B1. 仓库、实现与数据恢复审计

本次阅读了已发布 README、execution、metrics、timing、sources、decision、artifact 清单，以及 prepare、Depth/Map 推理与评价、媒体生成和 B3 深度权重/局部位姿实现。重点确认了以下内容：

- 人工隐藏在模型推理前置零，NN 使用同一隐藏输入，主计分 mask 固定；没有发现把隐藏答案回灌模型的明显实现问题。现有米制 NPY、RGB 通道和 K_cv 约定与协议吻合。
- 配准采用 native 点到 RGB 单像素前投影、最近 Z 的 z-buffer。848×480 到 1280×720 即使所有源点有效且无碰撞，单点占用上限也仅 **44.1667%**；实际全距离空值为 **66.2080%**。规则黑孔有明确采样链来源，但不能据此断言全部都是栅格洞或这是模型长尾的唯一成因。
- 原生无效射线没有深度，无法仅用标定唯一归因到一个 RGB 空洞；新的来源记录必须允许 unknown。需要 native packet 的工作与仅依赖现有 RGB-Z 的工作分开恢复。
- B3 已经实施独立组、源组排除、支持/冲突权重和留组核验，新算法应改变候选、学习目标和缺失位置监督，而非重复命名已有可靠性权重。
- VR Room/iPhone/Faro 有真实历史数据哈希、划分与执行 receipt；可以建立第二场景验证，但当前恢复状态还需核查。旧 Faro 指标中 MSE 与 RMSE 的有效区域、旧 log 指标定义等需版本化说明，不覆盖旧结果。

#### B2. 图片与已发布短视频复核

实际下载并解码 [Depth 预览](reports/lingbot-depth-map/20261128-v1/previews/depth_preview.mp4)和 [Map 预览](reports/lingbot-depth-map/20261128-v1/previews/map_preview.mp4)，按约两秒间隔检查画面；前者为 **12 秒、60 帧、5 Hz**，后者为 **16 秒、80 帧、5 Hz**。并查看 depth_structures、map_difficult、全训练轨迹，以及 B3 桌沿、椅腿、泡沫齿边、柱面的同相机对照。

Depth 节选约覆盖采集 20–32 秒，Map 节选约为 252–268 秒，包含 frame_01304 附近。观察支持“主体布局/部分结构可见、薄结构与接触面仍有明显不确定性”，不支持物体直径、闭合碰撞体或物理材质已经正确。

没有访问用户电脑的 `127.0.0.1` 查看器，也没有将完整视频清单当作已经看完六分钟全程视频。后续执行者仍需生成并审阅完整时间段异常索引。图像着色、点云抽样和不同色标会影响观感，数值与来源证据必须同时保留。

#### B3. 本地 Map Sim(3) 诊断：只有参考拟合，没有新模型推理

实际对已提交的 `full_train_1478/trajectory.json` 与 `alignment.json` 做 NumPy/SVD 相机中心对齐复算，记录在 [map_alignment_diagnostic.json](reports/interactive-world-review/2026-11-28/map_alignment_diagnostic.json)。输入共 1,478 个相机中心，前段使用原冻结的 32 个 fit IDs。下表的米单位属于 B2 参考坐标。

| 本地只读诊断 | 尺度 | 全 1,478 中心差异中位 | P95 | 解释 |
|---|---:|---:|---:|---|
| 复算原固定尺度 + 前段刚体对齐 | 8.0477147 | 0.826540 m | 1.621608 m | 与原发布结果吻合 |
| 仅用原 32 个中心拟合 Sim(3) | 8.3201199 | 0.586327 m | 1.249863 m | 仍以 B2 作为估计参考 |
| 用全部 1,478 中心拟合，再在同样中心计分 | 9.1032523 | 0.114385 m | 0.397663 m | 乐观的同参考拟合诊断，不是精度提升 |

前段 Sim(3) 在 32 个 fit IDs 之外的中位/P95 为 0.605941/1.251034 m；虽然没有在这些中心上直接拟合，它们仍来自同一相关训练序列和相同 B2 参考，不是独立 benchmark。这个诊断说明需拆开尺度、姿态、局部几何问题，**没有生成一条更准确的新轨迹，没有修复 B2，也没有运行 Map 新实验**。

#### B4. 执行前发现的具体风险，尚不是运行时修复成果

本次交叉审查提出并纳入新计划修订的约束包括：outer fold 内另分 fit/calibration；新 mask 的预测与 MDM 特征不得复用未遮挡缓存；边界任务不能继续与旧 `primary & ~band` 参考相交而得到空分母；hole mask 必须从全距离 Z 定义；`inf` 候选需先过滤再计算损失，不能用 `0×inf` 掩掉；R1–R5 的共同预算及历史 R0 可比性必须明确；VR 不能沿用大厅固定分辨率/尺度接口。

上述是计划与实现要求，不是“已经修复并跑通”的声明。真正执行后须在后续条目补实际代码、输入、结果与失败记录。

### 5. C｜下一阶段已明确的工作，当前尚未执行

六条 lane 形成流水线，而非同时启动六个大训练进程：A 为 Depth/来源/质量学习，B 为 Map/独立尺度与位姿候选，C 为 DN/薄结构，D 为对象与碰撞世界接口，E 为真实交互与校准/数据有效性，F 为独立几何评价、视频、资源调度和发布。

A 线拟真正训练质量头：**Q0 不使用 MDM 特征，Q1 使用冻结 MDM 特征**，保持可审计的共同接口、训练预算和对照容量；结合结构化隐藏与 out-of-fold 预测，比较固定规则和经验风险。另做主距离外置零与 footprint 输入表示消融。模型输出的 validity mask 不叫“正确概率”；训练过质量头也不等于微调了整个 LingBot。

C 线拟完成下列五组必做 240k，并根据真实改动执行两个条件组。这是**下一阶段预算**，不计入当日已完成步数：

| 组 | 拟比较的变化 | 当前状态 |
|---|---|---|
| R1 | B3 共同基线 + sensor-only 保守缺失监督 | 未启动，目标 240k |
| R2 | 同候选体系增加冻结 LingBot，以独立证据硬规则选择 | 未启动，目标 240k |
| R3 | 与 R2 相同候选，使用 out-of-fold 质量头及拒绝规则 | 未启动，目标 240k |
| R4 | R3 + 同预算边界容量分配 | 未启动，目标 240k |
| R5 | R3 仅关闭 normal TV | 未启动，目标 240k |
| R6 | R3 + 独立核验通过、确实变化的局部位姿 | 条件未判定；触发后 240k |
| R7 | 已分别验证的边界与位姿模块组合 | 条件未判定；触发后 240k |

原 sensor 数值/B3 权重只读；缺失监督单独归一化，不能让约 9 亿缺失位置的伪标签数量压倒约 2.35 亿主距离内测量。零候选只有在文件和损失/梯度等价证据齐全时才能作数值等价注释，五个必做组仍保留完整预算；未实施、失败或缺资产不等于等价。历史 R0 不符合新共同协议时另加 R0C。

D/E 线拟实现局部对象、碰撞代理、reset/step/observe、动作与观测时间对齐、可见性与对象版本更新。**PhysTwin 的公开真实 RGB-D 交互数据支线将开展来源/标定/动作接口审计及后续实验，当前未执行该数据恢复、训练或物理校准。** 公开真实动作与受控合成数据的角色必须分开；大厅静态采集本身不能辨识质量、摩擦或接触响应。模型生成画面不自动等于可用真实训练标签。

F 线拟恢复 VR/Faro 做第二场景验证，Faro 仅供评价；同时交付固定视角、全时间段、失败片段与数据有效性评估。新协议不以表现差为由偷偷缩短完整训练，不设墙钟 deadline。各 lane 缺失资源时分别标真实阻塞，并继续其他可执行工作。

### 6. 当日实际完成计数与资源

| 项目 | 本次复核/计划阶段新增量 |
|---|---:|
| GPU Depth 推理 | 0 |
| GPU Map 推理 | 0 |
| DN 训练更新 | 0 |
| 质量头训练 epoch | 0 |
| 物理 rollout / 实机动作 | 0 / 0 |
| 本地已发布轨迹对齐诊断 | 1 组，含 3 种对齐口径、1,478 个中心 |
| 解码并分时段检查的已发布预览视频 | 2 段，共 28 秒；不是新生成的模型视频 |

本次由多个审阅分支并行完成 Depth/数据、Map/轨迹、交互/文献工作，协调者汇总计划与报告；没有在 AutoDL 上并行启动这些新实验。当前记录不提供虚构 GPU 利用率、训练 PID 或远程任务完成时间。

历史 LingBot 运行使用 RTX PRO 6000 Blackwell Server 96 GB；Depth 稳定 GPU 段中位约 55.1 ms，端到端约 96.2 ms。历史 DN 峰 RSS 约 79 GiB。这些是后续调度的参考值，**不是今日资源采样**。下轮需依据实际 25 CPU/120 GB RAM/96 GB VRAM 的可用配额、RSS 与 reserved 实测安排重任务；共享缓存、有限预取与 CPU 评价可并行，不能默认六个 79 GiB 训练进程同时运行。

### 7. 未解决问题、负结果与本阶段偏差

- LingBot 在当前固定人工隐藏任务整体落后 NN，现阶段未证明自然孔洞的净新增几何价值；候选质量头、输入表示和独立几何评价均待执行。
- 原孔洞的完整来源比例还没有测量；native packet 未获当前恢复 receipt，不能将未知 RGB 空洞强行标成传感器失效。
- B2 仍不是 GT，可疑局部姿态尚未由本次诊断修复；全轨迹 Sim(3) 的较低差异只是参考拟合结果。
- 当前外观改善不能认证薄杆直径、接触表面或碰撞闭合性；物理对象资产、质量/摩擦识别、动作响应与合成数据增益尚无新完成结果。
- VR/Faro、B2/B3 checkpoint 和 DN 环境的当前服务器可用性需先核实；PhysTwin 实际数据条件与可用动作/观测证据也需核查。
- 本次新增范围是更完整的算法与交互执行计划；没有改变已发布 v1 的输出、阈值或分母来美化旧结果。新风险判断和输入消融应产生新版本，不能覆盖原失败证据。

### 8. 产物与身份清单

本次新增或更新的发布内容包括 [现状复核](reports/interactive-world-review/2026-11-28/README.md)、[完整计划](docs/17_interactive_world_plan.md)、[执行提示词](prompts/interactive_world_20261128.md)、[协议](configs/interactive_world_protocol.json)、[文献说明](reports/interactive-world-review/2026-11-28/literature.md)、[本地对齐诊断](reports/interactive-world-review/2026-11-28/map_alignment_diagnostic.json)和本日志。计划/提示词/协议的最终文件身份随所在 Git 提交固定，不在写作中预填可能失效的哈希，也不计算日志自身的递归哈希。

本次实际读取或生成并核对过的稳定证据身份如下：

| 仓库相对路径 | 字节数 | SHA-256 |
|---|---:|---|
| `reports/lingbot-depth-map/20261128-v1/previews/depth_preview.mp4` | 638426 | `d911ecc1290af0fdb629800bb8334085b5296443ef53df3d8baca28304ff213c` |
| `reports/lingbot-depth-map/20261128-v1/previews/map_preview.mp4` | 869517 | `5123ab92d26d5c370cfe9831f98db264b8d0cfd92c7bffe966dcd30dab4bc126` |
| `reports/interactive-world-review/2026-11-28/map_alignment_diagnostic.json` | 5503 | `b624461c4c9f100c643533d2c0e7cc505819078bd252b7134dd4ac0b34e0ee7f` |

诊断输入哈希已记录在 JSON 中：`trajectory.json` 为 `961ae9c7008e5c99f9826b411205b34216e085c3f37a2176ff7f4e80c8d97103`；`alignment.json` 为 `eecb6524d019bfb2ba6583dc4c50fb28cc4586404012586488c552d9694b3821`。二者均来自审阅的 `ffc157fe…`，不是新采集数据。

同日并发录屏另有已核对的 receipt：`/root/autodl-tmp/DN-Splatter/.local/runs/lingbot-depth-map-20261128-v1/media/hall-recording/hall_rgb_sensor_map_cloud_1080p.mp4`，**169,046,476 bytes**，SHA-256 `2aa994e60386b39ab4501bb7b34783dafb6be898ed38489809c163366a88b864`。这里登记的是 [hall_recording.json](reports/lingbot-depth-map/20261128-v1/hall_recording.json) 的服务器产物身份，不表示本次已经取得并观看该文件。

其他历史大产物仍以 [LingBot artifacts.json](reports/lingbot-depth-map/20261128-v1/artifacts.json)、[B3 video.json](reports/capture-full-quality-v3/20261122-v1/video.json)及相应训练清单为准。本次计划/审阅自身没有新增服务器 checkpoint、完整模型视频或物理轨迹，不能将并发录屏成果计作本次模型实验。

### 9. GitHub 发布、恢复入口与下一步

**GitHub 提交：本条目随所在提交发布，以 Git 历史为准。** 历史算法结果审阅基线为 `ffc157fe30b01796f12e41361c6b2d3ace33521e`；同日录屏回执追加于 `94d17e968ec65854d616f2215bca5135b9d5d905`。新的计划提交应保留并基于该更新；本条目不预编一个未来推送 SHA。协调者推送后读取远端核验真实提交和文件，后续日条目可引用该实际 commit。

新阶段恢复/启动的规范入口是 [完整 Codex 提示词](prompts/interactive_world_20261128.md)与 [机器协议](configs/interactive_world_protocol.json)。当前尚未启动新阶段，不存在可填的运行 PID、已完成更新或最新新 checkpoint。执行者先检查现有进程/锁和资产，实施所需脚本，再以真实入口记录启动；不能仅因为计划列出脚本名称就声称命令已经可运行。

下一步先完成新阶段实际 inventory、恢复所需旧资产并冻结输入/划分；A/B/F 的独立准备并行，C 在各对照输入就绪后进入完整训练，D/E 按各自数据与接口进度推进。Nerfstudio 恢复应读取本 run 的实际 `saved_step`，大厅剩余量为 `240000-(saved_step+1)`，保留调度、优化器、采样/RNG 与初始化身份；不加载历史 B2 checkpoint 后宣称从头公平对照。

真正需要新的实机数据时，另列所需对象、动作、观测、标定和授权，不能用大厅静态序列假装已取得交互真值。后续每天继续在本文件记录实际完成、负结果、资源和可恢复状态。

## 2026-11-28｜新服务器正式执行与 tmux 交接

用户指定本项目专用实例，授权在 tmux 稳定、任务可以脱离聊天后交接。运行 ID 为 `interactive-world-20261128-v1`。核对工作区、origin、旧任务锁后，本地 main 确认包含 `f10e91805877b197d49107347844b675df856d3f`，服务器 main 从 `94d17e9` 快进至该版本。初次盘点为 25 核 cgroup 配额、208 个可见逻辑 CPU、120 GiB 主存上限、97,887 MiB GPU 显存和约 507 GiB 可用项目盘。历史五个 LingBot 锁文件属于旧 run；本轮用不同目录和内核锁。

**恢复。** 读取既有 H 盘归档，封存输入索引 SHA-256 为 `a33d58c5e9dfd725cf2536bbd0e95a33c6268a01c39086f7581af5239bc6de97`。服务器逐文件校验了环境与固定源码 94,445 个文件、1,399 个链接，大厅输入和权重 7,026 个文件，VR/Faro 2,437 个文件，以及 B2/B3 两个 210k checkpoint 及配置共 4 个文件。原生 packet 校验 3,696 个成员、714,380,816 B。恢复器首轮将空文件误作有内容的分块，改在目标创建并核验空文件。R0C 首次准备时 LPIPS 重复下载，在零更新时保留下载部分，接入从归档恢复的 244,408,911 B 正确权重。两条 gsplat CUDA 后端均在真实 VR 帧上通过梯度检查。[恢复收据与哈希](reports/interactive-world/20261128-v1/inventory.json) · [Blackwell 核验](reports/interactive-world/20261128-v1/dn_blackwell.json)

**A、B。** 全 1,478 帧完成 4,434 个遮挡组合、13,302 个条件输入。原生足迹、连续块、几何边界单种输入表示的计分分母为 5,959,722、4,220,696、23,316,483。首次多进程继承 OpenCV 线程锁，改用 spawn 后，3 个 CPU worker 用 428.08 s 完成 R1 监督。R1 在 901,836,249 个自然孔洞中接受 509,421 个候选，R2 接受 761,830 个，其中 379,369 个选择 MDM。五折收益以外折评分为准。B 线 840 帧拟合和 558 帧审计得到全局尺度 9.23860；在 12,333,809 个有效审计像素上，MAE 从历史前缀尺度的 0.4880 m 降至 0.4010 m，跨视角条件 MAE 升至 1.1022 m。有序预取流实际处理 1,478 帧，流式 24.59 帧/s，含冷启动 5.75 帧/s。全部 653,819,904 个 CPU 输入标量逐位相等，输出从首个八帧块已有差异，具体 GPU 环节待定位。三个预注册疑难帧的候选均未通过单帧验收，其他时间段继续计算。[Map 计时](reports/interactive-world/20261128-v1/map_ordered_stream.json) · [输入核验](reports/interactive-world/20261128-v1/map_cpu_equivalence.json)

**C、F。** B3 真实曲线给出统一高斯上限 2,071,347，触发 R0C。R0C 首个正式 batch 与历史 B3 的帧 01584、loss 0.3945617676、122,209,473 个梯度一致。R0C、R1 首次实际改变容量均在 step 5600。北京时间 19:14:31，R0C 至少 43,000 步，R1 至少 37,800 步，VR0C 完成 30,000 步，三个已校验恢复点分别为 40k、30k、30k，均含 7 个优化器、调度器、采样队列及 RNG。大厅 30k 全 184 张验证图、29,222,080 个参考像素中，R0C/R1 MAE 为 0.14210/0.14429 m，10 cm 正确覆盖为 71.7965%/71.8428%。VR 实际宽 738、高 994，原计划文字对调 H/W。全 376 训练帧无自然缺测，VR1/VR2 两组预算均保留。VR0C 的 42 帧同采集、186 帧异采集 Faro MAE 为 3.420、4.319 cm，分母 26,525,046、114,993,165；已完成 418 视角裁剪，严格 376 视角另记。[首个验证节点、Faro 与条件位姿](reports/interactive-world/20261128-v1/first_validation_summary.json)

**D、E。** 大厅首次单中心选面失败，固定桌面 ROI 重建后，0.247 × 0.1365 m 支撑区内部 75/75 个原始点位于 35 mm 带内，另有 9 个训练时组支持。插入合成刚体的 90 步回放重复一致，51 步出现接触，最大穿透 0.767 mm。PushT 真实 60 条按组冻结 36/12/12，测试集 10,082 帧中有 9,778 个有效对象标签。C0/C1/C2/C3 开环位置误差分别为 36.19、38.07、35.50、35.50 mm。512 组相同初态与动作生成 1,536 条、230,400 步，每个版本均有 148/512 条发生 >5 mm 穿透，均已保留。15 个小预测器各完成 20k；C0 增强在三个种子的真实 2 秒任务上都优于 RealOnly，C2、C3 的 512 条对象轨迹完全相同。PhysTwin 两例在 0.5 s 末帧的 G-NN→G-LingBot 跟踪误差为 17.31→21.47 cm（8 点）、21.43→15.02 cm（9 点），G-Q1 依赖折 0 校准。真实机器人策略成功率未评测。[完整数据和探针结果](reports/interactive-world/20261128-v1/pusht_results.json) · [推绳](reports/interactive-world/20261128-v1/phystwin_results.json) · [留出视频](reports/interactive-world/20261128-v1/previews/pusht_heldout_60s.mp4)

**资源与发布。** DN 共享只读 mmap 并预取下一帧，调度依据实际 cgroup 可回收文件页、增量工作集、PSS/USS、CUDA reserved。18:14:12–19:14:11 的 180 次采样平均 GPU 利用率为 99.36%，R0C、R1、VR0C 分别推进 21,700、19,600、5,500 步，cgroup OOM 与 OOM kill 为零。19:10 起另用 4 个 CPU worker 拟合已落盘的匹配。项目根目录执行 `bash scripts/launch_interactive_world.sh status`，总调度在 `dn-interactive` socket 的 `interactive-world` 会话，A 线使用 `iw-a` 的 `iw-a-coordinator`。PushT 窗口已以退出码 0 收尾。服务器自身的 GitHub 写入探测未获确认，发布使用本机现有 GitHub 身份；训练、评价、每日日志在服务器独立执行。首次执行提交哈希以本条所在 Git 历史及远端回读为准。[试验总表](reports/interactive-world/20261128-v1/README.md) · [实测状态、产物路径与哈希](reports/interactive-world/latest.json)

### 19:21｜VR0C 与两例真实 RGB-D 完成评价

首次执行提交 `7872b084574844c7c4802324236694afc002768d` 已分别从 GitHub、本机和指定服务器读回。该提交含完整代码、分支回执、留出视频和复现入口。此后 VR0C 的严格 376 视角裁剪完成，F-score 为 0.868006，accuracy 0.028940 m，completeness 0.022032 m。调度器已释放其评价槽并启动 VR1。大厅第 0 折在 39,950,843 个校准候选上选择 Q0 的第 20 个、Q1 的第 22 个 epoch。两例 PhysTwin 均完成 G-Q1，0.5 秒末帧跟踪误差为粗绳 17.46 cm（8 点）、细绳 17.59 cm（9 点）。G-Q1 在粗绳减轻直接补全的误差，在细绳介于 NN 和直接补全之间。[质量头选点](reports/interactive-world/20261128-v1/q_fold0_epoch_selection.json) · [Faro 完成清单](reports/interactive-world/20261128-v1/vr0c_complete.json) · [推绳三路结果](reports/interactive-world/20261128-v1/phystwin_results.json)

19:24 补充。全 1,846 个大厅时刻已作为独立静态诊断运行，沿用 30 个遮挡位置、3 个困难位姿，37 组共 472 个背景锚点。实测 259 帧支持充分、1,553 帧证据弱。可更新的 11,084 个跨组比较含 9,017 次前方遮挡，未因此撤回旧表面；1,324 次可靠看穿中，M1/M2 各新撤回 233 次。静止假设下记疑似误更新，未获得独立运动标签。结果 SHA-256 为 `844befa4593b3f3f078015014553a3ed7cb0ba1d21fef36623dd4485fb409966`。[静态全厅审计](reports/interactive-world/20261128-v1/hall_static_memory.json)


### 2026-11-28T19:24:35+08:00｜交互世界服务器实测

运行 `interactive-world-20261128-v1` 使用逐文件哈希核验的大厅、VR、DN 环境和原生深度。采样时仓库提交为 `4fce6619f49b1662700508b0bdf80ac859cd8200`。当前正式训练累计下界 121,800 步，其中可从已落盘 checkpoint 恢复 110,000 步。

| 训练组 | 阶段 | 已执行步数下界 | 可恢复步数 | PID | 调度实态 |
|---|---|---:|---:|---:|---|
| R0C | train | 47,900 | 40,000 | 15993 | running |
| R1 | train | 42,300 | 40,000 | 16581 | running |
| R2 | train | 0 | 0 | — | three heavy workers already active |
| R3 | train | 0 | 0 | — | waiting for a_depth/supervision/R3/manifest.json |
| R4 | train | 0 | 0 | — | waiting for a_depth/supervision/R3/manifest.json |
| R5 | train | 0 | 0 | — | waiting for a_depth/supervision/R3/manifest.json |
| R6 | train | 0 | 0 | — | awaiting independent pose audit |
| R7 | train | 0 | 0 | — | awaiting independent pose audit |
| VR0C | evaluate | 30,000 | 30,000 | — | evaluation receipt present |
| VR1 | train | 1,600 | 0 | 33189 | running |
| VR2 | train | 0 | 0 | — | three heavy workers already active |

A 线 sensor-only 在 1,478 个训练帧的 901,836,249 个自然孔洞中接受 509421 个。B 线有序流已记录 1478 帧，独立训练时组的尺度审计使用 558 帧。固定大厅相机与传感器读数都来自同一采集。

D 线实测支撑面的代理物体接触为 51/90 控制步。E 线 RGB 跟踪的共同有效标签为 46558/47582 帧。生成轨迹数为 1536，对应 230400 个控制时刻。小预测器的 15 项条件与种子组合累计训练 300000 步。

资源快照含 25 核 CPU 配额，GPU 利用率 100%，整卡已用显存 16.72 GiB，cgroup 总用量 99.25 GiB，清洁非活跃文件页 24.65 GiB。逐阶段错误与重试见 run 下 `state/coordinator-events.jsonl` 和各 worker 日志。

本次产物身份、字节数、条件分支和运行 PID 见 [实测状态](reports/interactive-world/latest.json)。在项目服务器根目录运行 `bash scripts/launch_interactive_world.sh status` 可读取最新实态。

<!-- interactive-world-evidence 6c2c480859bcd096 -->


### 2026-11-28T19:27:35+08:00｜交互世界服务器实测

运行 `interactive-world-20261128-v1` 使用逐文件哈希核验的大厅、VR、DN 环境和原生深度。采样时仓库提交为 `4fce6619f49b1662700508b0bdf80ac859cd8200`。当前正式训练累计下界 127,300 步，其中可从已落盘 checkpoint 恢复 110,000 步。

| 训练组 | 阶段 | 已执行步数下界 | 可恢复步数 | PID | 调度实态 |
|---|---|---:|---:|---:|---|
| R0C | train | 49,700 | 40,000 | 15993 | running |
| R1 | train | 44,000 | 40,000 | 16581 | running |
| R2 | train | 0 | 0 | — | three heavy workers already active |
| R3 | train | 0 | 0 | — | waiting for a_depth/supervision/R3/manifest.json |
| R4 | train | 0 | 0 | — | waiting for a_depth/supervision/R3/manifest.json |
| R5 | train | 0 | 0 | — | waiting for a_depth/supervision/R3/manifest.json |
| R6 | train | 0 | 0 | — | awaiting independent pose audit |
| R7 | train | 0 | 0 | — | awaiting independent pose audit |
| VR0C | evaluate | 30,000 | 30,000 | — | evaluation receipt present |
| VR1 | train | 3,600 | 0 | 33189 | running |
| VR2 | train | 0 | 0 | — | three heavy workers already active |

A 线 sensor-only 在 1,478 个训练帧的 901,836,249 个自然孔洞中接受 509421 个。B 线有序流已记录 1478 帧，独立训练时组的尺度审计使用 558 帧。固定大厅相机与传感器读数都来自同一采集。

D 线实测支撑面的代理物体接触为 51/90 控制步。E 线 RGB 跟踪的共同有效标签为 46558/47582 帧。生成轨迹数为 1536，对应 230400 个控制时刻。小预测器的 15 项条件与种子组合累计训练 300000 步。

资源快照含 25 核 CPU 配额，GPU 利用率 99%，整卡已用显存 17.36 GiB，cgroup 总用量 98.43 GiB，清洁非活跃文件页 24.85 GiB。逐阶段错误与重试见 run 下 `state/coordinator-events.jsonl` 和各 worker 日志。

本次产物身份、字节数、条件分支和运行 PID 见 [实测状态](reports/interactive-world/latest.json)。在项目服务器根目录运行 `bash scripts/launch_interactive_world.sh status` 可读取最新实态。

<!-- interactive-world-evidence 5efbd8169ede1248 -->


### 2026-11-28T19:28:36+08:00｜交互世界服务器实测

运行 `interactive-world-20261128-v1` 使用逐文件哈希核验的大厅、VR、DN 环境和原生深度。采样时仓库提交为 `4fce6619f49b1662700508b0bdf80ac859cd8200`。当前正式训练累计下界 129,200 步，其中可从已落盘 checkpoint 恢复 120,000 步。

| 训练组 | 阶段 | 已执行步数下界 | 可恢复步数 | PID | 调度实态 |
|---|---|---:|---:|---:|---|
| R0C | train | 50,300 | 50,000 | 15993 | running |
| R1 | train | 44,600 | 40,000 | 16581 | running |
| R2 | train | 0 | 0 | — | three heavy workers already active |
| R3 | train | 0 | 0 | — | waiting for a_depth/supervision/R3/manifest.json |
| R4 | train | 0 | 0 | — | waiting for a_depth/supervision/R3/manifest.json |
| R5 | train | 0 | 0 | — | waiting for a_depth/supervision/R3/manifest.json |
| R6 | train | 0 | 0 | — | awaiting independent pose audit |
| R7 | train | 0 | 0 | — | awaiting independent pose audit |
| VR0C | evaluate | 30,000 | 30,000 | — | evaluation receipt present |
| VR1 | train | 4,300 | 0 | 33189 | running |
| VR2 | train | 0 | 0 | — | three heavy workers already active |

A 线 sensor-only 在 1,478 个训练帧的 901,836,249 个自然孔洞中接受 509421 个。B 线有序流已记录 1478 帧，独立训练时组的尺度审计使用 558 帧。固定大厅相机与传感器读数都来自同一采集。

D 线实测支撑面的代理物体接触为 51/90 控制步。E 线 RGB 跟踪的共同有效标签为 46558/47582 帧。生成轨迹数为 1536，对应 230400 个控制时刻。小预测器的 15 项条件与种子组合累计训练 300000 步。

资源快照含 25 核 CPU 配额，GPU 利用率 100%，整卡已用显存 17.38 GiB，cgroup 总用量 100.88 GiB，清洁非活跃文件页 25.53 GiB。逐阶段错误与重试见 run 下 `state/coordinator-events.jsonl` 和各 worker 日志。

本次产物身份、字节数、条件分支和运行 PID 见 [实测状态](reports/interactive-world/latest.json)。在项目服务器根目录运行 `bash scripts/launch_interactive_world.sh status` 可读取最新实态。

<!-- interactive-world-evidence cae91e96f179375f -->


### 2026-11-28T19:38:39+08:00｜交互世界服务器实测

运行 `interactive-world-20261128-v1` 使用逐文件哈希核验的大厅、VR、DN 环境和原生深度。采样时仓库提交为 `4fce6619f49b1662700508b0bdf80ac859cd8200`。当前正式训练累计下界 146,400 步，其中可从已落盘 checkpoint 恢复 130,000 步。

| 训练组 | 阶段 | 已执行步数下界 | 可恢复步数 | PID | 调度实态 |
|---|---|---:|---:|---:|---|
| R0C | train | 56,100 | 50,000 | 15993 | running |
| R1 | train | 49,800 | 40,000 | 16581 | running |
| R2 | train | 0 | 0 | — | three heavy workers already active |
| R3 | train | 0 | 0 | — | waiting for a_depth/supervision/R3/manifest.json |
| R4 | train | 0 | 0 | — | waiting for a_depth/supervision/R3/manifest.json |
| R5 | train | 0 | 0 | — | waiting for a_depth/supervision/R3/manifest.json |
| R6 | train | 0 | 0 | — | awaiting independent pose audit |
| R7 | train | 0 | 0 | — | awaiting independent pose audit |
| VR0C | evaluate | 30,000 | 30,000 | — | evaluation receipt present |
| VR1 | train | 10,500 | 10,000 | 33189 | running |
| VR2 | train | 0 | 0 | — | three heavy workers already active |

A 线 sensor-only 在 1,478 个训练帧的 901,836,249 个自然孔洞中接受 509421 个。B 线有序流已记录 1478 帧，独立训练时组的尺度审计使用 558 帧。固定大厅相机与传感器读数都来自同一采集。

D 线实测支撑面的代理物体接触为 51/90 控制步。E 线 RGB 跟踪的共同有效标签为 46558/47582 帧。生成轨迹数为 1536，对应 230400 个控制时刻。小预测器的 15 项条件与种子组合累计训练 300000 步。

资源快照含 25 核 CPU 配额，GPU 利用率 100%，整卡已用显存 17.07 GiB，cgroup 总用量 101.84 GiB，清洁非活跃文件页 26.25 GiB。逐阶段错误与重试见 run 下 `state/coordinator-events.jsonl` 和各 worker 日志。

本次产物身份、字节数、条件分支和运行 PID 见 [实测状态](reports/interactive-world/latest.json)。在项目服务器根目录运行 `bash scripts/launch_interactive_world.sh status` 可读取最新实态。

<!-- interactive-world-evidence 88b364e1b9f4d815 -->


### 2026-11-28T19:39:40+08:00｜交互世界服务器实测

运行 `interactive-world-20261128-v1` 使用逐文件哈希核验的大厅、VR、DN 环境和原生深度。采样时仓库提交为 `4fce6619f49b1662700508b0bdf80ac859cd8200`。当前正式训练累计下界 148,200 步，其中可从已落盘 checkpoint 恢复 140,000 步。

| 训练组 | 阶段 | 已执行步数下界 | 可恢复步数 | PID | 调度实态 |
|---|---|---:|---:|---:|---|
| R0C | train | 56,700 | 50,000 | 15993 | running |
| R1 | train | 50,300 | 50,000 | 16581 | running |
| R2 | train | 0 | 0 | — | three heavy workers already active |
| R3 | train | 0 | 0 | — | waiting for a_depth/supervision/R3/manifest.json |
| R4 | train | 0 | 0 | — | waiting for a_depth/supervision/R3/manifest.json |
| R5 | train | 0 | 0 | — | waiting for a_depth/supervision/R3/manifest.json |
| R6 | train | 0 | 0 | — | awaiting independent pose audit |
| R7 | train | 0 | 0 | — | awaiting independent pose audit |
| VR0C | evaluate | 30,000 | 30,000 | — | evaluation receipt present |
| VR1 | train | 11,200 | 10,000 | 33189 | running |
| VR2 | train | 0 | 0 | — | three heavy workers already active |

A 线 sensor-only 在 1,478 个训练帧的 901,836,249 个自然孔洞中接受 509421 个。B 线有序流已记录 1478 帧，独立训练时组的尺度审计使用 558 帧。固定大厅相机与传感器读数都来自同一采集。

D 线实测支撑面的代理物体接触为 51/90 控制步。E 线 RGB 跟踪的共同有效标签为 46558/47582 帧。生成轨迹数为 1536，对应 230400 个控制时刻。小预测器的 15 项条件与种子组合累计训练 300000 步。

资源快照含 25 核 CPU 配额，GPU 利用率 100%，整卡已用显存 17.28 GiB，cgroup 总用量 102.58 GiB，清洁非活跃文件页 26.91 GiB。逐阶段错误与重试见 run 下 `state/coordinator-events.jsonl` 和各 worker 日志。

本次产物身份、字节数、条件分支和运行 PID 见 [实测状态](reports/interactive-world/latest.json)。在项目服务器根目录运行 `bash scripts/launch_interactive_world.sh status` 可读取最新实态。

<!-- interactive-world-evidence 36749b6e4909dbe8 -->


### 2026-11-28T19:45:41+08:00｜交互世界服务器实测

运行 `interactive-world-20261128-v1` 使用逐文件哈希核验的大厅、VR、DN 环境和原生深度。采样时仓库提交为 `4fce6619f49b1662700508b0bdf80ac859cd8200`。当前正式训练累计下界 159,200 步，其中可从已落盘 checkpoint 恢复 150,000 步。

| 训练组 | 阶段 | 已执行步数下界 | 可恢复步数 | PID | 调度实态 |
|---|---|---:|---:|---:|---|
| R0C | train | 60,100 | 60,000 | 15993 | running |
| R1 | train | 53,700 | 50,000 | 16581 | running |
| R2 | train | 0 | 0 | — | three heavy workers already active |
| R3 | train | 0 | 0 | — | waiting for a_depth/supervision/R3/manifest.json |
| R4 | train | 0 | 0 | — | waiting for a_depth/supervision/R3/manifest.json |
| R5 | train | 0 | 0 | — | waiting for a_depth/supervision/R3/manifest.json |
| R6 | train | 0 | 0 | — | awaiting independent pose audit |
| R7 | train | 0 | 0 | — | awaiting independent pose audit |
| VR0C | evaluate | 30,000 | 30,000 | — | evaluation receipt present |
| VR1 | train | 15,400 | 10,000 | 33189 | running |
| VR2 | train | 0 | 0 | — | three heavy workers already active |

A 线 sensor-only 在 1,478 个训练帧的 901,836,249 个自然孔洞中接受 509421 个。B 线有序流已记录 1478 帧，独立训练时组的尺度审计使用 558 帧。固定大厅相机与传感器读数都来自同一采集。

D 线实测支撑面的代理物体接触为 51/90 控制步。E 线 RGB 跟踪的共同有效标签为 46558/47582 帧。生成轨迹数为 1536，对应 230400 个控制时刻。小预测器的 15 项条件与种子组合累计训练 300000 步。

资源快照含 25 核 CPU 配额，GPU 利用率 100%，整卡已用显存 16.48 GiB，cgroup 总用量 104.87 GiB，清洁非活跃文件页 27.18 GiB。逐阶段错误与重试见 run 下 `state/coordinator-events.jsonl` 和各 worker 日志。

本次产物身份、字节数、条件分支和运行 PID 见 [实测状态](reports/interactive-world/latest.json)。在项目服务器根目录运行 `bash scripts/launch_interactive_world.sh status` 可读取最新实态。

<!-- interactive-world-evidence 5156aff84375afec -->


### 2026-11-28T19:47:42+08:00｜交互世界服务器实测

运行 `interactive-world-20261128-v1` 使用逐文件哈希核验的大厅、VR、DN 环境和原生深度。采样时仓库提交为 `4fce6619f49b1662700508b0bdf80ac859cd8200`。当前正式训练累计下界 163,200 步，其中可从已落盘 checkpoint 恢复 150,000 步。

| 训练组 | 阶段 | 已执行步数下界 | 可恢复步数 | PID | 调度实态 |
|---|---|---:|---:|---:|---|
| R0C | train | 61,300 | 60,000 | 15993 | running |
| R1 | train | 54,900 | 50,000 | 16581 | running |
| R2 | train | 0 | 0 | — | three heavy workers already active |
| R3 | train | 0 | 0 | — | waiting for a_depth/supervision/R3/manifest.json |
| R4 | train | 0 | 0 | — | waiting for a_depth/supervision/R3/manifest.json |
| R5 | train | 0 | 0 | — | waiting for a_depth/supervision/R3/manifest.json |
| R6 | train | 0 | 0 | — | awaiting independent pose audit |
| R7 | train | 0 | 0 | — | awaiting independent pose audit |
| VR0C | evaluate | 30,000 | 30,000 | — | evaluation receipt present |
| VR1 | train | 17,000 | 10,000 | 33189 | running |
| VR2 | train | 0 | 0 | — | three heavy workers already active |

A 线 sensor-only 在 1,478 个训练帧的 901,836,249 个自然孔洞中接受 509421 个。B 线有序流已记录 1478 帧，独立训练时组的尺度审计使用 558 帧。固定大厅相机与传感器读数都来自同一采集。

D 线实测支撑面的代理物体接触为 51/90 控制步。E 线 RGB 跟踪的共同有效标签为 46558/47582 帧。生成轨迹数为 1536，对应 230400 个控制时刻。小预测器的 15 项条件与种子组合累计训练 300000 步。

资源快照含 25 核 CPU 配额，GPU 利用率 100%，整卡已用显存 16.07 GiB，cgroup 总用量 103.65 GiB，清洁非活跃文件页 27.23 GiB。逐阶段错误与重试见 run 下 `state/coordinator-events.jsonl` 和各 worker 日志。

本次产物身份、字节数、条件分支和运行 PID 见 [实测状态](reports/interactive-world/latest.json)。在项目服务器根目录运行 `bash scripts/launch_interactive_world.sh status` 可读取最新实态。

<!-- interactive-world-evidence b18a53dc5a161cb8 -->


### 2026-11-28T19:51:43+08:00｜交互世界服务器实测

运行 `interactive-world-20261128-v1` 使用逐文件哈希核验的大厅、VR、DN 环境和原生深度。采样时仓库提交为 `4fce6619f49b1662700508b0bdf80ac859cd8200`。当前正式训练累计下界 170,800 步，其中可从已落盘 checkpoint 恢复 160,000 步。

| 训练组 | 阶段 | 已执行步数下界 | 可恢复步数 | PID | 调度实态 |
|---|---|---:|---:|---:|---|
| R0C | train | 63,700 | 60,000 | 15993 | running |
| R1 | train | 57,100 | 50,000 | 16581 | running |
| R2 | train | 0 | 0 | — | three heavy workers already active |
| R3 | train | 0 | 0 | — | waiting for a_depth/supervision/R3/manifest.json |
| R4 | train | 0 | 0 | — | waiting for a_depth/supervision/R3/manifest.json |
| R5 | train | 0 | 0 | — | waiting for a_depth/supervision/R3/manifest.json |
| R6 | train | 0 | 0 | — | awaiting independent pose audit |
| R7 | train | 0 | 0 | — | awaiting independent pose audit |
| VR0C | evaluate | 30,000 | 30,000 | — | evaluation receipt present |
| VR1 | train | 20,000 | 20,000 | 33189 | running |
| VR2 | train | 0 | 0 | — | three heavy workers already active |

A 线 sensor-only 在 1,478 个训练帧的 901,836,249 个自然孔洞中接受 509421 个。B 线有序流已记录 1478 帧，独立训练时组的尺度审计使用 558 帧。固定大厅相机与传感器读数都来自同一采集。

D 线实测支撑面的代理物体接触为 51/90 控制步。E 线 RGB 跟踪的共同有效标签为 46558/47582 帧。生成轨迹数为 1536，对应 230400 个控制时刻。小预测器的 15 项条件与种子组合累计训练 300000 步。

资源快照含 25 核 CPU 配额，GPU 利用率 99%，整卡已用显存 16.63 GiB，cgroup 总用量 106.19 GiB，清洁非活跃文件页 27.34 GiB。逐阶段错误与重试见 run 下 `state/coordinator-events.jsonl` 和各 worker 日志。

本次产物身份、字节数、条件分支和运行 PID 见 [实测状态](reports/interactive-world/latest.json)。在项目服务器根目录运行 `bash scripts/launch_interactive_world.sh status` 可读取最新实态。

<!-- interactive-world-evidence 4900f0347d743177 -->


### 2026-11-28T19:57:45+08:00｜交互世界服务器实测

运行 `interactive-world-20261128-v1` 使用逐文件哈希核验的大厅、VR、DN 环境和原生深度。采样时仓库提交为 `4fce6619f49b1662700508b0bdf80ac859cd8200`。当前正式训练累计下界 181,800 步，其中可从已落盘 checkpoint 恢复 170,000 步。

| 训练组 | 阶段 | 已执行步数下界 | 可恢复步数 | PID | 调度实态 |
|---|---|---:|---:|---:|---|
| R0C | train | 67,200 | 60,000 | 15993 | running |
| R1 | train | 60,100 | 60,000 | 16581 | running |
| R2 | train | 0 | 0 | — | three heavy workers already active |
| R3 | train | 0 | 0 | — | waiting for a_depth/supervision/R3/manifest.json |
| R4 | train | 0 | 0 | — | waiting for a_depth/supervision/R3/manifest.json |
| R5 | train | 0 | 0 | — | waiting for a_depth/supervision/R3/manifest.json |
| R6 | train | 0 | 0 | — | awaiting independent pose audit |
| R7 | train | 0 | 0 | — | awaiting independent pose audit |
| VR0C | evaluate | 30,000 | 30,000 | — | evaluation receipt present |
| VR1 | train | 24,500 | 20,000 | 33189 | running |
| VR2 | train | 0 | 0 | — | three heavy workers already active |

A 线 sensor-only 在 1,478 个训练帧的 901,836,249 个自然孔洞中接受 509421 个。B 线有序流已记录 1478 帧，独立训练时组的尺度审计使用 558 帧。固定大厅相机与传感器读数都来自同一采集。

D 线实测支撑面的代理物体接触为 51/90 控制步。E 线 RGB 跟踪的共同有效标签为 46558/47582 帧。生成轨迹数为 1536，对应 230400 个控制时刻。小预测器的 15 项条件与种子组合累计训练 300000 步。

资源快照含 25 核 CPU 配额，GPU 利用率 99%，整卡已用显存 16.51 GiB，cgroup 总用量 105.57 GiB，清洁非活跃文件页 27.94 GiB。逐阶段错误与重试见 run 下 `state/coordinator-events.jsonl` 和各 worker 日志。

本次产物身份、字节数、条件分支和运行 PID 见 [实测状态](reports/interactive-world/latest.json)。在项目服务器根目录运行 `bash scripts/launch_interactive_world.sh status` 可读取最新实态。

<!-- interactive-world-evidence ceec288275531307 -->


### 2026-11-28T19:59:46+08:00｜交互世界服务器实测

运行 `interactive-world-20261128-v1` 使用逐文件哈希核验的大厅、VR、DN 环境和原生深度。采样时仓库提交为 `4fce6619f49b1662700508b0bdf80ac859cd8200`。当前正式训练累计下界 185,500 步，其中可从已落盘 checkpoint 恢复 170,000 步。

| 训练组 | 阶段 | 已执行步数下界 | 可恢复步数 | PID | 调度实态 |
|---|---|---:|---:|---:|---|
| R0C | train | 68,400 | 60,000 | 15993 | running |
| R1 | train | 61,200 | 60,000 | 16581 | running |
| R2 | train | 0 | 0 | — | three heavy workers already active |
| R3 | train | 0 | 0 | — | waiting for a_depth/supervision/R3/manifest.json |
| R4 | train | 0 | 0 | — | waiting for a_depth/supervision/R3/manifest.json |
| R5 | train | 0 | 0 | — | waiting for a_depth/supervision/R3/manifest.json |
| R6 | train | 0 | 0 | — | awaiting independent pose audit |
| R7 | train | 0 | 0 | — | awaiting independent pose audit |
| VR0C | evaluate | 30,000 | 30,000 | — | evaluation receipt present |
| VR1 | train | 25,900 | 20,000 | 33189 | running |
| VR2 | train | 0 | 0 | — | three heavy workers already active |

A 线 sensor-only 在 1,478 个训练帧的 901,836,249 个自然孔洞中接受 509421 个。B 线有序流已记录 1478 帧，独立训练时组的尺度审计使用 558 帧。固定大厅相机与传感器读数都来自同一采集。

D 线实测支撑面的代理物体接触为 51/90 控制步。E 线 RGB 跟踪的共同有效标签为 46558/47582 帧。生成轨迹数为 1536，对应 230400 个控制时刻。小预测器的 15 项条件与种子组合累计训练 300000 步。

资源快照含 25 核 CPU 配额，GPU 利用率 99%，整卡已用显存 16.51 GiB，cgroup 总用量 105.58 GiB，清洁非活跃文件页 27.94 GiB。逐阶段错误与重试见 run 下 `state/coordinator-events.jsonl` 和各 worker 日志。

本次产物身份、字节数、条件分支和运行 PID 见 [实测状态](reports/interactive-world/latest.json)。在项目服务器根目录运行 `bash scripts/launch_interactive_world.sh status` 可读取最新实态。

<!-- interactive-world-evidence e2ad21514abcea52 -->


### 2026-11-28T20:02:47+08:00｜交互世界服务器实测

运行 `interactive-world-20261128-v1` 使用逐文件哈希核验的大厅、VR、DN 环境和原生深度。采样时仓库提交为 `4fce6619f49b1662700508b0bdf80ac859cd8200`。当前正式训练累计下界 191,500 步，其中可从已落盘 checkpoint 恢复 180,000 步。

| 训练组 | 阶段 | 已执行步数下界 | 可恢复步数 | PID | 调度实态 |
|---|---|---:|---:|---:|---|
| R0C | train | 70,300 | 70,000 | 15993 | running |
| R1 | train | 63,000 | 60,000 | 16581 | running |
| R2 | train | 0 | 0 | — | three heavy workers already active |
| R3 | train | 0 | 0 | — | waiting for a_depth/supervision/R3/manifest.json |
| R4 | train | 0 | 0 | — | waiting for a_depth/supervision/R3/manifest.json |
| R5 | train | 0 | 0 | — | waiting for a_depth/supervision/R3/manifest.json |
| R6 | train | 0 | 0 | — | awaiting independent pose audit |
| R7 | train | 0 | 0 | — | awaiting independent pose audit |
| VR0C | evaluate | 30,000 | 30,000 | — | evaluation receipt present |
| VR1 | train | 28,200 | 20,000 | 33189 | running |
| VR2 | train | 0 | 0 | — | three heavy workers already active |

A 线 sensor-only 在 1,478 个训练帧的 901,836,249 个自然孔洞中接受 509421 个。B 线有序流已记录 1478 帧，独立训练时组的尺度审计使用 558 帧。固定大厅相机与传感器读数都来自同一采集。

D 线实测支撑面的代理物体接触为 51/90 控制步。E 线 RGB 跟踪的共同有效标签为 46558/47582 帧。生成轨迹数为 1536，对应 230400 个控制时刻。小预测器的 15 项条件与种子组合累计训练 300000 步。

资源快照含 25 核 CPU 配额，GPU 利用率 99%，整卡已用显存 16.39 GiB，cgroup 总用量 106.97 GiB，清洁非活跃文件页 28.55 GiB。逐阶段错误与重试见 run 下 `state/coordinator-events.jsonl` 和各 worker 日志。

本次产物身份、字节数、条件分支和运行 PID 见 [实测状态](reports/interactive-world/latest.json)。在项目服务器根目录运行 `bash scripts/launch_interactive_world.sh status` 可读取最新实态。

<!-- interactive-world-evidence e7d5615ab8ff938b -->


### 2026-11-28T20:05:48+08:00｜交互世界服务器实测

运行 `interactive-world-20261128-v1` 使用逐文件哈希核验的大厅、VR、DN 环境和原生深度。采样时仓库提交为 `4fce6619f49b1662700508b0bdf80ac859cd8200`。当前正式训练累计下界 197,200 步，其中可从已落盘 checkpoint 恢复 190,000 步。

| 训练组 | 阶段 | 已执行步数下界 | 可恢复步数 | PID | 调度实态 |
|---|---|---:|---:|---:|---|
| R0C | train | 72,300 | 70,000 | 15993 | running |
| R1 | train | 64,900 | 60,000 | 16581 | running |
| R2 | train | 0 | 0 | — | three heavy workers already active |
| R3 | train | 0 | 0 | — | waiting for a_depth/supervision/R3/manifest.json |
| R4 | train | 0 | 0 | — | waiting for a_depth/supervision/R3/manifest.json |
| R5 | train | 0 | 0 | — | waiting for a_depth/supervision/R3/manifest.json |
| R6 | train | 0 | 0 | — | awaiting independent pose audit |
| R7 | train | 0 | 0 | — | awaiting independent pose audit |
| VR0C | evaluate | 30,000 | 30,000 | — | evaluation receipt present |
| VR1 | evaluate | 30,000 | 30,000 | 35279 | running |
| VR2 | train | 0 | 0 | — | three heavy workers already active |

A 线 sensor-only 在 1,478 个训练帧的 901,836,249 个自然孔洞中接受 509421 个。B 线有序流已记录 1478 帧，独立训练时组的尺度审计使用 558 帧。固定大厅相机与传感器读数都来自同一采集。

D 线实测支撑面的代理物体接触为 51/90 控制步。E 线 RGB 跟踪的共同有效标签为 46558/47582 帧。生成轨迹数为 1536，对应 230400 个控制时刻。小预测器的 15 项条件与种子组合累计训练 300000 步。

资源快照含 25 核 CPU 配额，GPU 利用率 98%，整卡已用显存 13.16 GiB，cgroup 总用量 107.39 GiB，清洁非活跃文件页 28.72 GiB。逐阶段错误与重试见 run 下 `state/coordinator-events.jsonl` 和各 worker 日志。

本次产物身份、字节数、条件分支和运行 PID 见 [实测状态](reports/interactive-world/latest.json)。在项目服务器根目录运行 `bash scripts/launch_interactive_world.sh status` 可读取最新实态。

<!-- interactive-world-evidence 9c419c2a794a08d1 -->


### 2026-11-28T20:07:49+08:00｜交互世界服务器实测

运行 `interactive-world-20261128-v1` 使用逐文件哈希核验的大厅、VR、DN 环境和原生深度。采样时仓库提交为 `4fce6619f49b1662700508b0bdf80ac859cd8200`。当前正式训练累计下界 200,100 步，其中可从已落盘 checkpoint 恢复 190,000 步。

| 训练组 | 阶段 | 已执行步数下界 | 可恢复步数 | PID | 调度实态 |
|---|---|---:|---:|---:|---|
| R0C | train | 73,800 | 70,000 | 15993 | running |
| R1 | train | 66,300 | 60,000 | 16581 | running |
| R2 | train | 0 | 0 | — | three heavy workers already active |
| R3 | train | 0 | 0 | — | waiting for a_depth/supervision/R3/manifest.json |
| R4 | train | 0 | 0 | — | waiting for a_depth/supervision/R3/manifest.json |
| R5 | train | 0 | 0 | — | waiting for a_depth/supervision/R3/manifest.json |
| R6 | train | 0 | 0 | — | awaiting independent pose audit |
| R7 | train | 0 | 0 | — | awaiting independent pose audit |
| VR0C | evaluate | 30,000 | 30,000 | — | evaluation receipt present |
| VR1 | evaluate | 30,000 | 30,000 | 35279 | running |
| VR2 | train | 0 | 0 | — | three heavy workers already active |

A 线 sensor-only 在 1,478 个训练帧的 901,836,249 个自然孔洞中接受 509421 个。B 线有序流已记录 1478 帧，独立训练时组的尺度审计使用 558 帧。固定大厅相机与传感器读数都来自同一采集。

D 线实测支撑面的代理物体接触为 51/90 控制步。E 线 RGB 跟踪的共同有效标签为 46558/47582 帧。生成轨迹数为 1536，对应 230400 个控制时刻。小预测器的 15 项条件与种子组合累计训练 300000 步。

资源快照含 25 核 CPU 配额，GPU 利用率 97%，整卡已用显存 16.68 GiB，cgroup 总用量 118.00 GiB，清洁非活跃文件页 23.14 GiB。逐阶段错误与重试见 run 下 `state/coordinator-events.jsonl` 和各 worker 日志。

本次产物身份、字节数、条件分支和运行 PID 见 [实测状态](reports/interactive-world/latest.json)。在项目服务器根目录运行 `bash scripts/launch_interactive_world.sh status` 可读取最新实态。

<!-- interactive-world-evidence 974e636ef4a89f1e -->


### 2026-11-28T20:13:51+08:00｜交互世界服务器实测

运行 `interactive-world-20261128-v1` 使用逐文件哈希核验的大厅、VR、DN 环境和原生深度。采样时仓库提交为 `4fce6619f49b1662700508b0bdf80ac859cd8200`。当前正式训练累计下界 208,500 步，其中可从已落盘 checkpoint 恢复 200,000 步。

| 训练组 | 阶段 | 已执行步数下界 | 可恢复步数 | PID | 调度实态 |
|---|---|---:|---:|---:|---|
| R0C | train | 78,100 | 70,000 | 15993 | running |
| R1 | train | 70,400 | 70,000 | 16581 | running |
| R2 | train | 0 | 0 | — | three heavy workers already active |
| R3 | train | 0 | 0 | — | waiting for a_depth/supervision/R3/manifest.json |
| R4 | train | 0 | 0 | — | waiting for a_depth/supervision/R3/manifest.json |
| R5 | train | 0 | 0 | — | waiting for a_depth/supervision/R3/manifest.json |
| R6 | train | 0 | 0 | — | awaiting independent pose audit |
| R7 | train | 0 | 0 | — | awaiting independent pose audit |
| VR0C | evaluate | 30,000 | 30,000 | — | evaluation receipt present |
| VR1 | evaluate | 30,000 | 30,000 | 35279 | running |
| VR2 | train | 0 | 0 | — | three heavy workers already active |

A 线 sensor-only 在 1,478 个训练帧的 901,836,249 个自然孔洞中接受 509421 个。B 线有序流已记录 1478 帧，独立训练时组的尺度审计使用 558 帧。固定大厅相机与传感器读数都来自同一采集。

D 线实测支撑面的代理物体接触为 51/90 控制步。E 线 RGB 跟踪的共同有效标签为 46558/47582 帧。生成轨迹数为 1536，对应 230400 个控制时刻。小预测器的 15 项条件与种子组合累计训练 300000 步。

资源快照含 25 核 CPU 配额，GPU 利用率 100%，整卡已用显存 13.54 GiB，cgroup 总用量 91.80 GiB，清洁非活跃文件页 17.68 GiB。逐阶段错误与重试见 run 下 `state/coordinator-events.jsonl` 和各 worker 日志。

本次产物身份、字节数、条件分支和运行 PID 见 [实测状态](reports/interactive-world/latest.json)。在项目服务器根目录运行 `bash scripts/launch_interactive_world.sh status` 可读取最新实态。

<!-- interactive-world-evidence 4935383fd8700be3 -->


### 2026-11-28T20:16:31+08:00｜交互世界服务器实测

运行 `interactive-world-20261128-v1` 使用逐文件哈希核验的大厅、VR、DN 环境和原生深度。采样时仓库提交为 `4fce6619f49b1662700508b0bdf80ac859cd8200`。当前正式训练累计下界 212,300 步，其中可从已落盘 checkpoint 恢复 210,000 步。

| 训练组 | 阶段 | 已执行步数下界 | 可恢复步数 | PID | 调度实态 |
|---|---|---:|---:|---:|---|
| R0C | train | 80,000 | 80,000 | 15993 | running |
| R1 | train | 72,300 | 70,000 | 16581 | running |
| R2 | train | 0 | 0 | — | three heavy workers already active |
| R3 | train | 0 | 0 | — | waiting for a_depth/supervision/R3/manifest.json |
| R4 | train | 0 | 0 | — | waiting for a_depth/supervision/R3/manifest.json |
| R5 | train | 0 | 0 | — | waiting for a_depth/supervision/R3/manifest.json |
| R6 | train | 0 | 0 | — | awaiting independent pose audit |
| R7 | train | 0 | 0 | — | awaiting independent pose audit |
| VR0C | evaluate | 30,000 | 30,000 | — | evaluation receipt present |
| VR1 | evaluate | 30,000 | 30,000 | 35279 | running |
| VR2 | train | 0 | 0 | — | three heavy workers already active |

A 线 sensor-only 在 1,478 个训练帧的 901,836,249 个自然孔洞中接受 509421 个。B 线有序流已记录 1478 帧，独立训练时组的尺度审计使用 558 帧。固定大厅相机与传感器读数都来自同一采集。

D 线实测支撑面的代理物体接触为 51/90 控制步。E 线 RGB 跟踪的共同有效标签为 46558/47582 帧。生成轨迹数为 1536，对应 230400 个控制时刻。小预测器的 15 项条件与种子组合累计训练 300000 步。

资源快照含 25 核 CPU 配额，GPU 利用率 99%，整卡已用显存 13.42 GiB，cgroup 总用量 92.70 GiB，清洁非活跃文件页 18.31 GiB。逐阶段错误与重试见 run 下 `state/coordinator-events.jsonl` 和各 worker 日志。

本次产物身份、字节数、条件分支和运行 PID 见 [实测状态](reports/interactive-world/latest.json)。在项目服务器根目录运行 `bash scripts/launch_interactive_world.sh status` 可读取最新实态。

<!-- interactive-world-evidence 56e9b256b99c6db8 -->


### 2026-11-28T20:17:32+08:00｜交互世界服务器实测

运行 `interactive-world-20261128-v1` 使用逐文件哈希核验的大厅、VR、DN 环境和原生深度。采样时仓库提交为 `4fce6619f49b1662700508b0bdf80ac859cd8200`。当前正式训练累计下界 213,800 步，其中可从已落盘 checkpoint 恢复 210,000 步。

| 训练组 | 阶段 | 已执行步数下界 | 可恢复步数 | PID | 调度实态 |
|---|---|---:|---:|---:|---|
| R0C | train | 80,800 | 80,000 | 15993 | running |
| R1 | train | 73,000 | 70,000 | 16581 | running |
| R2 | train | 0 | 0 | — | three heavy workers already active |
| R3 | train | 0 | 0 | — | waiting for a_depth/supervision/R3/manifest.json |
| R4 | train | 0 | 0 | — | waiting for a_depth/supervision/R3/manifest.json |
| R5 | train | 0 | 0 | — | waiting for a_depth/supervision/R3/manifest.json |
| R6 | train | 0 | 0 | — | awaiting independent pose audit |
| R7 | train | 0 | 0 | — | awaiting independent pose audit |
| VR0C | evaluate | 30,000 | 30,000 | — | evaluation receipt present |
| VR1 | evaluate | 30,000 | 30,000 | 35279 | running |
| VR2 | train | 0 | 0 | — | three heavy workers already active |

A 线 sensor-only 在 1,478 个训练帧的 901,836,249 个自然孔洞中接受 509421 个。B 线有序流已记录 1478 帧，独立训练时组的尺度审计使用 558 帧。固定大厅相机与传感器读数都来自同一采集。

D 线实测支撑面的代理物体接触为 51/90 控制步。E 线 RGB 跟踪的共同有效标签为 46558/47582 帧。生成轨迹数为 1536，对应 230400 个控制时刻。小预测器的 15 项条件与种子组合累计训练 300000 步。

资源快照含 25 核 CPU 配额，GPU 利用率 99%，整卡已用显存 12.70 GiB，cgroup 总用量 97.12 GiB，清洁非活跃文件页 18.21 GiB。逐阶段错误与重试见 run 下 `state/coordinator-events.jsonl` 和各 worker 日志。

本次产物身份、字节数、条件分支和运行 PID 见 [实测状态](reports/interactive-world/latest.json)。在项目服务器根目录运行 `bash scripts/launch_interactive_world.sh status` 可读取最新实态。

<!-- interactive-world-evidence 3f172c3add8a5e6f -->


### 2026-11-28T20:25:34+08:00｜交互世界服务器实测

运行 `interactive-world-20261128-v1` 使用逐文件哈希核验的大厅、VR、DN 环境和原生深度。采样时仓库提交为 `4fce6619f49b1662700508b0bdf80ac859cd8200`。当前正式训练累计下界 219,100 步，其中可从已落盘 checkpoint 恢复 210,000 步。

| 训练组 | 阶段 | 已执行步数下界 | 可恢复步数 | PID | 调度实态 |
|---|---|---:|---:|---:|---|
| R0C | train | 83,600 | 80,000 | 15993 | running |
| R1 | train | 75,500 | 70,000 | 16581 | running |
| R2 | train | 0 | 0 | — | three heavy workers already active |
| R3 | train | 0 | 0 | — | waiting for a_depth/supervision/R3/manifest.json |
| R4 | train | 0 | 0 | — | waiting for a_depth/supervision/R3/manifest.json |
| R5 | train | 0 | 0 | — | waiting for a_depth/supervision/R3/manifest.json |
| R6 | train | 0 | 0 | — | awaiting independent pose audit |
| R7 | train | 0 | 0 | — | awaiting independent pose audit |
| VR0C | evaluate | 30,000 | 30,000 | — | evaluation receipt present |
| VR1 | evaluate | 30,000 | 30,000 | 35279 | running |
| VR2 | train | 0 | 0 | — | three heavy workers already active |

A 线 sensor-only 在 1,478 个训练帧的 901,836,249 个自然孔洞中接受 509421 个。B 线有序流已记录 1478 帧，独立训练时组的尺度审计使用 558 帧。固定大厅相机与传感器读数都来自同一采集。

D 线实测支撑面的代理物体接触为 51/90 控制步。E 线 RGB 跟踪的共同有效标签为 46558/47582 帧。生成轨迹数为 1536，对应 230400 个控制时刻。小预测器的 15 项条件与种子组合累计训练 300000 步。

资源快照含 25 核 CPU 配额，GPU 利用率 100%，整卡已用显存 15.61 GiB，cgroup 总用量 98.49 GiB，清洁非活跃文件页 17.83 GiB。逐阶段错误与重试见 run 下 `state/coordinator-events.jsonl` 和各 worker 日志。

本次产物身份、字节数、条件分支和运行 PID 见 [实测状态](reports/interactive-world/latest.json)。在项目服务器根目录运行 `bash scripts/launch_interactive_world.sh status` 可读取最新实态。

<!-- interactive-world-evidence 55b1dc339cfcf1ff -->


### 2026-11-28T20:37:38+08:00｜交互世界服务器实测

运行 `interactive-world-20261128-v1` 使用逐文件哈希核验的大厅、VR、DN 环境和原生深度。采样时仓库提交为 `4fce6619f49b1662700508b0bdf80ac859cd8200`。当前正式训练累计下界 228,900 步，其中可从已落盘 checkpoint 恢复 220,000 步。

| 训练组 | 阶段 | 已执行步数下界 | 可恢复步数 | PID | 调度实态 |
|---|---|---:|---:|---:|---|
| R0C | train | 88,800 | 80,000 | 15993 | running |
| R1 | train | 80,100 | 80,000 | 16581 | running |
| R2 | train | 0 | 0 | — | three heavy workers already active |
| R3 | train | 0 | 0 | — | waiting for a_depth/supervision/R3/manifest.json |
| R4 | train | 0 | 0 | — | waiting for a_depth/supervision/R3/manifest.json |
| R5 | train | 0 | 0 | — | waiting for a_depth/supervision/R3/manifest.json |
| R6 | train | 0 | 0 | — | awaiting independent pose audit |
| R7 | train | 0 | 0 | — | awaiting independent pose audit |
| VR0C | evaluate | 30,000 | 30,000 | — | evaluation receipt present |
| VR1 | evaluate | 30,000 | 30,000 | 35279 | running |
| VR2 | train | 0 | 0 | — | three heavy workers already active |

A 线 sensor-only 在 1,478 个训练帧的 901,836,249 个自然孔洞中接受 509421 个。B 线有序流已记录 1478 帧，独立训练时组的尺度审计使用 558 帧。固定大厅相机与传感器读数都来自同一采集。

D 线实测支撑面的代理物体接触为 51/90 控制步。E 线 RGB 跟踪的共同有效标签为 46558/47582 帧。生成轨迹数为 1536，对应 230400 个控制时刻。小预测器的 15 项条件与种子组合累计训练 300000 步。

资源快照含 25 核 CPU 配额，GPU 利用率 99%，整卡已用显存 15.57 GiB，cgroup 总用量 99.53 GiB，清洁非活跃文件页 18.12 GiB。逐阶段错误与重试见 run 下 `state/coordinator-events.jsonl` 和各 worker 日志。

本次产物身份、字节数、条件分支和运行 PID 见 [实测状态](reports/interactive-world/latest.json)。在项目服务器根目录运行 `bash scripts/launch_interactive_world.sh status` 可读取最新实态。

<!-- interactive-world-evidence d28606e0cb9c5585 -->


### 2026-11-28T20:39:38+08:00｜交互世界服务器实测

运行 `interactive-world-20261128-v1` 使用逐文件哈希核验的大厅、VR、DN 环境和原生深度。采样时仓库提交为 `4fce6619f49b1662700508b0bdf80ac859cd8200`。当前正式训练累计下界 231,700 步，其中可从已落盘 checkpoint 恢复 230,000 步。

| 训练组 | 阶段 | 已执行步数下界 | 可恢复步数 | PID | 调度实态 |
|---|---|---:|---:|---:|---|
| R0C | train | 90,000 | 90,000 | 15993 | running |
| R1 | train | 81,700 | 80,000 | 16581 | running |
| R2 | train | 0 | 0 | — | three heavy workers already active |
| R3 | train | 0 | 0 | — | waiting for a_depth/supervision/R3/manifest.json |
| R4 | train | 0 | 0 | — | waiting for a_depth/supervision/R3/manifest.json |
| R5 | train | 0 | 0 | — | waiting for a_depth/supervision/R3/manifest.json |
| R6 | train | 0 | 0 | — | awaiting independent pose audit |
| R7 | train | 0 | 0 | — | awaiting independent pose audit |
| VR0C | evaluate | 30,000 | 30,000 | — | evaluation receipt present |
| VR1 | evaluate | 30,000 | 30,000 | 35279 | running |
| VR2 | train | 0 | 0 | — | three heavy workers already active |

A 线 sensor-only 在 1,478 个训练帧的 901,836,249 个自然孔洞中接受 509421 个。B 线有序流已记录 1478 帧，独立训练时组的尺度审计使用 558 帧。固定大厅相机与传感器读数都来自同一采集。

D 线实测支撑面的代理物体接触为 51/90 控制步。E 线 RGB 跟踪的共同有效标签为 46558/47582 帧。生成轨迹数为 1536，对应 230400 个控制时刻。小预测器的 15 项条件与种子组合累计训练 300000 步。

资源快照含 25 核 CPU 配额，GPU 利用率 99%，整卡已用显存 15.23 GiB，cgroup 总用量 102.41 GiB，清洁非活跃文件页 18.06 GiB。逐阶段错误与重试见 run 下 `state/coordinator-events.jsonl` 和各 worker 日志。

本次产物身份、字节数、条件分支和运行 PID 见 [实测状态](reports/interactive-world/latest.json)。在项目服务器根目录运行 `bash scripts/launch_interactive_world.sh status` 可读取最新实态。

<!-- interactive-world-evidence 735713e803e3296d -->


### 2026-11-28T20:45:40+08:00｜交互世界服务器实测

运行 `interactive-world-20261128-v1` 使用逐文件哈希核验的大厅、VR、DN 环境和原生深度。采样时仓库提交为 `4fce6619f49b1662700508b0bdf80ac859cd8200`。当前正式训练累计下界 235,500 步，其中可从已落盘 checkpoint 恢复 230,000 步。

| 训练组 | 阶段 | 已执行步数下界 | 可恢复步数 | PID | 调度实态 |
|---|---|---:|---:|---:|---|
| R0C | train | 92,000 | 90,000 | 15993 | running |
| R1 | train | 83,500 | 80,000 | 16581 | running |
| R2 | train | 0 | 0 | — | three heavy workers already active |
| R3 | train | 0 | 0 | — | three heavy workers already active |
| R4 | train | 0 | 0 | — | three heavy workers already active |
| R5 | train | 0 | 0 | — | three heavy workers already active |
| R6 | train | 0 | 0 | — | awaiting independent pose audit |
| R7 | train | 0 | 0 | — | awaiting independent pose audit |
| VR0C | evaluate | 30,000 | 30,000 | — | evaluation receipt present |
| VR1 | evaluate | 30,000 | 30,000 | 35279 | running |
| VR2 | train | 0 | 0 | — | three heavy workers already active |

A 线 sensor-only 在 1,478 个训练帧的 901,836,249 个自然孔洞中接受 509421 个。B 线有序流已记录 1478 帧，独立训练时组的尺度审计使用 558 帧。固定大厅相机与传感器读数都来自同一采集。

D 线实测支撑面的代理物体接触为 51/90 控制步。E 线 RGB 跟踪的共同有效标签为 46558/47582 帧。生成轨迹数为 1536，对应 230400 个控制时刻。小预测器的 15 项条件与种子组合累计训练 300000 步。

资源快照含 25 核 CPU 配额，GPU 利用率 100%，整卡已用显存 14.72 GiB，cgroup 总用量 100.73 GiB，清洁非活跃文件页 18.41 GiB。逐阶段错误与重试见 run 下 `state/coordinator-events.jsonl` 和各 worker 日志。

本次产物身份、字节数、条件分支和运行 PID 见 [实测状态](reports/interactive-world/latest.json)。在项目服务器根目录运行 `bash scripts/launch_interactive_world.sh status` 可读取最新实态。

<!-- interactive-world-evidence 473a1bc3b72be361 -->


### 2026-11-28T20:56:44+08:00｜交互世界服务器实测

运行 `interactive-world-20261128-v1` 使用逐文件哈希核验的大厅、VR、DN 环境和原生深度。采样时仓库提交为 `4fce6619f49b1662700508b0bdf80ac859cd8200`。当前正式训练累计下界 244,700 步，其中可从已落盘 checkpoint 恢复 230,000 步。

| 训练组 | 阶段 | 已执行步数下界 | 可恢复步数 | PID | 调度实态 |
|---|---|---:|---:|---:|---|
| R0C | train | 96,800 | 90,000 | 15993 | running |
| R1 | train | 87,900 | 80,000 | 16581 | running |
| R2 | train | 0 | 0 | — | three heavy workers already active |
| R3 | train | 0 | 0 | — | three heavy workers already active |
| R4 | train | 0 | 0 | — | three heavy workers already active |
| R5 | train | 0 | 0 | — | three heavy workers already active |
| R6 | train | 0 | 0 | — | awaiting independent pose audit |
| R7 | train | 0 | 0 | — | awaiting independent pose audit |
| VR0C | evaluate | 30,000 | 30,000 | — | evaluation receipt present |
| VR1 | evaluate | 30,000 | 30,000 | — | evaluation receipt present |
| VR2 | train | 0 | 0 | 46859 | running |

A 线 sensor-only 在 1,478 个训练帧的 901,836,249 个自然孔洞中接受 509421 个。B 线有序流已记录 1478 帧，独立训练时组的尺度审计使用 558 帧。固定大厅相机与传感器读数都来自同一采集。

D 线实测支撑面的代理物体接触为 51/90 控制步。E 线 RGB 跟踪的共同有效标签为 46558/47582 帧。生成轨迹数为 1536，对应 230400 个控制时刻。小预测器的 15 项条件与种子组合累计训练 300000 步。

资源快照含 25 核 CPU 配额，GPU 利用率 98%，整卡已用显存 12.07 GiB，cgroup 总用量 95.48 GiB，清洁非活跃文件页 18.37 GiB。逐阶段错误与重试见 run 下 `state/coordinator-events.jsonl` 和各 worker 日志。

本次产物身份、字节数、条件分支和运行 PID 见 [实测状态](reports/interactive-world/latest.json)。在项目服务器根目录运行 `bash scripts/launch_interactive_world.sh status` 可读取最新实态。

<!-- interactive-world-evidence 34c0a8bba42c2038 -->
