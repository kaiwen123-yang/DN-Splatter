# Codex 执行提示词：灵波支持的可校准交互世界

请在 kaiwen123-yang/DN-Splatter 仓库执行本轮完整实验。目标是把真实场景逐步变成能接受动作、随交互校准、生成有效训练数据的世界。本轮以 LingBot-Depth / MDM 与 LingBot-Map / Map 2 的工作为主要参照，实际推进算法、完整重建对照和交互验证；VLA、VA、World 大模型训练暂缓。

用户已授权本轮实现、必要独立环境/公开数据下载、新 DN 训练、并行计算，以及将代码、配置、报告与结果推送本仓库。不要停在计划、环境或启动进程；推进到所有可执行工作完成。没有中间人工审批门槛。真实缺数据/访问/容量的分支准确标 blocked，继续其他分支，不把困难分支静默删掉。

## 一、先完整读取当前入口，保护历史结果

按顺序读取：

1. AGENTS.md 的最上方当前任务入口。
2. docs/17_interactive_world_plan.md。
3. configs/interactive_world_protocol.json。
4. reports/interactive-world-review/2026-11-28/README.md 与 literature.md。
5. DAILY_RESEARCH_LOG.md。
6. 上述文件引用的 B3、LingBot 来源/配置/输入/执行回执及必要源码。

旧 docs/16 的“仅离线、不新训 DN”和首轮单训练进程/全串行限制已由用户本轮授权覆盖。旧 B2/B3/VR/LingBot 输入与结果保持只读，不重跑旧阶段来凑完成数。新 run 用 interactive-world-20261128-v1，报告 reports/interactive-world/20261128-v1，状态 reports/interactive-world/latest.json。当前文件中的 status=planned 必须根据真实执行更新，不能只改状态字符串。

审阅指标基线为 ffc157fe30b01796f12e41361c6b2d3ace33521e；94d17e968ec65854d616f2215bca5135b9d5d905 新增完整录像和脚本，必须保留。拉取时核对 remote、现有未提交改动和任务锁；并发新提交应合并，不强制重置或 force push。

## 二、实际盘点与恢复，不假定服务器资产仍存在

先只读核对当前 AutoDL 的 GPU、实际 CPU quota/affinity、RAM、磁盘、项目进程/锁、独立环境和输入清单。上轮新实例只恢复了部分 B2 RGB/Z/相机/划分与两个 LingBot 环境，不能从历史路径推断 DN 环境、B2/B3 checkpoint、原生 depth packet、VR/Faro 都在。

根据既有 manifest/哈希恢复必要资产，保留输入身份。允许读取已存在的旧归档恢复所需文件，不创建新的电脑归档。不重提整个 bag 替代可恢复 packet。native packet 缺失只阻塞依赖 native 的分支；已有 RGB-Z、Map、公共动作和其他几何实验继续。

优先复用既有 Blackwell 适配与已固定官方代码；DN、LingBot、MuJoCo/PhysTwin 各在兼容的独立项目环境运行。不能向 base/系统/其他项目装包，不改驱动、全局 CUDA、全局代理或 shell 配置，不终止其他项目。新 MuJoCo 依赖不强塞 DN 的 Python 3.10。完整录屏已有 PyAV 15.1.0 + NVENC 回执，可复用 scripts/lingbot_record_hall.py；旧查看器已经关闭，不假定 localhost 在线，默认交付离线文件。

## 三、建立六条工作线，并行实施

由一个 coordinator 维护 DAG、run 身份、锁和发布。能力允许时由多个代理/worker 分工；即使代理数量不足，也将六条工作线实现为可并行的进程任务。worker 只写自己的目录，coordinator 统一提交和维护每日日志。

### A：Depth 条件输入、候选与 MDM 表征迁移

- 保存原生量测/投影 footprint/source lineage/遮挡/范围来源；无法唯一判断的孔洞为 unknown。
- 自然缺失由 full-range raw Z 的 finite-positive 定义，不用截带后的 B2 NPY 零值冒充缺测。
- 保留 NN、保边界 sensor footprint、独立其他训练视角三个强基线，记录插值的真实源身份。
- 全 1478 train 构造 native footprint、连续块、几何边界三类隐藏。每 frame/mask 固定一个实例，最多 4434 基础组合；三种输入表示最多 13302 条件输入，实际数量分别登记。缺 native 的任务单列阻塞。
- 在所有候选和特征生成前删除 hidden source 及其全部后代。每个 mask 重新推理，禁止复用该帧未遮挡的 v1 输出。
- 按 10 秒组做 5-fold cross-fitting；outer k 只评分/导出，(k+1)%5 校准，剩余三折 fit。归一化/PCA 只 fit，排除重复源和近邻泄漏。边界计分不复用旧排边 reference；空分母为 null。
- 真正训练 Q0/Q1：Q0 为几何统计，Q1 加冻结 MDM 的局部 features；正确映射 infer_feat/forward_feat，而不是把 cls_token 当逐像素几何。统一两层 64 hidden units、32 维表征槽、30 epochs、AdamW 1e-3。缓存读取或 context 外 clone，避免 inference tensor 直接反传；最多 4096 分层训练像素/frame/mask，完整评分。
- R3 固定用 Q1；Q0 与简单校准基线检验表征收益。报告实际参数量、风险覆盖、分层 Brier/ECE、正确覆盖与拒绝，5% 错误接纳目标不是预先保证。
- 比较全距离稀疏、主距离带外置缺失（不 clamp）、footprint-aware 条件。原测量保持只读；保留互斥深度模式，不能平均前后景制造新表面。

### B：Map 的尺度、匹配、局部位姿和吞吐

- 复用已交付轨迹；全参考 Sim(3) 只解释 gauge，不声称精度提高。B2 不是 GT。
- 冻结 train 内 fit/audit，以独立原始深度/对应验证全局与低自由度尺度。不能逐帧把深度尺度乘绝对相机中心。
- 固定 1304/1305/1514 并覆盖全 train 正常段。新边来自时间/RGB 检索/MASt3R，不由可疑 B2 朝向唯一排除。
- 相同约束/预算下比较 B2 与 Map 初值，真实米制锚点，独立 audit 才可接受 pose。旧 B3 已有 SIFT/PnP/局部优化；新贡献必须来自新证据/初值/约束。
- pose 改动后重生成受影响输入/初始化/法线坐标到新目录。1846 探索含 val/test，不能回流 train 先验。
- 实施 CPU 预解码/resize 有界队列，单消费者严格按 raw_index 推 Map stream，异步保存独立副本。跑完整 1478 改进序列，逐 ID 核对和数值比较，报告端到端和峰值，不只报 kernel。
- MapAnything 可作扩展比较，不成为主线依赖。公开 Map base 不含 GCR，不声称实现外部证据记忆就改写了模型 KV。

### C：完整 DN 与薄结构对照

R1–R5 从相同初始化、seed=42、原 1478/184/184 划分、完整 1280×720 开始，各累计 240k；save 10k、全 val 30k、增长停止 120k，统一 Gaussian cap/增长调度。不能降分辨率、删难帧或以短训练代替。

- R1：B3 + sensor-only 保守缺失监督。
- R2：同候选体系加入冻结 LingBot，独立几何硬规则选择。
- R3：与 R2 相同候选，Q1 out-of-fold 风险选择/拒绝。
- R4：R3 + 有真实几何支持的边界容量重分配，总高斯预算相同。
- R5：R3 仅关闭 normal TV，检验是否主要是过度平滑。
- R0C：历史 B3 与新共同协议不完全一致时，追加同协议 240k 基线。
- R6：出现实际且独立验收通过的 pose 更新时，R3 + pose 完整 240k。
- R7：边界与 pose 各自有效后，组合完整 240k。

五个必做即 120 万次新更新。效果差也完整交付；等价监督必须核验并标注，不能冒充新收益，不能把未实现称等价。R6/R7 按证据自动判定，无需重复请求确认。

新 hole loss 与原 B3 sensor loss 分开归一化，lambda_h=0.1×原 sensor_depth_lambda。R1–R5 接受后 q=1，质量概率只选/拒绝；误差为 log1p(abs(pred-target))，继承 B3 两方向 RGB 因子/归一化，不误写成 log-depth 差。先筛 accepted/finite/positive/range-valid，再算残差，避免 0×inf。零接受集返回可求导零并检查原梯度不变。R4 不能只按纹理边加点，不用空贡献 ID 假接口；记录近似归因/必要 renderer 补丁。模型几何与相同网格出口分别评价。

恢复时读取本 run 的 saved_step，remaining=240000-(saved_step+1)，保留原总调度、优化器、sampler/RNG，防止追加超预算。

### D：对象、碰撞、动作和可修订证据记忆

先交付一个可信支撑面、一个可动刚体、一个 pusher 的局部世界，不等待全厅精细化。没有真实完整对象时可用明确标注 synthetic_proxy 的标准体完成接口，不能算作已恢复真实对象。

采用独立 MuJoCo CPU 物理 + GS 渲染桥梁，视觉和碰撞分层；对象从背景中移除，避免复制残影。保存 object_id、形状/碰撞来源、mass/inertia/friction 的 measured/assumed/identified 状态与版本。

明确 T_physics_reconstruction、支撑平面/重力、米秒kg/rad、wxyz 四元数、相机转换及对象原点；GS 中心/协方差/法线/方向外观同步变化，使用深度正确合成。整厅不能一个凸包。实现 reset/step/observe/replay，240 Hz 物理、30 Hz 控制，teleport 仅 reset。

实现 M0 不可撤销融合、M1 可见性/空闲证据更新、M2 对象身份/版本/回滚。相机只用静态背景锚，移动对象不参与相机拟合。遮挡不等于消失；只有真实看穿旧位置才能删除。外部新状态须被后续融合、碰撞和生成实际读取。每 episode reset 同时恢复物理、对象、ledger、缓存和校准器。

### E：真实交互、校准、灵波归因和数据有效性

公开 PushT 用全部可用 60 episodes，按完整去重组固定 36 fit/12 calibration/12 test。xy 是执行器 waypoint，不是对象位姿/力；该数据没有 depth。配套 T-block 是静态资产，不是额外真实交互数据。

用独立冻结跟踪器和可审计关键帧建立共享对象标签及共同评价 mask。先校准相机/初态/执行器响应，再拟合可辨识物性；不要从单纯滑行同时宣称辨识质量和摩擦。C0 默认、C1 对齐、C2 物性、C3 局部证据更新分别评价。

开环预测与在线更新分表：开环用 fit/calibration 后冻结世界；在线各方法得到相同因果观测。test 状态不进入生成资产或其他 test episodes。合成观测来自独立 teacher，不能让候选世界自渲染、自验证。

另执行 PhysTwin 的真实 RGB-D 归因支线，优先 single_push_rope_1，核验后加入 rope_4。用仅由 train prefix 形成的相同拓扑/物理先验，test 首帧仅 camera0 masked RGB-D，比较 G-NN/G-LingBot/G-Q1。先验可能含未来则重建 prefix 或明确 blocked。绳体保留官方 spring–mass，不放进刚体 MuJoCo。修订一次状态，给定同一记录控制点，开环预测 0.5 秒；camera1/2、未来物体观测与 gt_track_3d 只评价。空有效点为 null，不能记零误差；结论限于已知物体的状态校正和条件预测。

预存 512 条完全一致的初态/动作数组，C0/C2/C3 三个已冻结世界各跑 150 控制步，共 1536 trajectories、230400 控制时刻。生成时没有新真实观测，C3 不在线自校准。全保存状态/动作/失败/接触；视觉按协议固定子集和5Hz预览、分片压缩，先预算磁盘。

训练小型状态+动作对象预测器（不是 VLA/VA）：RealOnly、Real+重采样Real、Real+C0、Real+C2、Real+C3；3 seeds，每种20k updates，两层128、AdamW1e-3。每步同64真实样本，增强/重采样另64辅助，固定权重；同真实曝光、同辅助预算和配对顺序。历史0.5秒，预测0.5/1/2秒足长窗口，加入保持/恒速基线，按真实 calibration 选点、真实 test 评价。不能把执行器轨迹当对象标签；标签不够如实 blocked。状态探针收益不等于视觉策略或真实机器人成功率。

### F：VR/Faro、完整视频、状态与每日记录

恢复 VR，按实际738×994、每帧K、深度单位和376/42/186划分。LingBot已是米，DN按自己的scale还原。VR1 sensor-only与VR2仅加质量选择各30k；历史协议不等价则VR0C，边界组合另列。Faro只评价，不调质量头；保留旧official并另报统一有效mask与标准误差、418历史/376严格train裁剪。

复核完整大厅时段，保留1846时间位置/既有遮挡与困难段，图上标frame/source/version/相同色标。动态视频并列真实、冻结预测、在线校准、残差证据。提交可离线播放的预览与大视频路径/哈希，不只交启动服务链接。坏结果、拒绝和未解决问题都进入总表。

## 四、算力、持续执行和恢复

使用全部已分配25 CPU、96GB显存及实际120GB主存，按实测容量调度。历史单DN RSS约79GiB，初始1重训练+Depth/解码/物理/轻评价并行；共享mmap/按需缓存后以cgroup总量、增量峰值、PSS/USS核准第2/3重训练，避免重复计算共享RSS。不人为设置GPU共享比例，也不盲启六个大进程。

CPU内部线程与worker总预算共同管理；保持GPU下一批数据就绪、已完成结果及时评价。Map单stream严格有序。后台用项目tmux/持久进程；不设墙钟完成时间、不用timeout/watchdog强杀、不自动关机。150仿真步/240k更新是实验预算，不是后台限时。

失败分支记录exit code、真实错误、输入身份、产物和resume命令，修复后继续；不重复启动已有有效进程。上下文切换不应丢任务状态或重跑完成阶段。不能通过缩小数据、降低分辨率、偷换指标解决资源问题。

## 五、每日记录与最终发布

DAILY_RESEARCH_LOG.md 是用户要求的单独长期文件。按Asia/Shanghai每天追加，coordinator汇总各lane事件；写清实际工作、问题和根因、修复与路径、指标/分母、负面结果、资源并发、产物/哈希、Git提交、正在运行的阶段、剩余任务和精确恢复入口。计划/启动/完成/验证分开，不只写“优化了一下”。

所有真正执行代码/补丁/配置进入Git，不能仅留.local。小报告和预览提交，大产物留服务器并登记路径/bytes/SHA-256。AutoDL GitHub优先官方学术加速，仅项目子shell；保留官方URL，不输出凭据，不force push。远端变更先合并，推送后读取远端确认。

**推到GitHub后，不再在用户电脑做归档，不删除服务器数据。** 不新建收费服务，不擅自执行真实机器人动作。需要新实机数据时给具体采集清单，其余可执行工作继续。

最终中文交付必须回答：灵波真实贡献是什么；五个完整DN对照与条件组结果；外部几何、碰撞、动作、校准和数据分别验证到哪层；哪些失败/blocked及原因；实际并行和吞吐；每日文件入口、Git commit和继续命令。不要以“后台已启动，本次对话可以结束”替代完成交付。
