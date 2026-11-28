# Codex 完整执行提示词：LingBot-Depth + LingBot-Map 2 大厅离线实验

适用仓库：`https://github.com/kaiwen123-yang/DN-Splatter.git`。
日期：2026-11-28（Asia/Shanghai）。本文件是**待执行任务**；文档提交不代表实验已经运行。

---

你现在负责我的 DN-Splatter 项目中的 LingBot-Depth 与 LingBot-Map 2 离线测试。请按下面的要求完整执行：读取真实仓库与服务器数据，完成必要适配、推理、评测、可视化和 GitHub 交付。推进到实测结果与结论，不停在重新写计划、安装依赖或下载权重。

## 一、先读取当前任务和已完成证据

确认远端是 `kaiwen123-yang/DN-Splatter`，读取最新 `AGENTS.md`，然后完整读取：

1. `docs/16_lingbot_depth_map_plan.md`
2. `configs/lingbot_depth_map_protocol.json`
3. `reports/capture-full-quality-v3/20261122-v1/overview.md`
4. `reports/capture-full-quality-v3/20261122-v1/B3_完整视频与指标对比.md`
5. `reports/capture-full-quality-v3/20261122-v1/selection_freeze.json` 和 `artifacts.json`
6. `configs/capture_quality_protocol.json`、`configs/capture_surface_quality_protocol.json`、`configs/capture_full_test_protocol.json`
7. `reports/capture-bag-review/2026-11-19/overview.md`、`scripts/capture_rectify.py`、`scripts/capture_quality_paths.py`、`scripts/capture_surface_submaps.py`

B2/B3 已经完成，当前冻结选择是 B2 的 210k 渲染模型和 B3 的 210k supported Poisson 网格。不要被 11 月 23 日的历史接续文字或 VR Room 的根 latest_status 带回旧任务。本轮单独建立 `lingbot-depth-map-20261128-v1`，保留所有旧输入与结果。

## 二、我的目标和本轮范围

我要实际看懂并测试两个模型：

- **Depth**：用大厅已有 RGB + 对齐米制传感器深度，检查补洞、原始测量保留、桌沿/椅腿等细结构，生成前后对照和数值证据。
- **Map**：只用按时间顺序的 RGB，恢复相机轨迹与场景点云，检查多视角一致性、尺度变化和困难片段。

两项独立开展，可以并行。VLA、VA 暂时不做；不需要动机械臂、不要求重新采集数据。腕部 DC1 留作后续相机迁移验证，本轮先把 Gemini 335 大厅数据做完整。

本轮使用预训练权重，不训练 LingBot，也不自动重训 DN-Splatter。交付包含可复用代码、完整离线结果、图/视频、失败分析和是否值得接入 DN 的结论。后续训练候选只做接口导出和单因素方案，不能覆盖 B2/B3 输入或直接把新深度/位姿换进正式训练。

## 三、执行位置、资源和持续运行

使用我已有的 AutoDL RTX PRO 6000 96GB、25 vCPU、120GB RAM，按实际可用资源推进。现有项目路径可能为 `/root/autodl-tmp/DN-Splatter`，从真实仓库和清单确认。若 Codex 在本地，通过已有授权连接协调服务器，不把重推理转到本地，也不要要求我重复搬数据或提交已经有的凭据。

1. 先检查现有进程、项目工作区、输入位置和资源，一次完成必要盘点。
2. Depth、Map 使用两个独立环境、日志、输出和 tmux 会话；可并行准备与推理。一个分支出现问题时，另一个继续。
3. 可以使用全部已分配 CPU/GPU。按实际显存/RAM安排 GPU 重任务，避免 CPU 进程与内部线程数量相乘；不要为了并行改分辨率或删帧。
4. 不设置整轮 deadline、`timeout` 强杀、自动关机或任务完成时间限制。保存运行身份、阶段状态、PID、退出码与恢复命令，跨会话接续同一任务。
5. 不结束其他项目进程，不修改系统/共享 CUDA，不在 base 或其他项目环境安装包。使用本项目环境和缓存；如用 `env -i`，保留最小必要变量，不重定义 HOME/CODEX_HOME。
6. AutoDL 的 GitHub 操作优先在本项目子 shell 使用 `source /etc/network_turbo` 和原官方 URL；不要修改全局代理或将带凭据访问转向第三方镜像。
7. 真实接口错误做最小修复并继续。结果差也完成规定评价并如实汇报，不以“效果不好”为由删除难帧或编造成功。
8. 不需要每个阶段找我确认。只有确切缺失的权限、数据或资源无法解决时报告具体阻塞，先完成其他不受阻工作。

## 四、固定输入，正确理解单位和相机

复用 B2 inputs 中已有 RGB、米制 NPY、transforms、splits、training_manifest，通过原 geometry_run/transfer 的 selection 和 calibration 恢复时间戳与标定。已有完整输入时不重新提取 bag、不重跑 SfM。

冻结 1846 主帧、1478 train / 184 val / 184 development-test；另外 100 张辅助建图图像不混入单相机序列。不改已有划分和帧身份。

建立公共 input manifest，包含 RGB/深度文件身份、frame ID、时间戳、split、相机、画布、单位、K、像素中心和已有位姿来源。原始数据只读，派生文件写本轮目录。

- 原始毫米传感器数据与已处理的米制 NPY 分清：**米制 NPY 不再除以 1000**。
- 深度是彩色光轴 Z；已有 D2C 图不再配准一次。RGB、D、有效域、内参必须是同一画布。
- Depth 官方反投影用整数像素中心，因此使用 `K_cv`；若从当前 DN 的 `K_colmap_ns` / transforms 获取，先将 cx、cy 各减 0.5，再按 W/H 归一化一次。已是 `K_cv` 时不重复减。
- Map 自动缩放裁剪到处理画布，保存映射，不能直接按原图像素比较输出。深度参考避免跨边界/孔洞双线性混合。
- 保留原值、无效 mask 与来源。主评测距离 0.3–6 m，区间外单列，不能静默截断以改善指标。

正式推理前固定 30 个 train 样例（约 10 普通、10 孔洞、10 边界/细结构）、256 帧 train 短序列和最多 32 个可靠 train 校准帧；记录选择理由和真实时间间隔。样例选择不能看过模型结果再改。

## 五、Depth 线：完整执行

模型：`robbyant/lingbot-depth-pretrain-vitl-14-v0.5`。
上游起点：`Robbyant/lingbot-depth@f3a237e434ae987bc38281476d6cfb5df3e4d739`。

### 1. 环境与入口

使用 `.local/conda/lingbot-depth-bw`。官方旧 torch/xformers 依赖需要针对真实 Blackwell 算子核查，不能降级其他环境。默认 nested-token 深度 masking 仍依赖 xformers；不能用关闭 depth masking 或仅调用 native SDPA 的方式冒充完成兼容。必要补丁保存并说明，记录实际工作的版本组合，不预设未经测试的组合必定可用。

官方例 0 跑通一次后进入自采数据，不另堆冒烟实验。实现米制 NPY 批处理适配器，RGB 为 `[0,1]` 的 RGB float32，三种输入放同一 device，模型显式 `.eval()`，batch size 1 起步，保留官方默认推理语义。

### 2. 正式首批与固定评价

30 帧保存原始/修复深度、有效 mask、彩色图、同视角点云。分三类报告：

- 原有效区：模型相对原值改变了多少，是否发生尺度偏移；称观测一致性/修改幅度，不把传感器当绝对真值。
- 人工遮挡：seed=42，散点隐藏 10%，64×64 连续块累计约 20%；连续块内全部输入深度置为无效，计分集合另与固定可靠参考集合取交集，不能因边缘不计分就留在输入。prepare 时固定边缘检测与 2 像素排除带，分别保存输入/计分 mask；mask 在自定义缩放、补洞、归一化和尺度统计前实施。目标值只进入评测。使用同一被遮挡输入的二维最近邻补洞作固定参照。
- 原生孔洞：补全覆盖、细结构和跨视角支持。没有独立观测支持的区域标未证实，不能因填满就写精度提高。

人工 mask 内报告覆盖率、无效输出、MAE/RMSE/AbsRel/P95、5/10/20 cm 正确恢复覆盖，逐帧与分组汇总。有效预测必须有限且严格为正并满足该方法有效性规则；条件误差在有效交集上计算并列支持数，覆盖和正确恢复比例用全部固定参考像素作分母。不得按预测值范围删除离群点；交集为空时误差 null、覆盖 0，汇总披露失败帧数。模型 mask/inf 不能悄悄缩小覆盖分母；原始缺测 0 不作为“错误预测”衬托模型。跨视角检查须处理遮挡和已有位姿不确定性。

### 3. 全训练集与可视化

接口正确后继续全部 1478 train 的无人工 mask 推理，不停在 30 帧。不因质量负结果中断完整统计；实际格式/数值故障修复后继续。

保存真正的 `.npy` 数值；输出 PNG 仅用于观看。对比图统一色标，输出 RGB/raw/refined 三联、差值图、同视角点云和固定片段视频，包含困难样例。可以生成保留 raw 有效值、仅用模型填孔的独立候选及来源 mask，但不修改现有 DN 深度。

## 六、Map 线：完整执行

模型：`robbyant/lingbot-map-v2-base-streaming`，当前公开的 pure-causal base，无 GCR。
上游起点：`Robbyant/lingbot-map-v2@078b54485830c74ab1d3797667073e3e8353f3b9`。

### 1. 环境、下载与正式入口

使用 `.local/conda/lingbot-map2-bw`，按实际 GPU 建立支持 BF16、torch>=2.8、FlashInfer 的环境。官方下载器取配套权重与 JSON，保持 SHA 校验，不用示例配置替代发布配置。

调用官方 `infer.py` 时**同时显式传 `--checkpoint` 和 `--config`**；省略后者并不保证找到同目录的 config.json。先落盘，再用 `demo.py --predictions` 查看同一结果。

按协议依次运行三个独立 stream：

1. `short_train_256`。
2. `full_train_1478`。
3. `full_all_1846_exploratory`，清楚标明使用了 val/test RGB 的全序列探索。

所有序列按原始时间排序，用 JSON 路径数组明确顺序。不将不同相机、不同时间段随意拼接。短序列缺少的 val/test 时间位置自然保留为间隔，不重标成无缺口采集。

### 2. 官方缓存与帧身份

保留 CLI 默认关键帧策略、8 帧初始化、512×288、64 帧窗口；首轮不加 `--keyframe-interval 1`。所有输入都应产生预测，进入持久记忆的只是所选帧。

前 7 次 API 返回 None，第 8 次一起返回 8 帧。用 manifest 和每个 NPZ 的 `raw_indices` 对齐原始 frame ID；不能用可复用的 `frame_indices` 或 NPZ 文件数代替帧数。核对输入、预测、持久帧数量，明确缺失/重复/失败。

每个 stream 独立状态，同一模型不并发开多个 stream。中断后只保存了 progress/NPZ 不能直接从中间开始假装恢复；无官方状态 checkpoint 时优先使用原始完整 image-list 重跑到新 attempt 目录。若适配器只重放至中断点，仍使用原完整序列的冻结 persistent raw IDs，不根据截短列表重新计算 Uniform320。保留原始配置、顺序、前缀和证据。

### 3. 尺度、坐标、评价和展示

保留原始任意尺度输出。Map 是 OpenCV C2W，深度与相机平移不能直接当米。三个独立序列/attempt 各自使用同一组 P0 冻结可靠 train 校准帧，分别估计自己的单一全局尺度与刚体对齐并冻结；不能把 short 的数值倍率/变换套用到 full stream，也不能在一个序列内逐帧重缩放掩盖漂移。

完整输出包括相机轨迹、内参、深度、原始 confidence、预处理映射、尺度证据。confidence 不是概率，过滤后展示必须同时报告保留覆盖。

检查轨迹跳变、同一桌子/墙面重影、重访区域不一致、固定尺度下的沿程残差，以及真实存在的困难帧。已有 B2/B3 相机不是 GT，与其差异只叫参考轨迹差异，不直接写成真实 ATE/RPE。没有真实重访则该指标标不适用。

保存可旋转的点云查看入口、轨迹图、按时间回放的视频和固定困难区域对照。查看器需要原图路径仍有效；优先服务器启动后端口转发，不再做本机归档。公开 base 没有 GCR，不自行编造这个阶段；点云不称完成的 3DGS 或物理仿真资产。

## 七、数据用途与后续判断

`full_train_1478` 的前缀、缓存、尺度标定与几何输出只使用 train。不能先跑全 1846，再筛出 train 结果称为严格训练先验。全 1846 的探索结果独立保存和披露；其与 train-only 的差异不冒充纯算法消融。

本轮不强行补齐 val/test 的候选位姿，也不假设 Map 自带可用的固定地图定位流程；不通过插值相机填满数量。原开发 test 不重新称盲测。

两条线分别给结论：值得做后续候选、仅局部有帮助、证据不足、当前设置不推荐、或真实阻塞。说明支持证据和反例，不设任意百分数强行验收。后续若接 DN，先分别设计“只换深度”“只换位姿”的对照；本轮不启动该训练。

## 八、代码、日志与 GitHub 交付

按计划实现最小可复用入口，职责涵盖 prepare、Depth、Map、evaluate、report 和持久启动/恢复。不只把代码留在 `.local/`；跟踪脚本、实际依赖和必要补丁进入仓库。不要导入带旧 RUN 写入副作用的脚本而覆盖 B3。

完整输出放 `.local/runs/lingbot-depth-map-20261128-v1/`。小报告放 `reports/lingbot-depth-map/20261128-v1/`，新状态索引为 `reports/lingbot-depth-map/latest.json`，保留旧根 latest_status。

必须提交：

1. 中文结果总报告，Depth/Map 各自的结论、失败和下一步。
2. 实际代码/模型 revision、SHA、环境、硬件、命令、输入清单、split 身份、样例和 mask 清单。
3. 每帧 CSV/JSON、指标定义、单位、固定分母、覆盖、分组与困难区域结果。
4. 可直接看的对照图、轨迹图、视频/三维查看入口。大文件保留服务器，清单写实际路径、大小和 SHA-256。
5. 真实运行状态、数量、峰值显存/RAM、分阶段耗时、失败条目与恢复命令。GPU 计时同步，首次加载/编译不混入稳定推理帧率。
6. 本轮代码和报告推送到 `kaiwen123-yang/DN-Splatter`，读回核验 commit 与文件链接。只 stage 本轮文件，保留并发改动，不 force push。

模型权重、环境、全量 NPY/NPZ、原始数据和大视频不进入普通 Git；默认不做 Windows/本机归档，不删除服务器数据。失败也推送已完成工作与实际错误；未完成项明确标注，不能用 null/0 混淆“未测”与“零误差”。

最终向我用中文报告：**实际跑了多少、我应该先看哪张图/哪个视频、Depth 有没有帮助、Map 有没有帮助、证据局限是什么、下一步是否值得接入 DN。**不要只给“进程已经启动”的回执；持续推进至本轮交付或确切阻塞。
