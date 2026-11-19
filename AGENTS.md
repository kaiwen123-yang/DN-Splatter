# 本项目的执行约定

## 任务和授权

用户已授权在其 AutoDL 实例上复现 DN-Splatter，创建本项目独立 Conda 环境，复用合适的既有资源，并将本项目计划、代码改动、配置、指标和结论推送到 `kaiwen123-yang/DN-Splatter`。目标机器标称 RTX PRO 6000 96GB、120GB RAM、25 核 CPU；以只读盘点得到的实际可用资源为准。

首轮目标：**Blackwell 适配 + VR Room 单场景累计 30,000 次更新 + 评估/渲染 + Poisson 网格和几何指标**。不要停在“环境装好”或短运行通过；满足实际运行条件后应继续到最终产物。

**最新资源授权（2026-11-18）：用户现在专心进行本项目，允许使用 AutoDL 实例分配的全部 CPU 和整张 GPU。此授权覆盖此前“其他任务优先、CPU 仅 4–6 线程、GPU 繁忙即排队”的策略。** CPU 预算以容器实际配额和 affinity 为准，标称 25 核；GPU 可使用整卡算力和可用的 96GB 显存，不人为设置共享份额。软件环境和文件仍按下文隔离。

## 开始前读取

**当前阶段（2026-11-19）：VR Room 首轮完成；本地 bag 审计也已完成并回传 `a13bb40445c1c049c2931094e533a5ed43472297`。** 已确认实际原件为 `H:\capture.bag\capture.bag`，上层是目录；设备为 Gemini 335。请先读 `reports/capture-bag-review/2026-11-19/overview.md`、`docs/08_capture_geometry_plan.md`、`configs/capture_geometry_protocol.json`，按所在位置执行 `prompts/local_capture_geometry.md` 或 `prompts/autodl_capture_geometry.md`。

下一阶段是指定首段提取、同帧 D2C、相机轨迹和米制几何验证、条件满足时准备真实数据集。允许按新协议执行必要的 COLMAP/SfM 和有限失败备选；不要继续用上一轮“只审计、禁止SfM”的结束条件阻塞当前数据准备。四组公开样例的同帧投影已在本次复核执行，首段提取/传输/位姿求解仍是计划。新任务到几何报告回传为止，不自动运行法线推理、DN训练或Poisson。

原包只读留在 Windows；本地依现有SQLite索引提取约5Hz的选定原值帧，AutoDL只接收紧凑派生包，不能凭Windows路径访问原件。保留原始审计与VR Room结果。原第二轮 A/B 基准计划仍在 `docs/04_round2_plan.md`，本次不新增它们，也不终止已在执行的服务器任务。

首轮 docs/00、01、02、03、05 和 `configs/vr_room_protocol.json` 继续作为实现/协议依据；已有环境与 VR Room checkpoint 不从头重做。先检查本项目工作区和远端，保留用户已有未提交修改。计划与已执行记录分开，第二轮未运行的指标不得填入完成状态。

所有后续场景和审计使用独立 run 目录及身份清单，不用旧 checkpoint、指标文件或完成标记误跳新场景阶段。保留第一轮原始 JSON/PLY；规范化审计指标另存，不静默修改历史实验结论。

## 隔离边界

以下 GPU、CUDA、25 核和 Conda 训练环境约定适用于 AutoDL。**本地 bag 审计另行盘点本机 CPU/RAM、磁盘和 Python，优先复用兼容读取工具；缺依赖时建立本项目专属轻量 venv 或 Conda 环境。** 不向 base、系统 Python 或其他项目装包，不要求安装完整 ROS、Docker、WSL 或复制 Blackwell/CUDA 环境。原始文件只读；小型报告写仓库，完整时间表和临时输出放项目 `.local/`。

1. 先只读盘点 GPU 进程、利用率、显存、CPU/RAM、磁盘、Conda 前缀和现成工具链。不要启动全 GPU 压测来“检查环境”。
2. 不结束、暂停、重启或修改其他项目进程，不关闭实例，不重启容器，不设置自动关机或任务完成时限。
3. 不在 base 或其他项目环境里 install/uninstall/upgrade；不使用全局 `pip`、`sudo pip`、`conda update --all`。
4. 不更换驱动，不修改 `/usr/local/cuda` 等共享链接，不修改系统 CUDA、全局包、全局 Git/Conda/pip 配置、shell 启动文件。
5. 新建独立 Conda prefix：`$DN_PROJECT_ROOT/.local/conda/dn-splatter-bw`。已经存在时核对归属与状态后续用，不能无判断覆盖或删除。
6. 合适的现成 CUDA Toolkit、只读数据/权重、wheel 下载可以复用。包版本和 ABI 必须核对；不要把其他环境的 site-packages、已编译 gsplat 扩展或整套环境通过软链接当成本项目环境。
7. 为本项目设置独立 Conda 包缓存、pip/HF/Torch/XDG/编译缓存和 TMPDIR。变更仅作用于本项目 shell / tmux 进程。不得重定义 HOME、CODEX_HOME。
8. 本项目首轮只运行一个正式训练进程，允许其使用整张目标 GPU。编译、预处理、评估和 Poisson 可使用全部已分配 CPU；显式记录 `DN_CPU_BUDGET` 和各阶段 worker/native 线程数，避免多进程与内部线程相乘。编译并发按实际内存需求调整，Poisson 显式设置 `n_threads=DN_CPU_BUDGET`。不为了占满资源新增实验或改变算法参数。
9. 不再要求其他项目优先，也不因看到 GPU 利用率或其他活跃进程就自动排队。检查设备健康、实际剩余显存/RAM/磁盘后推进；已有任务的存在本身不是阻塞。全资源使用授权不包含终止、暂停或重排其他项目进程。只有实际容量不足或运行失败时才记录具体阻塞并保留恢复入口，不承诺仍在运行的任务没有性能影响。

## GitHub 与网络

用户明确要求 GitHub 优先使用 AutoDL 官方学术加速。AutoDL 上的 GitHub clone/fetch/push、GitHub 源码依赖下载，先在本项目子 shell 中 `source /etc/network_turbo`，保留原始 `https://github.com/...` 地址。参考 https://www.autodl.com/docs/network_turbo/ 。

该加速约定按执行位置适用。本地 Windows/WSL 审计使用本机现有 GitHub 连接和凭据，不假定存在 AutoDL 的 `/etc/network_turbo`，不复制 AutoDL 代理或修改全局代理。

不得向 .bashrc、全局 gitconfig 或其他项目写代理。加速脚本缺失/不可用时记录，再尝试官方域名直连，保留可重试工作；不要臆造加速端口、证书设置或第三方镜像。禁止关闭 TLS 校验。Zenodo/PyPI/Conda 未被官方内置服务列为支持域，独立处理这些下载，避免错误代理影响正常网络。

不要输出代理密码、令牌、完整环境变量、SSH 信息或带身份头的 HTTP 调试日志。优先使用实例现有的 GitHub 凭据，不复制凭据到代码或报告。

## 固定源码与方法

官方副本放 `.local/upstream/dn-splatter`，固定提交 `811f5cdf342d92d12a9d9a4be185d374c430c184`。它是经审计选出的历史候选，不是作者认证的唯一论文版本。原始工作副本只服务本项目。

- 首轮模型为 dn-splatter，数据为 vr_room/iPhone，预训练 Omnidata，历史版 `sensor-depth-lambda` / `normal-format opengl`。
- 当前 main 的权重/开关差异、历史版尺度项差异见 docs/03；不得把 AGS-Mesh、新版 API 或最新 README 的参数直接混进本轮。
- 先尝试在支持 sm_120 的工具链上编译 gsplat 1.0.0 的正常和 legacy 后端。RGB/depth 与 normal 两条渲染路径都要验证梯度。
- 仅为实际编译/运行错误做最小兼容修复，保存补丁、原因、前后行为与版本。新增可复用代码放 scripts/ 等跟踪目录；不能只留在被忽略的 .local/。
- 不静默改变训练损失、深度尺度、点数、图像分辨率、数据划分、迭代数、网格方法或评估阈值来得到更好结果。若必要变更，先记录偏差并保留原始协议，不伪装成论文原样结果。
- 首轮不额外启动整套消融、Splatfacto、多场景或参数搜索。

## 持续执行与恢复

任务持续到结果交付或真实阻塞。使用本项目命名的 tmux 会话或等效持久进程，保存日志、PID、退出码和阶段状态。不要给训练/网格任务套固定完成 deadline、`timeout`、自动关机或 watchdog 强杀。

允许对只读环境命令和失效连接设置短命令超时；这不等于给完整实验设置时限。只有实测资源不足等真实阻塞才进入资源等待，不设置“其他 GPU 进程全部退出”或“GPU 必须空闲”的固定门槛。确需等待时可跨多次上下文或会话，留下具体原因、状态、下一步和恢复命令，避免重复启动。

Nerfstudio 1.1.3 的恢复训练会把 max_num_iterations 作为本次追加步数。累计目标是 30,000，零基最后 step=29,999；从 checkpoint 读取实际 saved_step，使用 `remaining=30000-(saved_step+1)`，保留原 30,000 步调度和原数据配置。不要从中间 checkpoint 再盲目加 30,000。不要用未经核实的 `--load-config` 覆盖规则丢失参数。

正式训练阶段每5,000步保存checkpoint，保留必要中间点；必要检查充分后进入该阶段完整训练，不堆叠无关smoke tests。**当前docs/08阶段只做首段几何准备，以上训练恢复规则不构成提前开启训练的入口。**

## 证据与推送

严格区分：计划已提交 / 环境可用 / 单场景流程完成 / 指标接近原文 / 全文复现。训练 loss、漂亮渲染图和程序返回码都不能单独证明论文指标对齐。

必须交付 docs/05 指定的配置、来源、实际计数、指标、图片、网格和产物清单。Faro 评估存在分轨迹聚合问题，保留官方原始结果，审计结果另存并解释定义，不覆盖原始输出。

只向本用户复现仓库推送，不向 maturk/dn-splatter 推送，不 force push。推送前检查 staged diff，排除凭据、原始进程参数、数据集、环境、模型大文件。大产物先本项目保留并做清单与哈希；普通 Git 先提交小文件和结论。有用户已配置的本项目大文件存储时再后台上传，不为此配置额外账号或花费。

默认不做 Windows 硬盘归档、不删除服务器数据、不清理其他项目缓存。失败也应推送已完成的工作、实际错误、精简日志和可继续执行的命令。只有真实权限/访问/资源阻塞才明确报告，不因常规实施选择重复请求许可。
