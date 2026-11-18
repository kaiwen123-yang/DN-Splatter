# 给 AutoDL Codex 的首轮提示词

下面正文可直接复制给运行在目标 AutoDL 实例上的 Codex。

---

你现在负责我的 DN-Splatter 论文复现项目。请实际执行，不要只给计划，也不要停在安装或短测试通过。

仓库：https://github.com/kaiwen123-yang/DN-Splatter.git

目标服务器是我租用的 AutoDL：RTX PRO 6000 96GB 显存、120GB 内存、25 核 CPU。它同时在运行其他项目的测试，其他项目持续运行优先。先确认你确实位于这个目标实例；若只有本地或 ChatGPT 容器，不能把它当成 AutoDL，也不要猜 SSH 信息。

【本轮目标】

在一个可持续、可恢复的工作流里，尽可能完成：
只读环境检查 → 新建独立 Conda → Blackwell 适配 → MuSHRoom VR Room 单场景累计 30,000 次训练更新 → 留出 RGB/深度评估与 RGB/深度/法线渲染 → Poisson 网格 → 可见性裁剪后的网格指标 → 推送代码、配置、结果和结论。

【先读取】

检查当前工作区与远端，保留已有修改。阅读仓库 README.md、AGENTS.md、docs/00_execution_plan.md、docs/01_autodl_isolation.md、docs/02_vr_room_runbook.md、docs/03_source_audit.md、docs/05_result_contract.md 和 configs/vr_room_protocol.json。若已有运行记录，先恢复进度，不从头重复安装或启动第二个相同训练。

【GitHub 优先 AutoDL 官方学术加速】

GitHub 的 clone/fetch/push 和 GitHub 源码依赖下载，优先在本项目子 shell 中 source /etc/network_turbo。保留官方 GitHub HTTPS 地址，不修改 .bashrc、全局 Git 代理或其他项目环境。加速脚本不可用时记录后尝试原官方域名直连。不要使用未知第三方镜像处理带凭据的 push。Zenodo、PyPI、Conda 按实际域名单独处理，不把代理无条件传播给全部任务。

【环境检查与隔离】

1. 先运行 scripts/inspect_host.py 并做必要的只读补充，了解正在运行的 GPU/CPU 任务、显存/RAM、磁盘、驱动、nvcc、Conda 与可复用资源。不要输出任何令牌、完整环境变量、SSH 信息或其他项目完整命令行。
2. 在本项目数据盘目录创建专属 Conda prefix .local/conda/dn-splatter-bw。复用适合的现成 CUDA Toolkit、下载包、权重或只读数据；不要向其他环境安装包，不用 base/global pip，不改驱动、系统 CUDA 链接或共享配置，不删除和清理其他项目。
3. 独立设置 Conda/pip/HF/Torch/XDG/编译缓存与临时目录，所有变量仅作用于本项目进程。先用 4–6 个 CPU 线程，本项目最多一个正式训练进程。Poisson 也要限制线程。
4. Conda 隔离不能消除共享 GPU 的性能竞争。其他 GPU 任务活跃且没有明确资源分配时，继续低负载 CPU 工作并为本项目 GPU 阶段持久排队；不要仅凭有空闲显存就启动抢算力的训练。不能结束、暂停或修改其他项目。

【实现和环境起点】

官方源码在本项目 .local/upstream/dn-splatter，固定 811f5cdf342d92d12a9d9a4be185d374c430c184。

优先尝试 Python3.10、PyTorch2.7.1+cu128、torchvision0.22.1、CUDA Toolkit/nvcc12.8、Nerfstudio1.1.3、gsplat1.0.0 源码构建、TORCH_CUDA_ARCH_LIST=12.0。实际 installed versions 由你核查和保存；这些目前是候选，不是已实测环境。

验证 RGB/depth 和法线 legacy 两条 gsplat 路径的实际前向与反向。仅为具体编译或运行错误做最小兼容修复，保存可重放 patch 和原因；不要直接换新版 gsplat、AGS-Mesh 或改变损失来绕过错误。

历史版使用 sensor-depth-lambda、normal-format opengl、Omnidata mono 法线监督。与当前 main 参数不同。历史版尺度项与论文公式仍有差异，首轮建立固定历史官方实现基线，不能提前叫论文完全复现。

【正式执行】

按 runbook 准备 vr_room/iPhone、Omnidata、参考几何及必要参考深度，检查官方 long/short 轨迹和 train/test 划分、RGB/深度/法线对应、单位和初始点云来源。只做直接决定正式运行的必要检查，随后进入完整 30k，不堆 smoke tests。

每 5,000 步保存 checkpoint。Nerfstudio1.1.3 恢复时 max_num_iterations 是本次追加量，必须从实际 checkpoint step 计算 remaining=30000-(saved_step+1)，目标零基最后 step=29999，保留原调度与数据配置。不要重新追加30k，也不要用 --load-config 覆盖掉所需参数。

使用项目专属 tmux 会话或等效持久运行方式。不给训练、排队或网格设置人为完成期限，不自动关闭实例。保存进程、阶段、日志、退出码和恢复命令。常规实施选择自行完成，不要完成每一步就等我确认。

训练后继续完成 ns-eval、代表视角 RGB/depth/normal 输出、gs-mesh dn（目标500万采样点、Poisson depth9）与官方可见性裁剪评估（F-score阈值5cm）。检查实际网格和渲染，不能只看退出码。保留官方 raw 指标；Faro 分轨迹和batch聚合问题按审计文档另存 audited 指标，不静默替换结果。

【交付与 GitHub】

按 docs/05_result_contract.md 保存精确源码/环境/补丁、命令、配置、随机种子、帧列表和来源、初始化/最终高斯数、累计更新数、耗时、峰值资源、metrics JSON、代表图、网格与checkpoint清单。没有真运行的数据保持 null，不填预期值。

优先将小文件、代码和配置、结论推送到 kaiwen123-yang/DN-Splatter，推送前检查 diff，禁止上传凭据、整套环境、数据集或巨大checkpoint到普通Git。大产物在本项目中完整保留并登记路径/大小/sha256；有现成且已配置的大文件目的地时再后台上传，未上传就如实标记。不向官方仓库push，不force push，不做Windows本机归档，不删除服务器数据。

最终告诉我：环境与现有项目是否隔离；是否真正完成30k+评估+Poisson；与论文VR Room表4（完整配置PSNR26.8、SSIM0.879、F-score87.1%）差多少及协议是否可比；哪些是源码/评估差异；GitHub提交与结果位置；如有真实阻塞，给准确原因、已完成项、当前队列/进程状态和下一条恢复操作。不要把任务启动、环境安装或单次出图写成复现成功。
