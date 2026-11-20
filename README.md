# DN-Splatter：AutoDL / Blackwell 复现

目标是在 **RTX PRO 6000 96GB、120GB RAM、25 核 CPU** 的现有 AutoDL 实例上，完成：

**环境盘点与隔离 → Blackwell 适配 → MuSHRoom VR Room 30,000 次训练 → RGB/深度评估与渲染 → Poisson 网格及几何评估。**

**资源策略已更新：用户现在专心进行本项目，授权使用全部已分配 CPU 和整张 GPU。** 标称 25 核 CPU、96GB 显存，以 AutoDL 容器实际配额为准；不再执行原来的 4–6 线程上限或其他任务优先排队策略。Conda、源码、缓存、数据和输出继续独立管理；已有其他项目进程不由本项目自动终止或修改。

## 当前状态

- 2026-11-18：建立复现计划、执行约束、协议配置和只读环境检查脚本。
- 2026-11-18：依据用户最新授权，将执行约定、配置和 Codex 提示词同步改为使用实例全部 CPU/GPU 资源。
- 目标 AutoDL 已完成盘点，独立 Python 3.10 环境及 VR Room iPhone/Faro 几何包已就位。最新实测阶段、checkpoint 和指标见 [运行状态](reports/latest_status.json)。下文列出的协议数值为目标值。
- 2026-11-19：VR Room 首轮完整执行结束，结果提交 `c747cc9`。累计 30k；同轨迹 PSNR 26.9720 / SSIM 0.88108；官方网格 F-score@5cm 86.8629%、C-L1 2.5517cm、NC 89.6993%。主要指标接近论文 Table 4，当前结论限于单场景历史实现。
- 已复核六张图像拼图、两张网格预览及运行/评价脚本，新增 [结果审查](reports/review_2026-11-19.md)、[第二轮计划](docs/04_round2_plan.md) 和 [自有数据接入说明](docs/06_custom_data.md)。仓库尚无第二场景完成记录。
- 2026-11-19：本地 Gemini 335 bag 审计已回传 `a13bb40`。实际原件 `H:\capture.bag\capture.bag`，5,919,666,852字节；RGB/毫米Z深度/双相机标定与时间关联可用，相机位姿缺失，原生深度未注册至RGB。见 [原始审计](reports/capture-bag-audit/20261119T091841Z-3982cb26/overview.md)。
- 已对四组回传原数值样例执行同帧D2C并核对SDK/DN接口，见 [数据复核](reports/capture-bag-review/2026-11-19/overview.md)。
- 2026-11-19：首段几何回传 `d72f568`：241对唯一RGB-D，193张训练图单一地图，48张留出图固定地图定位，统一尺度1.46499035 m/SfM单位，436361点训练初始化；历史读取器逐帧通过。见 [首段几何结果](reports/capture-geometry/20261119-s01-5hz-3982cb26/overview.md)。
- 2026-11-19：capture-s01已完整完成，结果提交 `ee3e87b`。241帧、193/24/24；累计30k、step29999、440882高斯；Test PSNR23.3502、SSIM0.77934、LPIPS0.30406，同采集深度等权帧MAE0.19838m。原始Poisson已交付；桌沿/椅腿细节和末段238/239仍有明显失败。见[完整结果](reports/capture-s01-dn-30k/20261119-v1/overview.md)与[效果复核](reports/capture-s01-review/2026-11-19/overview.md)。
- 2026-11-20：capture-full已完整结束，结果提交 `e26ae72`。1846主帧零缺口，1478 train /184 val /184 test；30k、step29999、172425高斯；Test PSNR15.8256、SSIM0.63841、LPIPS0.43844，合并参考像素深度MAE约0.492m；原始Poisson和全程视频完成。见[完整结果](reports/capture-full-dn-30k/20261119-v1/overview.md)。
- 2026-11-20：[全程效果复核](reports/capture-full-review/2026-11-20/overview.md)发现旧同源48张留出中47张PSNR下降，旧239深度反而改善；训练容量/调度和原始网格额外遮挡面需分别解释。
- 2026-11-20 已启动 **Metric3D v2 ViT-Large** 完整法线替换对照。B1 沿用 B0 的几何、传感器深度和 30k 训练配置，两组按相同规则评价和导出网格。最近一次核验的进度与结果见[本轮状态](reports/capture-full-metric3d-30k/latest.json)。
- 目标不是仅安装成功，而是尽可能在一个持续执行的任务中完成上述完整流程。
- 原论文历史实现与论文公式存在待核实差异，首轮结果应标记为“固定历史官方实现的单场景基线”，不能提前称完整论文复现成功。

## 从这里开始

**当前执行 [本地Codex Metric3D协调提示词](prompts/local_capture_metric3d.md)，或在AutoDL直接执行[服务器提示词](prompts/autodl_capture_metric3d.md)。** 所需全程数据已在AutoDL，不重新提取/上传bag或重做位姿。先读[本次复核](reports/capture-full-review/2026-11-20/overview.md)、[docs/11计划](docs/11_capture_metric3d_plan.md)、[机器协议](configs/capture_metric3d_protocol.json)和[法线接入约定](docs/12_metric3d_normal_adapter.md)。

当前主线：固定ViT-Large全量先验 → 从相同初始化完整30k，仅替换法线 → 全部val/test和B0/B1逐帧/分时段比较 → 全程视频 → B1原始Poisson → B0/B1同参数[支持约束网格](docs/13_capture_supported_mesh.md) → 报告推送。Metric3D深度只作诊断，训练仍使用原米制传感器Z。

复用已有Blackwell训练环境；推理依赖需要时单独隔离，不照装旧Torch requirements。全部CPU/GPU授权继续，按实测主存顺序安排阶段。B0、s01、VR的原始输入和结果保留。本轮不同时修训练采样、延长240k、加深度补洞或扫参数，无新增小训练和阶段审阅门槛。

[首段提取](prompts/local_capture_geometry.md)、[首段几何](prompts/autodl_capture_geometry.md)、[s01训练](prompts/autodl_capture_s01_train.md)、[全程初测](prompts/autodl_capture_full_test.md)及docs/08/09/10保留为历史重放依据；其停止点不适用于当前任务。[基准第二轮](prompts/autodl_round2.md)仍保留，不自动并行启动。

[首轮启动提示词](prompts/autodl_first_run.md) 作为历史执行入口保留；无需重复安装或重新训练已经完成的 VR Room。

读取顺序：

1. [AGENTS.md](AGENTS.md)：本项目执行约束。
2. [总执行计划](docs/00_execution_plan.md)：阶段、完成条件、持续执行方式。
3. [环境检查与隔离](docs/01_autodl_isolation.md)：现有环境复用、项目独立前缀、学术加速。
4. [VR Room 执行手册](docs/02_vr_room_runbook.md)：真实参数、断点续训、评估与网格。
5. [源码与协议审计](docs/03_source_audit.md)：当前 main 与历史候选的差异。
6. [结果交付约定](docs/05_result_contract.md)：哪些证据进 Git，哪些大文件保留并登记。

机器可读计划：当前 [Metric3D完整对照协议](configs/capture_metric3d_protocol.json)；已执行的 [全程初测协议](configs/capture_full_test_protocol.json)、 [s01训练协议](configs/capture_s01_train_protocol.json)、[首段几何协议](configs/capture_geometry_protocol.json)和[bag审计协议](configs/capture_bag_audit.json)保留原计划快照；[基准第二轮协议](configs/round2_protocol.json)保留待后续安排。[首轮协议](configs/vr_room_protocol.json) 保留原计划字段，VR实际完成情况以 `reports/vr-room-iphone-30k/` 和 `reports/latest_status.json` 为准，不能作为capture的完成标记。

## GitHub：优先 AutoDL 官方学术加速

在 **AutoDL 的本项目子 shell** 中启用官方内置服务，保留原始 GitHub HTTPS URL。例如首次获取仓库：

```bash
(
  source /etc/network_turbo
  git clone https://github.com/kaiwen123-yang/DN-Splatter.git /root/autodl-tmp/DN-Splatter
)
```

如果目标目录已经存在，先检查其远端与工作区，不能覆盖或强制重置。`/root/autodl-tmp/DN-Splatter` 是建议路径，实际以数据盘和已有目录盘点结果为准。

后续 GitHub clone/fetch/push、从 GitHub 获取依赖，也优先使用这个方式。不要写全局代理或修改其他项目配置。内置服务不可用时记录原因，再尝试原官方域名直连；不能把带身份凭据的 push 改投第三方镜像。Zenodo、PyPI、Conda 下载按域单独处理，不假定该服务支持所有网址。

本地 Windows/WSL 的 bag 审计使用本机现有 GitHub 连接；上述 AutoDL 命令不要求在本机执行。

依据：[AutoDL 官方学术资源加速文档](https://www.autodl.com/docs/network_turbo/)。

## 固定的起点

| 项目 | 首轮约定 |
| --- | --- |
| 用户复现仓库 | `kaiwen123-yang/DN-Splatter` |
| 官方源码 | `maturk/dn-splatter` |
| 历史候选提交 | `811f5cdf342d92d12a9d9a4be185d374c430c184` |
| 模型 / 数据 | `dn-splatter` / MuSHRoom `vr_room`，iPhone |
| 法线 | 预训练 Omnidata；历史 parser 参数为 `opengl` |
| 总训练更新数 | 30,000；零基 checkpoint 目标 step 为 29,999 |
| 网格 | `gs-mesh dn`；5,000,000 目标采样点；Poisson depth 9 |
| 几何协议 | 固定官方裁剪脚本，F-score 阈值 0.05 m；审计确认实际使用全部 long_capture 注册视角，严格 training-only 另存诊断 |
| 首轮训练并发 | 一个正式训练进程使用整张 GPU；不额外并行启动其他实验 |
| CPU 预算 | 允许全部已分配 CPU，标称 25 核；按实际 quota/affinity 确认，按阶段设置并发 |
| 首轮实测环境 | Python 3.10.22、PyTorch 2.7.1/cu128、torchvision 0.22.1、CUDA Toolkit 12.8、Nerfstudio 1.1.3、gsplat 1.0.0 源码构建 |

这组环境已用于目标机器的正式 30k、评价和网格导出；实际包清单见 [environment_actual.txt](reports/environment_actual.txt)，编译信息见 [compiled_extensions.json](reports/vr-room-iphone-30k/compiled_extensions.json)。独立梯度探针没有运行，正式运行证据与该探针状态分别记录。

## 项目目录约定

| 路径 | 用途 | 进入普通 Git |
| --- | --- | --- |
| `docs/`、`prompts/`、`configs/` | 计划、提示词、协议 | 是 |
| `scripts/` | 项目工具与最终可复用执行脚本 | 是 |
| `patches/` | 对固定官方源码的可重放最小补丁 | 是 |
| `reports/` | 脱敏状态、指标、结论、产物清单 | 是 |
| `.local/conda/dn-splatter-bw/` | 专属 Conda 前缀 | 否 |
| `.local/upstream/dn-splatter/` | 固定提交的官方工作副本 | 否；提交 SHA 与补丁 |
| `.local/data/`、`.local/cache/`、`.local/tmp/` | 数据、独立缓存、临时文件 | 否 |
| `.local/runs/<run-id>/` | checkpoint、完整日志、渲染、网格 | 否；提交指标和清单 |
| `.local/reports/` | 原始环境盘点及私有诊断 | 否；公开摘要需脱敏 |

不向官方仓库推送。VR Room和s01结果保持原样；当前自采全程任务由docs/10和configs/capture_full_test_protocol.json规定。docs/04中的基准第二轮留待后续，不自动增加AGS-Mesh或超参数搜索。

## 文献和实现依据

论文：*DN-Splatter: Depth and Normal Priors for Gaussian Splatting and Meshing*，WACV 2025。

[来源与核查入口](docs/sources.md)。仓库不会上传用户提供的出版商 PDF、公开数据集原包或模型权重到普通 Git。
