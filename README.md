# DN-Splatter：AutoDL / Blackwell 复现

目标是在 **RTX PRO 6000 96GB、120GB RAM、25 核 CPU** 的现有 AutoDL 实例上，完成：

**环境盘点与隔离 → Blackwell 适配 → MuSHRoom VR Room 30,000 次训练 → RGB/深度评估与渲染 → Poisson 网格及几何评估。**

**资源策略已更新：用户现在专心进行本项目，授权使用全部已分配 CPU 和整张 GPU。** 标称 25 核 CPU、96GB 显存，以 AutoDL 容器实际配额为准；不再执行原来的 4–6 线程上限或其他任务优先排队策略。Conda、源码、缓存、数据和输出继续独立管理；已有其他项目进程不由本项目自动终止或修改。

## 当前状态

- 2026-11-18：建立复现计划、执行约束、协议配置和只读环境检查脚本。
- 2026-11-18：依据用户最新授权，将执行约定、配置和 Codex 提示词同步改为使用实例全部 CPU/GPU 资源。
- 已完成论文与官方源码静态审阅；**尚未在目标 AutoDL 实例检查环境、安装依赖、训练或获得实验结果**。
- 目标不是仅安装成功，而是尽可能在一个持续执行的任务中完成上述完整流程。
- 原论文历史实现与论文公式存在待核实差异，首轮结果应标记为“固定历史官方实现的单场景基线”，不能提前称完整论文复现成功。

## 从这里开始

将 [完整 Codex 启动提示词](prompts/autodl_first_run.md) 交给运行在目标 AutoDL 实例、或已连接该实例的 Codex。

读取顺序：

1. [AGENTS.md](AGENTS.md)：本项目执行约束。
2. [总执行计划](docs/00_execution_plan.md)：阶段、完成条件、持续执行方式。
3. [环境检查与隔离](docs/01_autodl_isolation.md)：现有环境复用、项目独立前缀、学术加速。
4. [VR Room 执行手册](docs/02_vr_room_runbook.md)：真实参数、断点续训、评估与网格。
5. [源码与协议审计](docs/03_source_audit.md)：当前 main 与历史候选的差异。
6. [结果交付约定](docs/05_result_contract.md)：哪些证据进 Git，哪些大文件保留并登记。

机器可读协议：[configs/vr_room_protocol.json](configs/vr_room_protocol.json)。

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
| 几何协议 | 官方训练视角可见性裁剪；F-score 阈值 0.05 m |
| 首轮训练并发 | 一个正式训练进程使用整张 GPU；不额外并行启动其他实验 |
| CPU 预算 | 允许全部已分配 CPU，标称 25 核；按实际 quota/affinity 确认，按阶段设置并发 |
| 环境候选 | Python 3.10、PyTorch 2.7.1/cu128、torchvision 0.22.1、CUDA Toolkit 12.8、Nerfstudio 1.1.3、gsplat 1.0.0 源码构建 |

环境组合是尚待目标机器验证的候选，不是已实测通过的锁文件。安装成功后才提交实际环境清单。

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

不向官方仓库推送。首轮不展开全数据集、AGS-Mesh、自有数据或超参数搜索。单场景闭环后再决定下一轮。

## 文献和实现依据

论文：*DN-Splatter: Depth and Normal Priors for Gaussian Splatting and Meshing*，WACV 2025。

[来源与核查入口](docs/sources.md)。仓库不会上传用户提供的出版商 PDF、公开数据集原包或模型权重到普通 Git。
