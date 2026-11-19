# AutoDL 环境盘点、复用与项目隔离

适用目标：用户提供的 RTX PRO 6000 96GB、120GB 内存、25 核 CPU。**用户于 2026-11-18 明确授权本项目使用实例全部 CPU 和 GPU 资源**，覆盖此前为同时进行的其他实验保留算力、限制 4–6 线程和等待 GPU 空闲的策略。Conda、源码、缓存和输出仍独立管理。2026-11-19 已完成目标机正式运行，实际环境见 [环境清单](../reports/environment_actual.txt) 与 [结果审查](../reports/review_2026-11-19.md)；后续优先复用，不重复安装。

## 1. 先盘点，再创建环境

在本仓库根目录执行：

```bash
python3 scripts/inspect_host.py
```

报告写入本项目 `.local/reports/host_<UTC时间>/`。脚本仅执行短时间只读查询：GPU 型号、驱动、显存和利用率、GPU 计算进程、无命令行参数的进程列表、CPU 数量、容器内存/CPU 限额、磁盘空间、`nvcc` 位置与版本、Conda 环境列表。Conda 列表直接从注册清单和已知前缀读取，不调用可能整理用户注册清单的 Conda 命令；未注册到这些位置的自定义环境可能被遗漏。它不安装依赖、不激活环境、不导入 PyTorch、不启动 CUDA 计算，也不修改任何进程。

如已从环境列表选出值得复用的环境，可只读检查其中的相关包版本：

```bash
python3 scripts/inspect_host.py \
  --inspect-env /absolute/path/to/existing/conda/env
```

该参数只读取版本，不复制或改变该环境。原始报告含本机路径和进程名称，只保留在已忽略的 `.local/`。推送 GitHub 的摘要应去除用户名、主机名、IP、绝对私人路径、访问令牌和与本任务无关的项目细节。不要导出完整环境变量、shell 历史、`ps` 命令行参数、SSH 文件或凭证配置。

### 盘点后必须回答的问题

1. 当前 GPU 上运行哪些任务？哪些属于其他项目，是否仍在训练或周期性推理？
2. 当前显存、GPU 利用率、可用内存和 CPU 占用怎样？记录它们用于识别容量不足和解释运行耗时；不把 GPU 非空闲本身作为排队条件。
3. AutoDL 容器的 CPU affinity/cpuset、cgroup CPU 配额和内存限额是多少？不要把宿主机 CPU 数或 `/proc/meminfo` 的总量当成容器可用配额。
4. 系统盘与数据盘各剩多少空间？先检查目标目录所在文件系统，避免把环境和数据误写到已满系统盘。
5. 现有驱动是否支持目标 GPU？`nvcc` 来自哪里，是否与计划使用的 PyTorch CUDA 版本匹配？
6. 哪些已下载 wheel、Conda 包归档、模型权重或数据可以只读复用？版本、来源、完整性是否可确认？

GPU 查询不可用时，记录诊断缺口并检查设备可见性、驱动和后续必要的 CUDA 运行结果；只有设备确实不可用等实际错误才阻塞 GPU 阶段。进程列表可见性不完整本身不撤销用户的全部资源授权。脚本内短超时只用于只读探测，不用于训练或整个项目的完成时间限制。

## 2. 项目目录与独立 Conda 前缀

优先使用空间足够的 AutoDL 数据盘。以下目录是推荐约定，根目录应按实际挂载盘确定：

| 内容 | 相对本项目根目录的位置 |
|---|---|
| 专属 Conda 环境 | `.local/conda/dn-splatter-bw/` |
| 固定版本上游源码 | `.local/upstream/dn-splatter/` |
| 本项目运行报告 | `.local/reports/` |
| 下载与运行数据 | `.local/data/` |
| 检查点、渲染和网格 | `.local/runs/` |
| 缓存和临时文件 | `.local/cache/`、`.local/tmp/` |

统一变量为 `DN_PROJECT_ROOT`（仓库实际绝对路径）、`DN_SOURCE_DIR=$DN_PROJECT_ROOT/.local/upstream/dn-splatter`、`DN_DATA_ROOT=$DN_PROJECT_ROOT/.local/data`、`DN_RUN_ROOT=$DN_PROJECT_ROOT/.local/runs/<run-id>`。每一轮使用独立 run-id，不能覆盖其他运行记录。

**不要克隆一个包含其他研究项目的大环境作为起点。** 盘点现有环境是为了复用已下载的兼容资源和了解可用工具链；新项目仍创建独立前缀，并记录解析后的依赖版本。

创建前确认本项目路径、磁盘空间以及无现存同名环境。在仓库根目录中可使用：

```bash
DN_PROJECT_ROOT="$(pwd -P)"
export DN_PROJECT_ROOT
export CONDA_PKGS_DIRS="$DN_PROJECT_ROOT/.local/cache/conda_pkgs"
mkdir -p "$CONDA_PKGS_DIRS" "$DN_PROJECT_ROOT/.local/conda"

conda create --copy \
  --prefix "$DN_PROJECT_ROOT/.local/conda/dn-splatter-bw" \
  python=3.10 pip -y
```

若该前缀已经存在，应读取其历史和包清单后恢复使用，不能直接覆盖或删除。`--copy` 用于减少环境文件与包缓存的硬链接耦合。后续安装通过该前缀的 Python 或 `conda run --prefix ...` 执行；不要使用来源不明确的裸 `pip`。

不改动 Conda base、其他项目环境、系统 Python、系统 CUDA 链接、驱动、`.condarc`、用户 shell 启动文件或全局 Git 配置。不执行全局升级、`sudo pip`、`conda update --all`，也不清理共享 Conda/Pip/Hugging Face 缓存。

## 3. 复用的边界

| 可复用资源 | 使用方式 |
|---|---|
| 已安装的 NVIDIA 驱动 | 只读使用，确认支持该 GPU；不自行升级、重载或重置 |
| 已有 CUDA Toolkit 12.8 | 确认路径和 `nvcc` 后，用项目进程变量引用；不更改全局软链接 |
| 已下载的 wheel / Conda 包归档 | 校验版本及哈希，从只读来源复制到本项目缓存或用只读 wheel 目录安装 |
| Omnidata 等模型权重 | 校验来源/版本/哈希，直接只读引用或复制到本项目目录 |
| MuSHRoom 数据包或已解压数据 | 核对版本和文件清单；训练脚本可能生成法线/点云，衍生文件必须写本项目目录 |

不复用其他项目的可写 `site-packages`、JIT 编译目录、训练输出目录或全局可写模型缓存。不要把已有环境的包目录加入 `PYTHONPATH`，也不要通过可写共享源码目录执行临时修改。

已有数据若必须被上游解析器原地补充法线、点云或 JSON，就先建立本项目工作副本。可按文件系统能力使用独立复制或 reflink；不要用硬链接后改写共享文件。符号链接只用于确定只读的输入，不能把自动生成结果写回共享数据目录。

## 4. 每个进程使用本项目缓存

下面的变量只对本项目进程生效，不写入用户启动文件：

```bash
export PIP_CACHE_DIR="$DN_PROJECT_ROOT/.local/cache/pip"
export HF_HOME="$DN_PROJECT_ROOT/.local/cache/hf"
export TORCH_HOME="$DN_PROJECT_ROOT/.local/cache/torch"
export XDG_CACHE_HOME="$DN_PROJECT_ROOT/.local/cache/xdg"
export TORCH_EXTENSIONS_DIR="$DN_PROJECT_ROOT/.local/cache/cuda_extensions"
export TMPDIR="$DN_PROJECT_ROOT/.local/tmp"
export PYTHONNOUSERSITE=1
export TORCH_CUDA_ARCH_LIST=12.0
mkdir -p "$PIP_CACHE_DIR" "$HF_HOME" "$TORCH_HOME" \
  "$XDG_CACHE_HOME" "$TORCH_EXTENSIONS_DIR" "$TMPDIR"
```

执行器从盘点结果确定 `DN_CPU_BUDGET`，使用实例分配的全部有效 CPU 配额，不再预留 4–6 线程上限。用户报告为 25 核；若 affinity/cpuset 和 cgroup 均允许 25 个 CPU，则在本项目 shell 中设置 `export DN_CPU_BUDGET=25`。若实际配额不同，使用确认后的值并记录依据，无需为常规预算选择再次请求确认。不能仅依赖可能显示宿主 CPU 总数的 `os.cpu_count()`。

以 OpenMP 为主要并行层时，可使用以下配置；它不是要求所有阶段都使用相同线程池：

```bash
: "${DN_CPU_BUDGET:?先依据实例的 affinity/cpuset 和 cgroup 配额设置 DN_CPU_BUDGET}"
export OMP_NUM_THREADS="$DN_CPU_BUDGET"
export OMP_MAX_ACTIVE_LEVELS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MAX_JOBS="$DN_CPU_BUDGET"
```

按实际执行阶段选择一个主要并行层，避免“25 个 worker × 每个 worker 25 个线程”：

| 阶段 | 全部 CPU 配额的使用方式 |
|---|---|
| CUDA/C++ 编译 | `MAX_JOBS` 可达到 `DN_CPU_BUDGET`；编译子进程内的数值线程通常设为 1。根据可用内存和实际峰值调整并发，避免 OOM；调整原因写入报告。 |
| 多进程数据预处理 | worker 总数不超过 `DN_CPU_BUDGET`，各 worker 的 OpenMP/BLAS 线程设为 1；单进程算法则可把主线程池开到全部预算。 |
| 单场景训练与渲染 | 一个正式训练进程使用目标 GPU；CPU 线程按实际数据加载和数值计算路径设置。若以 BLAS 为主要并行层，可将相应 BLAS 线程提高到全部预算，并减少外层并发。 |
| Poisson | 核对 Open3D 接口，在项目副本中显式传入 `n_threads=DN_CPU_BUDGET`，避免仅依赖环境变量。 |

记录每个阶段的实际 worker、OpenMP/BLAS、编译和 Poisson 线程数。授权使用全部资源不意味着必须把显存、内存或 CPU 利用率填满；不为达到 100% 利用率扩大点数、分辨率、批量、并行实验数量或改变论文协议。

清理本项目启动进程继承的 `PYTHONPATH`、`PYTHONHOME` 等跨环境路径，按实际工具链设置其 `PATH`/`CUDA_HOME`/必要库搜索路径。不要把 `HOME` 或 `CODEX_HOME` 改成项目目录。不同 PyTorch/CUDA 组合的编译实验使用不同项目内扩展缓存目录，避免旧二进制被误用。

GPU 工作绑定盘点确认的目标设备，允许本项目使用整张 RTX PRO 6000 的算力和可用显存，不人为设置共享算力或显存比例限制。绑定设备和用户授权不会自动终止旧进程；不要修改其他任务，也不要修改 GPU 的功率、时钟、计算模式、MIG、MPS 或持久化设置。

## 5. GitHub 优先使用 AutoDL 官方学术加速

用户要求 GitHub 优先走 AutoDL 官方学术加速。[AutoDL 官方说明](https://www.autodl.com/docs/network_turbo/)给出的启用方式是 `source /etc/network_turbo`，说明的服务范围包含 GitHub、GitHub 资源域名和 Hugging Face。

仅在需要访问这些服务的本项目子 shell 中启用，保留原始 `https://github.com/...` 地址：

```bash
(
  if [ -r /etc/network_turbo ]; then
    source /etc/network_turbo
  fi
  git -C "$DN_PROJECT_ROOT" fetch origin
)
```

同样适用于本项目的 GitHub clone/fetch/push 与 `git+https://github.com/...` 依赖获取。子 shell 退出后不改变父进程环境。不设置全局 Git proxy、不改 `.bashrc` 或系统配置，不把 GitHub URL 替换成来历不明的第三方代理地址。

脚本缺失或官方加速不可用时，记录状况后可尝试官方域名直连；不要打印代理变量、令牌或认证信息。盘点脚本仅记录 `/etc/network_turbo` 是否可读，不读取或输出其内容，也不会主动启用它。

Zenodo、PyPI、Conda 等下载域名不应被认为已经由此加速；按各自域名的实测可达性安排获取。不要将加速环境套在整个训练流程或其他项目上。

## 6. Blackwell 兼容起点与检查边界

首轮实际使用 Python 3.10.22、PyTorch 2.7.1 + cu128、torchvision 0.22.1、CUDA Toolkit 12.8、Nerfstudio 1.1.3、gsplat 1.0.0 源码构建，架构 `12.0`。正式 30k、评价和导出已经完成；具体兼容修改见 [patches](../patches/README.md)。独立梯度探针未单独执行，不把它的缺失当作正式运行失败。

官方依据：

- [NVIDIA GPU 计算能力表](https://developer.nvidia.com/cuda/gpus)：RTX PRO 6000 Blackwell 属于 12.0。
- [PyTorch 2.7 发布说明](https://pytorch.org/blog/pytorch-2-7/)：提供 Blackwell 与 CUDA 12.8 支持。
- [PyTorch 历史版本安装说明](https://pytorch.org/get-started/previous-versions/)：提供 2.7.1/0.22.1 cu128 配套版本。
- [gsplat 1.0.0 构建脚本](https://github.com/nerfstudio-project/gsplat/blob/v1.0.0/setup.py)：包含新旧两个 CUDA 后端。

首轮优先在新工具链上编译原 gsplat 1.0.0，并保留 DN-Splatter 需要的 legacy 法线接口。仅针对真实编译/API 错误做最小兼容修改，记录补丁与原因；不要一次性升级所有依赖。需要检查 RGB/depth 与 legacy 法线两条路径的前向、反向及一次真实优化更新。通过后继续正式 30k 训练，不能把导入成功或短迭代检查标为复现完成。

## 7. 全部 CPU/GPU 授权下的执行策略

本项目可以使用当前 AutoDL 实例全部已分配的 CPU 和目标 GPU；软件环境隔离继续有效。旧版“其他任务优先”“GPU 有活跃任务即持久排队”“CPU 初始只用 4–6 线程”的规则不再适用。若已有进程仍在运行，实际吞吐仍可能受竞争影响，应如实记录，不承诺其他任务零性能影响。

第一轮仍只运行一个 DN-Splatter 正式训练进程，以完成指定的 VR Room 基线。Omnidata 法线预计算、训练、评估渲染和网格导出各阶段均可使用目标 GPU；CPU 预处理、编译和 Poisson 可使用全部确认后的 CPU 预算。内存使用以容器实际限额和可用容量为准，监测峰值以避免 OOM，不为其他项目人为保留原来的计算预算。

执行策略：

1. 保存当前进程和资源的基线，用于诊断与耗时解释；不要给其他进程发信号。
2. 确认环境、数据、设备和阶段所需容量后直接继续。GPU 利用率不为零、存在其他 GPU 进程或旧队列状态，都不是继续无限等待的理由；若已有本项目队列，先检查其 PID 和阶段，避免重复启动。
3. 仅在实际阻塞时暂停相应阶段，例如设备不可用、显存/内存无法满足当前步骤、磁盘不足或 CUDA 分配失败。先保存真实错误和本项目状态，处理本项目可修复的占用或阶段重叠，不改论文算法来掩盖失败。
4. 对确实等待资源释放才能继续的步骤，记录 `waiting_for_resources`、所缺资源和恢复入口；可持久重试，周期约 30—60 秒，不做高频轮询。设备或技术错误记为 `blocked`，继续可完成的其他准备工作。
5. 全部资源授权不包含终止、暂停、迁移或重启其他项目，也不包含改变它们的优先级。若它们的实际占用阻止本项目继续，报告具体阻塞，让用户决定如何处理旧任务；不因为单纯存在竞争而退回旧的礼让策略。

不设整个项目、30k 训练或后台工作的完成截止时间。必须如实报告尚未运行的阶段，不能将排队或仅启动 tmux 作为完成。

本轮首要目标仍是完成一个场景的训练、评估和网格，不为占满硬件自动启动其他场景、消融或超参数搜索。获得完整结果后再汇报后续实验建议。

## 8. 可提交的盘点结论

推送仓库的环境摘要只包含与复现有关的事实：GPU/驱动、容器资源配额、CUDA/Python/PyTorch/gsplat/Nerfstudio 版本、环境前缀的项目相对路径、复用项及哈希、并发策略、资源状态、已执行步骤和仍未验证的部分。完整原始主机报告继续留在 `.local/`，不推送 GitHub。

若报告写着“环境已就绪”，必须同时说明两个 gsplat 后端是否已在目标 GPU 验证、数据是否准备完成以及 30k 训练是否开始；这些状态分别记录。
