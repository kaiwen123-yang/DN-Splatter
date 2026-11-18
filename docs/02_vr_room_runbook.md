# VR Room 首轮复现执行手册

更新日期：2026-11-18（Asia/Shanghai）。状态：**计划与源码核查完成，尚未在目标 AutoDL 服务器上验证**。

本轮目标是在同一轮持续任务中完成 **Blackwell 环境适配 → VR Room 单场景累计 30,000 次优化 → RGB/深度评估及渲染 → Poisson 网格 → 网格评估 → 结果推送**。首轮只运行一个训练任务。已有项目优先，资源忙时排队；不为本项目终止、迁移或修改其他任务。

先读仓库根目录 `AGENTS.md` 和 [隔离与复用要求](01_autodl_isolation.md)，再执行本手册。机器规格由用户提供：RTX PRO 6000 96GB、120GB RAM、25 核 CPU；现有占用需要在服务器现场检查。

## 1. 本轮到底复现哪个实现

| 项目 | 本轮固定值 |
|---|---|
| 官方仓库 | <https://github.com/maturk/dn-splatter> |
| 初始候选提交 | `811f5cdf342d92d12a9d9a4be185d374c430c184` |
| 方法入口 | `dn-splatter` |
| Nerfstudio / gsplat | `1.1.3` / `1.0.0` |
| 数据 | MuSHRoom `vr_room`，`iphone` |
| 划分 | 官方 `long_capture/test.txt` 与独立 `short_capture`，`eval-mode all` |
| 训练目标 | 累计 30,000 次优化；从零训练的最后零基 step 为 `29999` |
| 法线 | Omnidata，历史脚本 `model-type omnidata`，历史 parser `normal-format opengl` |
| 深度 | 真实传感器深度，`EdgeAwareLogL1`，`sensor-depth-lambda 0.2` |
| 法线参数 | `normal-lambda 0.1`，启用 mono 法线、TV、扁平化 |
| 网格 | `gs-mesh dn`，目标采样 5,000,000 点，Poisson depth 9 |
| 几何评价 | 官方训练相机可见性裁剪，F-score 阈值 0.05 m |

**名称应写成“历史官方实现基线”，不要预先写成“论文完全复现成功”。** 该提交是基于源码审计选出的候选，不是作者公布的唯一论文提交。

该历史实现把法线 L1、TV 和高斯扁平化项合并后乘 `normal_lambda`；论文公式把扁平化项单列。首轮保留历史实现的算法行为，记录区别，不悄悄重写损失来追指标。当前主分支的损失重构还有其他权重/开关差异，因此不自动跟随 `main`。详细证据可查看文末固定源码链接。

本轮不自动增加 Splatfacto、AGS-Mesh、`dn-splatter-big`、其他场景、其他法线网络或超参搜索；先完成用户指定闭环。

## 2. 目录与执行约定

执行器先解析真实仓库路径，然后把下列值写进本项目本地运行记录；不要把占位路径直接当真实路径执行。

```bash
DN_PROJECT_ROOT="/实际仓库绝对路径/DN-Splatter"
DN_SOURCE_DIR="$DN_PROJECT_ROOT/.local/upstream/dn-splatter"
DN_ENV_PREFIX="$DN_PROJECT_ROOT/.local/conda/dn-splatter-bw"
DN_DATA_ROOT="$DN_PROJECT_ROOT/.local/data"
DN_RUN_ROOT="$DN_PROJECT_ROOT/.local/runs/实际运行编号"
DN_SCENE_DIR="$DN_DATA_ROOT/room_datasets/vr_room"
```

- `.local/` 下的环境、源码工作副本、数据、缓存、运行输出属于本项目；不提交大文件到 Git。
- 缓存使用 `.local/cache/{pip,hf,torch,xdg,cuda_extensions,conda_pkgs}`，具体变量见隔离文档。
- 以下上游命令以 `DN_SOURCE_DIR` 为工作目录，并使用绝对数据/输出路径。
- 原仓库外部数据若可复用，先核对来源、版本和校验值，再独立复制/写时复制到本项目。parser 会生成点云、法线等文件，不能把其他项目的可写数据目录直接当本项目工作目录。
- 所有命令保留实际参数、退出码和日志。`tee` 管道必须保留训练进程退出码，例如使用 `set -o pipefail`；退出码仍不能代替 checkpoint 步数检查。
- GitHub 的 clone/fetch/pull/push 优先按隔离文档使用 AutoDL 官方学术加速：只在本项目命令的子 shell 内 `source /etc/network_turbo`，保留 GitHub 官方 HTTPS URL，不写全局 Git/Conda/pip 代理。Zenodo 权重和数据下载不属于已核实承诺的加速域；不要把 GitHub 子 shell 的代理环境无差别继承到所有下载或训练命令。

## 3. 阶段 A：完成环境适配与必要检查

环境候选为 Python 3.10、PyTorch 2.7.1 + cu128、torchvision 0.22.1、CUDA Toolkit 12.8、Nerfstudio 1.1.3、从源码构建的 gsplat 1.0.0、`TORCH_CUDA_ARCH_LIST=12.0`。这些版本是有依据的起点，是否兼容必须现场确认。

1. 只读检查 GPU/CPU/RAM/磁盘、已有任务、Conda 环境和 CUDA Toolkit，形成环境盘点与复用决定。
2. 新建本项目独立 Conda prefix；只读复用可验证的安装包、权重、数据和已有 Toolkit。基础环境、其他项目环境、驱动和全局配置均保持原状。
3. 在本项目构建 gsplat 的普通渲染与 legacy 法线两个后端。记录实际 PyTorch、CUDA runtime、`nvcc`、编译器和扩展源码版本。
4. 只做能消除具体风险的检查：CLI 参数存在；一组真实 RGB/深度/法线前向与反向有效，所需参数梯度非空且有限；模型 checkpoint 可保存并恢复。复用同一次短检查，不叠加多轮短训练。
5. 一旦上述检查通过，就进入正式 30k；不要把“成功导入包”“跑了几十步”当成完成任务。

实际编译错误允许在**本项目副本**中做最小兼容修复，记录原始报错、修改目的、diff 和验证结果。区分 `build_compatibility`、`resource_limit`、`evaluation_correction` 与 `algorithm_change`。前三类必要修复可以持续推进；算法更改不混入首轮基线。

注意 PyTorch 新版本的 checkpoint 加载行为、旧 C++/CUDA 接口、图像/网格依赖 API 都可能产生真实兼容问题；只针对实际错误修复，不预先大范围改库。确需改变 checkpoint 加载参数时，仅针对来源已核实的官方权重和本轮自己生成的 checkpoint，不能全局取消加载约束。

## 4. 阶段 B：准备单场景数据和先验

先检查本项目目录中是否已有完整数据，避免反复下载和解包。官方脚本的成功文字不能替代文件核验；某些下载函数没有严格传播下载失败。

```bash
cd "$DN_SOURCE_DIR"

python dn_splatter/data/download_scripts/download_omnidata.py

python dn_splatter/data/download_scripts/mushroom_download.py \
  --room-name vr_room --sequence iphone --save-dir "$DN_DATA_ROOT"

python dn_splatter/data/download_scripts/mushroom_download.py \
  --room-name vr_room --sequence faro --save-dir "$DN_DATA_ROOT"
```

Omnidata 历史脚本默认读取源码工作目录中的 `omnidata_ckpt/omnidata_dpt_normal_v2.ckpt`；该目录位于本项目 `.local/upstream` 内。复用权重时复制到此处或使用本项目控制的只读引用，并记录来源与校验值，不改其他项目的权重文件。

需要输出渲染并计算 Faro 独立深度评价时，还需要参考深度：

```bash
python dn_splatter/data/mushroom_utils/reference_depth_download.py \
  --save-dir "$DN_DATA_ROOT"
```

这个入口下载参考深度总包，不是逐房间接口。下载/解包前检查剩余磁盘与压缩包大小。若只能取出 VR Room 所需成员，记录所选成员清单和来源；不能默默删除 GT 评价步骤，也不能为腾空间删除其他项目数据。

### 数据检查清单

- `iphone/long_capture` 和 `iphone/short_capture` 都有图像、深度和 `transformations_colmap.json`。
- `long_capture/test.txt` 存在，训练帧与留出帧不重叠；记录三组实际帧名：训练、同轨迹留出、独立轨迹。
- 逐帧确认 RGB、深度、法线命名与尺寸对应，避免只统计数量。相机位姿、深度单位和当前 parser 的缩放一致。
- 记录实际加载分辨率、downscale、位姿来源、深度有效像素比例以及数据/权重 manifest。不要为了让显存占用变小而隐式改变论文协议。
- 点云只由训练帧构建。历史函数以约 100 万点为目标；已有 `iphone_pointcloud.ply` 会直接复用，应记录来源、实际点数、hash，而不是不经检查声称正好 100 万点。
- Faro GT 只用于评价；`use-faro-scanner-depths=False`、`use-faro-scanner-pd=False`，避免把评价真值加入训练或初始化。
- 训练前抽查少量 RGB/深度/法线可视化，以及几帧反投影的几何对齐。这是数据检查，不启动额外训练轮次。

历史 `mushroom` parser 可以在法线缺失时调用 Omnidata。若先显式预计算，使用该版本真实参数：`--model-type omnidata`、`--transforms-name transformations_colmap.json`、`--save-path <该序列>/normals_from_pretrain`，长短两段分别处理。Omnidata `low` 模式内部使用 384×384 输入再还原输出尺寸，不代表训练 RGB 应统一改成 384×384。法线预测也使用 GPU，必须遵守与正式训练相同的资源准入。

## 5. 阶段 C：正式累计 30k 训练

初始单进程，CPU/native 线程从 4–6 起步，具体预算由环境检查确定。共享 GPU 没有资源隔离机制时不能保证零性能影响；已有任务繁忙时保留可恢复的等待任务，先完成允许的 CPU 工作。不要因为空闲显存看起来足够，就擅自争抢其他任务的 GPU 计算资源。

以下是从零开始的模板。只有资源准入通过、环境和数据检查通过后才执行。参数顺序已对照历史源码：模型/Trainer 参数在 `mushroom` 之前，数据参数在之后。

```bash
cd "$DN_SOURCE_DIR"

ns-train dn-splatter \
  --output-dir "$DN_RUN_ROOT/training" \
  --experiment-name vr_room_iphone_dn_official \
  --machine.seed 42 \
  --max-num-iterations 30000 \
  --steps-per-save 5000 \
  --save-only-latest-checkpoint False \
  --vis tensorboard \
  --viewer.quit-on-train-completion True \
  --pipeline.model.use-depth-loss True \
  --pipeline.model.depth-loss-type EdgeAwareLogL1 \
  --pipeline.model.sensor-depth-lambda 0.2 \
  --pipeline.model.use-normal-loss True \
  --pipeline.model.normal-lambda 0.1 \
  --pipeline.model.use-normal-tv-loss True \
  --pipeline.model.normal-supervision mono \
  --pipeline.model.two-d-gaussians True \
  mushroom \
  --data "$DN_SCENE_DIR" \
  --mode iphone \
  --normal-format opengl \
  --eval-mode all \
  --load-depths True \
  --depth-mode sensor \
  --load-normals True \
  --normals-from pretrained \
  --use-faro-scanner-depths False \
  --use-faro-scanner-pd False
```

`tensorboard` 是无交互窗口的运行方式，保留训练日志且不占用他人的 viewer 端口。若使用 viewer，必须使用本项目独立端口并在训练完成后退出；不能让完成训练的 viewer 长期保留模型显存。

首次运行前只核对一次对应 `--help`，并保存解析后的真实 `config.yml`。若安装的 Tyro 显示不同参数形式，以**同一固定源码**真实 CLI 为准记录调整，不能借此切换到新主分支配置。不要使用该历史版本含重复关键字问题的 `eval/batch_run.py`。

### 断点与恢复：Nerfstudio 1.1.3 有特殊语义

已检查该版本 `Trainer.train()`：循环为 `range(_start_step, _start_step + max_num_iterations)`。恢复后 `max_num_iterations` 表示**这次追加多少步**，不能再次给 30,000。

设 checkpoint 内实际 `step=s`，则：

```text
completed = s + 1
remaining = 30000 - completed
```

- 若 `remaining > 0`，沿用原完整训练参数，加 `--load-dir <原nerfstudio_models目录>`、`--load-step <s>`，将 `--max-num-iterations` 改成 `remaining`；输出到记录关联关系的恢复段目录。
- 保留原 30k 学习率调度、优化器/调度器/梯度缩放器状态和原输入划分。记录恢复前后对应状态与 LR，不能只加载模型权重后当作正常续训。
- 若 `remaining == 0`，直接开始评估。若负数，标记训练协议偏离，不继续累加。
- `--load-config` 会在 `train.py` 中用 YAML 对象整体替换 CLI 配置，因此不能认为同时传的 CLI 覆盖一定生效。使用完整原 CLI+恢复参数，或先生成并检查完整恢复配置，两种方式选择一种。
- checkpoint 记录的零基 `step=29999` 才对应从 step 0 完成 30,000 次优化；不能按日志中一个“30000”字符串或文件个数判断。
- 原 checkpoint 不完整保存所有随机数发生器状态，恢复链必须如实记录，不能承诺中断前后逐位一致。

训练命令不包完成期限 `timeout`。使用独立 `tmux` 或同类持久进程；真实错误先保存日志、定位并最小修复，再从本项目 checkpoint 继续。只有本项目的资源冲突、错误进程可以由执行器处理，不能碰其他任务。

## 6. 阶段 D：评估、渲染和指标审计

解析训练日志得到最终 `DN_CONFIG`，核对它引用的最终 checkpoint 和数据。以下路径均为本项目自己的输出目录。

### 6.1 先保存不触发 Faro 的官方 RGB/传感器深度结果

```bash
ns-eval \
  --load-config "$DN_CONFIG" \
  --output-path "$DN_RUN_ROOT/evaluation/rgb_depth_official.json"
```

Nerfstudio 1.1.3 `ns-eval` 将 `--render-output-path` 传到 pipeline，**`--output-path` JSON 路径本身不会触发渲染或 Faro 评价**。

| 指标前缀 | 真实含义 |
|---|---|
| `within_*` | `long_capture/test.txt` 的同轨迹留出帧 |
| `with_*` | 独立 `short_capture` 轨迹 |

历史 parser 的文字注释把二者描述反了，以实际索引与文件列表为准。首轮使用 `all`，保留两个集合；不能把两个轨迹的指标混为一个数字。

### 6.2 保存 RGB、深度、法线渲染及原始 Faro 输出

在 Faro reference depth 准备齐全后：

```bash
ns-eval \
  --load-config "$DN_CONFIG" \
  --output-path "$DN_RUN_ROOT/evaluation/rgb_depth_faro_official_raw.json" \
  --render-output-path "$DN_RUN_ROOT/renders_official"
```

检查真实生成的 RGB、depth、normal 以及 depth raw 数组。至少保存覆盖两个测试轨迹的固定视角对照图，并做视觉检查，不能只检查文件存在。若某输出需要修复历史代码的真实兼容错误，保持渲染定义不变并记录补丁。

### 6.3 Faro 原评估存在聚合问题：raw 与 audited 分开

历史 `depth_eval_faro()` 在外层创建 batch 指标列表，然后先后处理长、短两段轨迹；第二段会累积第一段的列表。另外，尾批次与完整批次做等权平均。这会影响 `with_faro_*` 的解释。

执行器必须：

1. 保留能获得的原始官方 JSON；若原脚本报错，保留原错误，不能伪造 raw 指标。
2. 对已保存的同一批渲染深度运行独立评估修正：每个 split 单独累计，记录样本/有效像素数量，并明确使用逐帧平均还是有效像素汇总；不要静默把两种定义互换。
3. 输出独立 `faro_audited.json` 和修正说明。修改只作用于本项目评估副本，不需要重新训练，也不覆盖 raw 文件。
4. 在结果表中明确分开 RGB、传感器深度、Faro 参考深度，写明 raw 中受影响的键。不能把修正后的指标冒充原脚本输出。

## 7. 阶段 E：Poisson 网格与几何评价

训练进程完成并释放 GPU 后再提取网格。导出阶段仍会加载模型渲染；CPU Poisson 也可能使用很多线程。先检查当前资源，沿用本项目预算。

历史导出器没有显式传入 Open3D Poisson 的 `n_threads`。共享服务器上应核对安装版本的接口，并在**本项目导出副本**中明确设置不超过本项目 CPU 预算的线程数；记录这一资源补丁，不改变采样量、Poisson depth、过滤或网格阈值。

```bash
gs-mesh dn \
  --load-config "$DN_CONFIG" \
  --output-dir "$DN_RUN_ROOT/mesh" \
  --total-points 5000000 \
  --poisson-depth 9
```

正常网格路径：

```text
<DN_RUN_ROOT>/mesh/DepthAndNormalMapsPoisson_poisson_mesh.ply
```

`--total-points` 是目标采样量，应记录实际提取/使用点数。不要换成历史版本不存在的 `o3dtsdf`，也不要将当前 README 的默认 200 万点当成 MuSHRoom 论文设置。

```bash
python dn_splatter/eval/eval_mesh_mushroom_vis_cull.py \
  --gt-mesh-path "$DN_SCENE_DIR" \
  --pred-mesh-path "$DN_RUN_ROOT/mesh/DepthAndNormalMapsPoisson_poisson_mesh.ply" \
  --device iphone
```

`gt-mesh-path` 这里需要的是包含 GT、对齐文件和采集数据的场景目录。脚本会做坐标对齐和训练视野可见性裁剪，默认在预测 mesh 目录保存 `mesh_metrics.json` 和裁剪后的 mesh。保留未裁剪与裁剪后两份；不把整场景未裁剪指标与论文裁剪指标混用。

检查网格非空、顶点/面有效、包围盒及单位合理，并保存至少两张固定视角网格图。任何网格清理、异常点过滤、简化或阈值更改都写入配置；不以额外清理掩盖原模型问题。

## 8. 阶段 F：交付、判断与推送

### 最低交付

| 内容 | 必须包含 |
|---|---|
| 环境 | 只读盘点、复用决定、独立 Conda 路径、包版本、CUDA/编译信息、可重建安装命令 |
| 源码 | 上游 SHA、基线与适配补丁、补丁分类、实际入口命令 |
| 数据 | 来源 URL/hash、三组帧清单、分辨率/内参/尺度、初始化点数、法线来源 |
| 训练 | 配置、日志、最终 step/累计步数、checkpoint 清单、恢复链、时间与峰值资源 |
| 外观/深度 | 官方 RGB-depth JSON、两个轨迹固定视角 RGB/深度/法线图、raw/audited Faro 结果 |
| 几何 | Poisson mesh、裁剪 mesh、原始几何 JSON、固定视角网格图 |
| 结论 | 完成与未完成项、论文参考差距、运行错误及修复、下一步建议 |

小型配置、脚本、补丁、JSON、摘要和代表性图像按仓库约定推送到用户仓库。环境、原始数据、检查点、大型 mesh/渲染数组保留在服务器本项目目录，用清单记录路径、大小和 hash；不塞进 Git 普通历史，不做用户电脑归档。公开推送前清除 token、SSH 信息、包含密钥的 URL、敏感进程参数和不必要的其他项目路径。

**完成闭环**意味着上述阶段有真实产物且累计步数正确。**论文对齐**还要求审计协议、代码差异及具体指标，二者不是同一结论。

论文表 4 的 VR Room 完整配置参考为 PSNR 26.8、SSIM 0.879、F-score 0.871（阈值 5 cm），30k、RTX 4090 训练计时 17.9 min。它不是本服务器的实测结果，也不是强制耗时上限。报告真实差距，不设置“达到某阈值就隐藏其他失败”的判据；单场景闭环结束后停止新增训练实验，先推送与汇报。

如果因资源持续繁忙、必要下载不可达或实际技术错误未完成，状态应为 `waiting_for_resources` 或 `blocked`，包含精确阶段、日志、可恢复入口；不能写成成功。可持久等待的任务继续等待，不给训练设置后台完成截止时间。

## 9. 固定源码与论文来源

- [官方历史模型](https://github.com/maturk/dn-splatter/blob/811f5cdf342d92d12a9d9a4be185d374c430c184/dn_splatter/dn_model.py)
- [历史训练配置](https://github.com/maturk/dn-splatter/blob/811f5cdf342d92d12a9d9a4be185d374c430c184/dn_splatter/dn_config.py)
- [历史 MuSHRoom parser](https://github.com/maturk/dn-splatter/blob/811f5cdf342d92d12a9d9a4be185d374c430c184/dn_splatter/data/mushroom_dataparser.py)
- [历史网格导出](https://github.com/maturk/dn-splatter/blob/811f5cdf342d92d12a9d9a4be185d374c430c184/dn_splatter/export_mesh.py)
- [历史网格评价](https://github.com/maturk/dn-splatter/blob/811f5cdf342d92d12a9d9a4be185d374c430c184/dn_splatter/eval/eval_mesh_mushroom_vis_cull.py)
- [历史 Faro 深度评价](https://github.com/maturk/dn-splatter/blob/811f5cdf342d92d12a9d9a4be185d374c430c184/dn_splatter/data/mushroom_utils/eval_faro.py)
- [Nerfstudio 1.1.3 Trainer：checkpoint 与恢复循环](https://github.com/nerfstudio-project/nerfstudio/blob/v1.1.3/nerfstudio/engine/trainer.py)
- [Nerfstudio 1.1.3 train.py：load-config 行为](https://github.com/nerfstudio-project/nerfstudio/blob/v1.1.3/nerfstudio/scripts/train.py)
- [Nerfstudio 1.1.3 ns-eval](https://github.com/nerfstudio-project/nerfstudio/blob/v1.1.3/nerfstudio/scripts/eval.py)
- [WACV 2025 正文](https://openaccess.thecvf.com/content/WACV2025/papers/Turkulainen_DN-Splatter_Depth_and_Normal_Priors_for_Gaussian_Splatting_and_Meshing_WACV_2025_paper.pdf)
- [WACV 2025 补充材料](https://openaccess.thecvf.com/content/WACV2025/supplemental/Turkulainen_DN-Splatter_Depth_and_WACV_2025_supplemental.pdf)

这些文件在准备计划时已做源码/文献检查，GPU 运行验证仍待执行。实际安装后还应核对本地文件与固定版本一致。
