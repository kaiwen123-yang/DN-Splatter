# DN-Splatter 源码与实验协议审计

更新时间：2026-11-18（Asia/Shanghai）。

本文件记录已经完成的论文阅读和源码静态审计，供 AutoDL 上的 Codex 执行者使用。**尚未在目标 RTX PRO 6000 上安装环境、训练或测得指标。** 源码核查不等于 Blackwell 兼容性已经验证，也不等于论文已经复现成功。

本轮任务是一个完整的 VR Room 实验闭环：环境适配、30k 训练、评估、Poisson 网格和结果回传。本文不授权另开超参数搜索、多场景训练或新的方法分支。

## 1. 固定来源与本轮实现身份

| 来源 | 固定版本或用途 |
|---|---|
| [DN-Splatter 正式论文](https://openaccess.thecvf.com/content/WACV2025/html/Turkulainen_DN-Splatter_Depth_and_Normal_Priors_for_Gaussian_Splatting_and_Meshing_WACV_2025_paper.html) | WACV 2025，正文页码 2421–2431 |
| [官方补充材料](https://openaccess.thecvf.com/content/WACV2025/supplemental/Turkulainen_DN-Splatter_Depth_and_WACV_2025_supplemental.pdf) | 初始化、数据划分、网格提取和评价协议 |
| [官方候选提交 811f5cd](https://github.com/maturk/dn-splatter/tree/811f5cdf342d92d12a9d9a4be185d374c430c184) | 本轮历史官方实现基线，完整 SHA：`811f5cdf342d92d12a9d9a4be185d374c430c184` |
| [核查时的官方 main](https://github.com/maturk/dn-splatter/tree/97588b4290128ce7ba6fdbfaac3020b42b17de4c) | 仅作为版本差异证据，完整 SHA：`97588b4290128ce7ba6fdbfaac3020b42b17de4c` |
| [MuSHRoom 官方数据说明](https://github.com/TUTvision/MuSHRoom) | `vr_room` 的 RGB-D、相机信息、划分和 Faro 参考几何 |

`811f5cd` 是损失策略重构之前的官方历史提交，不是作者明确认证的“论文唯一对应提交”。本轮结果应称为“固定历史官方实现的 VR Room 复现结果”。必须记录它与论文公式的差异，不能把运行成功写成严格复现成功。

## 2. 当前 main 的已确认实现差异

证据：[main 的 regularization_strategy.py](https://github.com/maturk/dn-splatter/blob/97588b4290128ce7ba6fdbfaac3020b42b17de4c/dn_splatter/regularization_strategy.py)；[main 的 dn_model.py](https://github.com/maturk/dn-splatter/blob/97588b4290128ce7ba6fdbfaac3020b42b17de4c/dn_splatter/dn_model.py)。

| 项目 | 源码事实 | 对本轮的决定 |
|---|---|---|
| 深度权重 | `get_depth_loss()` 最后执行 `depth_loss += self.depth_lambda * depth_loss`。对本轮所用 EdgeAwareLogL1 路径，参数 0.2 在这一步产生 1.2 倍中间损失。 | 不直接使用 main 的参数解释来对照论文的 0.2 权重。 |
| 法线权重 | `get_normal_loss()` 直接相加法线损失与平滑损失；配置的 `normal_lambda` 没有进入这条最终权重路径。 | 不能认为更改该 CLI 参数就改变了实际法线权重。 |
| 法线平滑开关 | `use_normal_tv_loss` 控制的 `self.tv_loss` 没有进入上述策略的最终计算；策略内部另行加入平滑项。 | 不用该开关在 main 上直接构造论文消融。 |
| 扁平化项开关 | `DNRegularization.get_loss()` 总是调用 `get_scale_loss()`；`two_d_gaussians` 未控制此调用。 | 关闭深度、法线后仍不能自动标记为纯 Splatfacto。 |

`git log -S 'depth_loss += self.depth_lambda'` 将相关重构定位到 [33b52d4](https://github.com/maturk/dn-splatter/commit/33b52d4)。该重构早于 AGS-Mesh 合并，因此仅选择“AGS-Mesh 之前的版本”仍不够。

这些是可从代码确认的事实。它们分别使论文指标变化多少，尚未进行实验；不得将未知影响写成确定的失败原因。

## 3. 历史候选也有必须披露的公式差异

证据：[811f5cd 的 dn_model.py](https://github.com/maturk/dn-splatter/blob/811f5cdf342d92d12a9d9a4be185d374c430c184/dn_splatter/dn_model.py)。

论文 Eq. (12) 给出的形式为：

```text
L = L_rgb + 0.2 L_depth + L_scale + 0.1 L_normal + 0.1 L_smooth
```

历史候选的实际路径，在本轮关闭额外 sparse/SDF 等项、启用法线和扁平化时，可概括为：

```text
L = L_rgb + 0.2 L_sensor_depth
    + 0.1 (L_normal + L_TV + L_flatten)
```

这里的名称指代码各自计算出的损失，不能忽略实现内部的归一化与数值约定。已确认的外部加权差异是：**历史版本把扁平化项也放进了 `normal_lambda` 加权的整体中。**

本轮保留历史实现，记录这一差异。不要为了让公式看起来一致而在首轮训练前静默重写目标函数；也不要把后续公式对照实验混入这一轮的结果。若实际执行需要改变损失语义，应另存配置、补丁及结果身份，并明确说明原定历史基线未按原方案完成。

## 4. 历史版本参数必须配套

证据：[811f5cd 数据解析器](https://github.com/maturk/dn-splatter/blob/811f5cdf342d92d12a9d9a4be185d374c430c184/dn_splatter/data/mushroom_dataparser.py)；[法线生成脚本](https://github.com/maturk/dn-splatter/blob/811f5cdf342d92d12a9d9a4be185d374c430c184/dn_splatter/scripts/normals_from_pretrain.py)；[网格导出](https://github.com/maturk/dn-splatter/blob/811f5cdf342d92d12a9d9a4be185d374c430c184/dn_splatter/export_mesh.py)。

| 用途 | `811f5cd` 的参数或入口 |
|---|---|
| 真实深度权重 | `--pipeline.model.sensor-depth-lambda 0.2` |
| Omnidata 法线监督 | `--pipeline.model.normal-supervision mono` |
| 训练数据法线格式 | `mushroom ... --normal-format opengl` |
| 单独生成 Omnidata 法线 | `--model-type omnidata` |
| 论文 Poisson 路线 | `gs-mesh dn` |
| MuSHRoom 表面采样 | `--total-points 5000000` |
| Poisson 八叉树深度 | `--poisson-depth 9` |

不要混入当前 main 的 `depth-lambda`、`normal-format omnidata` 等名称。该历史版本不提供后续加入的 `o3dtsdf` 入口。本轮逐场景执行 `ns-train`、`ns-eval`、`gs-mesh`，无需使用历史 `eval/batch_run.py`；后者存在重复关键字，不应为了首轮单场景任务展开批处理框架修复。

## 5. 初始化与数据划分

### 5.1 初始化点数要读取实际值

[初始化代码](https://github.com/maturk/dn-splatter/blob/811f5cdf342d92d12a9d9a4be185d374c430c184/dn_splatter/data/mushroom_utils/pointcloud_utils.py) 的 iPhone 路径以 `num_points=1_000_000` 为目标，但按帧抽样、可用点数和累积 TSDF 点云决定实际数量。它排除了 `long_capture/test.txt` 的测试帧，并由训练帧深度构造点云。

如果 `long_capture/iphone_pointcloud.ply` 已存在，解析器会直接读取，而不是重新保证它符合相同生成过程。执行者需要记录：

- 初始化文件来源、哈希、实际点数；
- 是否由本次固定版本生成，以及所用训练帧列表；
- 初始点与深度、位姿是否处于相同尺度和坐标系。

不要将未知来源点云直接标记为“严格使用 100 万训练深度点”。不要用 Faro 几何真值初始化本轮模型；Faro 是独立评价来源。

### 5.2 `within` 与 `with` 按实际帧集合解释

部分注释的说明与实际实现相反。本轮采用 `--eval-mode all`，并在报告里保存两个集合的帧列表和数量：

| 输出前缀 | 实际集合 |
|---|---|
| `within_*` | `long_capture/test.txt` 中同轨迹留出帧 |
| `with_*` | `short_capture` 独立采集轨迹 |

证据：[数据解析器](https://github.com/maturk/dn-splatter/blob/811f5cdf342d92d12a9d9a4be185d374c430c184/dn_splatter/data/mushroom_dataparser.py) 和 [评估 pipeline](https://github.com/maturk/dn-splatter/blob/811f5cdf342d92d12a9d9a4be185d374c430c184/dn_splatter/dn_pipeline.py)。不要混合两组平均值，或用其中一组冒充另一组。

## 6. 评估路径的已确认风险及处理方式

### 6.1 保留 `ns-eval` 的标准入口

[Nerfstudio 1.1.3 的 eval.py](https://github.com/nerfstudio-project/nerfstudio/blob/v1.1.3/nerfstudio/scripts/eval.py) 调用：

```python
pipeline.get_average_eval_image_metrics(
    output_path=self.render_output_path,
    get_std=True,
)
```

历史 DN pipeline 的 `get_std=False` 分支存在取 `within_` 前缀键的错误风险，标准 `ns-eval` 入口使用 `get_std=True`。本轮无需改为直接调用这个内部函数，也不应把未执行分支的问题误报成标准 `ns-eval` 已经失败。

不给 `--render-output-path` 时，`output_path` 为 `None`，该 pipeline 不执行后面的渲染导出和 Faro 参考深度评价。给出该参数后，会导出结果并调用 Faro 评价。因此，**指标 JSON 的路径与渲染目录是两种不同参数**；需要渲染和 Faro 指标时，必须准备对应参考深度。

### 6.2 Faro 深度指标存在跨轨迹累计问题

证据：[811f5cd 的 eval_faro.py](https://github.com/maturk/dn-splatter/blob/811f5cdf342d92d12a9d9a4be185d374c430c184/dn_splatter/data/mushroom_utils/eval_faro.py)。

已确认：

1. 各项 batch 指标列表在内部 `calculate_metrics()` 函数外创建。
2. 函数先处理 `long_capture`，再处理 `short_capture`。
3. 第二次调用没有清空列表，`with_faro_*` 因而混入前一组的 batch 指标。
4. 代码对 batch 指标等权平均；如果最后一批不足完整 batch 大小，结果会依赖分批方式。RMSE 等非线性指标的汇总还需要明确统计定义。

处理规则：保留官方原始评估输出及其身份；在形成最终结论前，使用同一组导出预测和参考深度，完成一次明确记录的分轨迹重算。结果分别保存为原始输出与经审计输出，不覆盖、不混名。经审计实现应在每条轨迹内独立累计，并记录选择的是逐帧平均还是全有效像素汇总，以及有效帧/像素数量。

这一问题属于评估汇总，不需要重训模型。不要根据污染的 `with_faro_*` 调整模型权重，也不要把参考缺失、空 batch 等异常替换成 0 分后继续宣布成功。

### 6.3 网格评价遵循原协议

正文及补充材料明确：Poisson depth 为 9；MuSHRoom 提取使用 500 万表面采样点；F-score 阈值为 5 cm；评价限制在训练相机可见范围。执行应使用 [MuSHRoom 可见性裁剪评估脚本](https://github.com/maturk/dn-splatter/blob/811f5cdf342d92d12a9d9a4be185d374c430c184/dn_splatter/eval/eval_mesh_mushroom_vis_cull.py)，并记录几何对齐、单位、裁剪与采样参数。

`--total-points` 是网格提取的目标表面采样数，与初始化高斯数、最终高斯数及网格评估阶段的表面采样数是不同概念。应同时记录请求值和实际产物数量。原始网格和裁剪后网格都保留，不能用手工修补后的外观图替代原始方法输出。

## 7. 首场景参考值与结论边界

上传论文第 2427 页 Table 4 给出 VR Room 的 30k 实验：

| 方法 | PSNR | SSIM | F-score |
|---|---:|---:|---:|
| Splatfacto，仅 RGB | 26.6 | 0.870 | 56.3% |
| 加深度监督 | 27.1 | 0.885 | 86.6% |
| 加深度与法线 | 26.9 | 0.883 | 87.7% |
| 加深度、法线与平滑 | 26.8 | 0.879 | 87.1% |

本轮只执行最后一类配置的历史官方实现基线。表格中的其他行是论文参考值，不是本仓库已经跑出的结果，也不要求本轮额外运行这些消融。

这些参考值不构成任意设置下的硬阈值承诺。报告应列出绝对差值、数据划分和实现差异，再判断结果达到何种程度。单场景闭环完成不等于整篇论文复现完成；论文六场景平均数也不能作为 VR Room 单场景目标。5 cm 阈值的 F-score 不证明毫米级几何精度。

## 8. 只针对真实问题做最小适配

| 改动类别 | 本轮处理 |
|---|---|
| Blackwell / PyTorch / C++ 扩展构建兼容 | 可以做必要的最小改动，保存完整 diff、版本与触发错误。 |
| 路径、下载来源、文件格式或已确认运行错误 | 可以修复；保留原数据、原始报错和转换记录。 |
| Faro 指标分组及聚合 | 允许独立重算；原始结果与经审计结果分别保存，说明统计定义。 |
| 损失权重、法线来源、初始化规模、模型类型 | 属于实验语义变化，不为赶进度静默修改；本轮优先保持固定历史基线。 |
| 换用新版 gsplat / AGS-Mesh / 另一种网格方法 | 不作为本轮环境适配的默认解决方式。 |

新旧后端都需通过实际前向和反向验证，然后进入正式训练。检查只覆盖真实风险，不增加与本轮目标无关的测试堆砌。记录所有差异，避免把“修复后运行完成”误写成“未经修改的官方版本已完全兼容”。

## 9. 全资源运行与软件隔离的边界

用户已将本轮切换为全资源运行，允许使用 AutoDL 实例分配的全部 CPU 和 GPU。独立 Conda、源码、缓存和输出目录继续保留，用于避免软件环境互相修改；若仍有其他进程运行，不能据此承诺没有性能竞争。

CPU 预算按 affinity/cpuset、cgroup quota 和实例实际分配共同确认，标称为 25 核。编译、预处理和 Poisson 按阶段使用全部合理预算，Poisson 显式记录 `n_threads`；避免多个并发层各自创建全核线程。训练、法线推断和网格渲染允许使用整卡，首轮仍只执行一个 VR Room 主训练任务，不增加实验或修改论文协议来填满资源。

其他进程活跃或 GPU 利用率高不再是自动排队条件；在实际容量允许时继续执行。仅在明确显存/内存/磁盘不足或当前阶段无法运行时，记录原因并可恢复等待或报告阻塞。不结束其他项目进程，不更改全局驱动、GPU 设置或它们的环境；需要处理故障时只操作本项目。报告仍须区分等待、训练中、失败与各阶段完成，不能把全资源授权写成已经在服务器验证成功。

## 10. 执行后必须补全的证据

- 目标服务器实际环境与资源情况，以及最终依赖锁定记录；
- 源码 SHA、所有适配补丁、真实命令与最终配置；
- 输入来源、划分列表、法线模型身份、初始化点云来源及哈希；
- 30k 训练完成证据、checkpoint、耗时、资源峰值和高斯数量；
- 两条轨迹的 RGB/传感器深度结果、独立 Faro 重算定义及输出；
- 原始/裁剪网格、可视化和网格指标；
- 与论文参考数值的差距，以及已确认原因和仍未确定的问题。

在这些证据产生之前，所有训练状态和指标都保持未运行或待验证，禁止填入推测数值。
