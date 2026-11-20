# capture-full：Metric3D v2 法线完整对照

日期：2026-11-20（Asia/Shanghai）。状态：**计划已形成，本次审阅未启动服务器推理或训练**。用户“metrics”暂按 Metric3D v2 理解；现有 PSNR/SSIM/LPIPS/深度 metrics 已经运行，不能把新先验模型与评价函数混同。

## 1. 目的与对照

在同一完整采集、同一几何和同一 30k 预算下，判断用 Metric3D v2 法线替换 Omnidata，是否改善 DN 的 RGB/深度和最终表面。依据见 [全程复核](../reports/capture-full-review/2026-11-20/overview.md)。ICCV 2025 EmbodiedSplat §3.2 有这条路线的直接先例，但不等于本采集必然改善。

| 模型 | 法线 | 训练输入及调度 | 原始导出 | 同一支持约束导出 |
| --- | --- | --- | --- | --- |
| B0：已完成 full | Omnidata | 冻结原配置，30k | 已有 5M / depth 9 | 本轮从旧最终 checkpoint 新导出 |
| B1：本轮 | Metric3Dv2 ViT-Large | 与 B0 相同，重新初始化并完整 30k | 同协议 5M / depth 9 | 与 B0 完全相同参数 |

这是 **一个新训练任务**，不是两个新训练或参数搜索。B1/B0 的模型指标与同种导出对照用于评价法线替换；同一个模型的两种导出比较用于评价导出适配。B1 支持网格相对 B0 原始网格的总变化不能单独归给 Metric3D。

## 2. 冻结输入与身份

- B0 结果提交：`e26ae72a75997dcfd3a907aa26b6c5c0677597a5`。
- B0 checkpoint SHA256：`b89ad55fb5781582a6306a8c1f006ae5afe1e7f7ee5704c1594c8b7bbe6b8094`；路径以原 artifact_manifest 为准。
- 几何：`.local/data/custom/capture-full/20261119-full-5hz-3982cb26`；manifest SHA256 `05873700af32712450b319f9f44f0f4cffa0735f974390be0dbd84b57f803802`。
- B0 训练输入：`.local/runs/capture-full-dn-30k-20261119-v1/inputs`；manifest SHA256 `4982f0a5a67fb0e1d90fc0c23bbf81b238f949e8bcb63412f8b6fc8a559e286a`。
- 1,846 主帧，1,478 train / 184 val / 184 test；100 辅助图只属于既有建图，不进入本轮先验/训练/尺度/点云采样。
- RGB 1280×720、原去畸变 K、原米制 OpenGL c2w、0.3–6 m 传感器光轴 Z、相同 2,001,906 点初始化、scene_scale=21。
- DN 历史源码 `811f5cdf342d92d12a9d9a4be185d374c430c184` 与已验证 Blackwell 补丁；沿用 B0 实际 Nerfstudio/gsplat/优化器与损失配置。

原数据只读复用，B1 在新的输入目录写法线和 transforms 路径副本。禁止通过可写硬链接修改 B0；明确文件共享方式和内容哈希。正常核对仅在本次正式准备中做一次。现成数据无需回到 Windows 提取、传输或重做位姿。

新 run：`.local/runs/capture-full-metric3d-30k-20261120-v1/`；报告：`reports/capture-full-metric3d-30k/20261120-v1/`。既有同身份 run 按状态、锁、PID、输入清单续接，禁止重复启动。

## 3. Metric3D 推理和法线接口

固定官方 `YvanYin/Metric3D` 提交 `eb5b6fac0dc155e4e52f576e304fbf11655ff339`，入口 `metric3d_vit_large`，权重 `metric_depth_vit_large_800k.pth`，RAFT 8 iterations。从官方渠道取权重，在全量推理前记录实际字节、SHA256、下载 URL/可取得的远端版本；本协议不虚构尚未下载权重的哈希，不自动改用 Small/Giant2。

推理依照 [固定数值适配](12_metric3d_normal_adapter.md)：保持宽高比到 616×1064 网络画布，均值 padding，推理后去 padding、恢复原训练网格并单位化前三通道。按原始 CV 相机 XYZ 编码 RGB PNG，保持当前 parser `normal_format=opengl`，让历史读取器执行唯一一次 `diag(1,-1,-1)`；以实际模型/loader 数学关系为准。禁止直接拿 OpenCV 可视化图作为训练法线。第 4 通道 κ 留档，不加入置信度阈值或新损失权重。

全量推理 1,846 帧，数据读取核对逐帧尺寸、顺序、有限性、PNG→历史 reader 的数值转换。归档原始有符号法线、κ 和深度；训练仅用原 train 法线。val/test 的预测只用于诊断展示，不参与初始化、训练、阈值选择或网格筛选。

深度使用真实去畸变 K，记录源像素中心约定，先正确变换到网络画布；去规范化仅用一次 `resized_fx/1000`。保存校正前后和实际缩放/裁剪元数据。**本轮模型训练继续只使用传感器 Z，mono_depth_lambda=0**。Metric3D 深度只与留出有效传感器 Z 比较，不用逐帧或 test 拟合尺度，不填孔、不把传感器监督扩至 10 m。既有相机 K 的半像素偏移不能重复应用。

## 4. 环境与资源

沿用 `.local/conda/dn-splatter-bw` 的已工作训练环境。Metric3D 官方旧 requirements 固定了旧 Torch，不能直接在该环境执行整份 `pip install -r` 或降级 CUDA/Torch。先只读核对实际依赖，合适的模型缓存和 wheel 可以复用；推理缺依赖时用独立 `.local/conda/metric3d-bw`，采用已验证支持 Blackwell 的 Torch/CUDA 组合，记录最小兼容补丁。官方 hub 有 mmengine Config 回退，优先核实真实依赖；不要为不存在的错误重建训练环境。

允许全部实际 CPU 配额和整卡 GPU。以现有实际内存为依据：B0 训练进程峰 RSS 约 79.32 GiB，cgroup 峰约 117.98 GiB。全量推理、DN 训练、支持网格依次执行；推理模型完成即释放，不与完整 DN 缓存并存。网格分块处理，不累计 38M 点的多份 float64 副本。线程数防止 worker×native 过度嵌套。

GitHub 操作在 AutoDL 本项目子 shell 优先 `source /etc/network_turbo`，保留官方 HTTPS 地址；独立处理 Hugging Face 下载，不写全局代理。不修改其它项目环境或进程，不设置自动关机、训练 deadline 或 watchdog 强杀。

## 5. 完整训练与记录

从相同初始化点云重新训练 B1，累计 30,000 更新，终点 step 29,999。B0/B1 配置语义 diff 必须随报告交付：可变项只有先验模型/法线输入、输出身份与不改变计算的记录代码。保持 seed42、原采样行为、学习率、损失、相机、分辨率、点云及所有 refine/cull 设置。正常路径不修改历史 DN 算法；确有接口错误时记录具体修复及数值影响。

核心值：EdgeAwareLogL1、sensor_depth_lambda=.2、mono_depth_lambda=0、normal_lambda=.1、mono normal、normal TV=True、use_depth_smooth_loss=False、two_d_gaussians=True、num_downscales=0、camera optimizer off、skip_point_metrics=True；warmup500、refine_every100、reset_alpha_every30、stop_split15000、stop_screen4000、continue_cull_post=True。

**明知旧采样器首遍后按顺序遍历，本轮仍保留这一行为以隔离法线差异。** 不同时加入每轮 shuffle、240k、阈值调整、动态 mask 或新单目深度。增加训练曝光量是下一独立实验的问题，不是本轮自动 fallback。

在启动 B1 前读取 B0 已有 TensorBoard、训练日志和 5k/10k/15k/20k/25k/final checkpoints，输出真实高斯数量、各 loss 分量与已有标量曲线。旧日志没有的 cull 原因写 unavailable，不从文件大小猜测。B1 随正式训练记录 sample frame/index、增密前后数量、split/duplicate/父点替换、低 opacity/世界尺度/屏幕尺度剔除及 union；重叠原因不能相加充当实际删点数。记录不能消耗额外 RNG 或改变算子顺序。

首个正式 batch 检查输入、loss/梯度有限性后继续整轮，不开 smoke 训练。每 5k 保存 checkpoint，最终身份唯一。恢复时按 saved_step 计算剩余更新，沿用原优化器/调度，防止 Nerfstudio 把 30k 再当追加步数。

## 6. 评价、视频和网格

同一 B1 最终 checkpoint 完成全部 val/test：RGB 逐帧与等权均值；传感器 Z 的等权帧和合并像素 MAE/RMSE/AbsRel/P50/P95；有效参考、无效预测、alpha0 数量；30s 时间窗。报告 B1−B0 逐帧差值和改善/退化帧数，另保留原 s01 同源 48 帧端到端对照。无独立几何真值，不伪造 Chamfer/F-score。

Metric3D 单帧深度评价单独命名为先验深度诊断，不能与 DN 渲染深度或表面精度混列。使用相同完整参考分母，预测失败记失败，不用挑掉困难帧改善均值。法线只做方向/深度几何一致性诊断，不称传感器估计法线为独立 ground truth。

按采集时间生成全程 RGB reference/B0/B1 对照视频，沿用全部 1,846 主帧位姿；保存首中末、原 238/239、1409、最大深度误差帧、353s 动态段及末端倾斜。回传本地沿用已授权连接和已有存储位置；Git 提交小代表图和视频路径、大小、哈希。

B1 原始 Poisson 保持全部 1,478 train、目标 5M、depth9、原始不新增筛选的协议。随后 B0/B1 各跑一次 [固定支持约束导出](13_capture_supported_mesh.md)：train-only 深度/alpha 筛选、每视角最多25,907点、1cm真实体素融合、depth11、多视角支持裁剪。全部阈值见机器协议和该文档；不根据 val/test 扫参数。

新旧模型、点云与网格在完整参考分母及共同命中像素上分开评价，公开覆盖率和 5/10/20cm 内正确覆盖率。三处原固定局部保留，同样输出全场与全部留出网格指标；只改善局部但丢失其它空间不算已经改善全场。

## 7. 一次任务终点与交付

持续到推理、B1 完整训练、全部评价、全程视频、B1 原始网格、B0/B1 支持网格和报告推送完成。无需新增短实验/人工审阅；指标不好也交付。只有源身份不一致、必要访问失败或实际不能解决的运行/容量错误才记录真实阻塞，保留明确恢复命令；不能把未启动/后台运行写为完成。

报告至少包含实际配置与 diff、源/权重/数据/代码 SHA、全量推理记录、B0/B1 曲线、逐帧指标、时间段与同源对照、原始和支持网格比较、覆盖与伪前景面、阶段资源、所有大产物清单。同步导出已有 full 与 s01 的米制 c2w 小 JSON，允许计算旧/新共同帧刚性对齐后的姿态差异，明确这不是独立精度真值，不回头重求解轨迹。

结论按实际结果选择：法线替换改善/无明显改善/有取舍；导出筛选是否以覆盖损失换来误差下降；高斯减少发生在哪些步骤。可建议后续完整调度适配，但本轮不自动再开 240k、其它先验或多组实验。

原结果、原 bag 和大产物保留。新增代码进入 scripts，计划与实测分开；普通 Git 仅小报告、图、配置、清单及代码。检查 diff 后推送用户 main，不 force push，并核对远端状态与报告身份。
