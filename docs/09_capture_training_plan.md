# capture-s01 首个 DN-Splatter 30k 基线执行计划

## 1. 审查决定与本轮终点

几何输入基于提交 d72f56836028d7a7885f21f02522e4f7cc310429。公开报告、源代码和代表图的复核支持进入首个自有场景训练；状态为 **accepted_for_first_training_baseline_with_known_limitations**。具体依据见 [几何复核](../reports/capture-geometry-review/2026-11-19/overview.md)。这不是独立几何精度认证，也不是训练已完成。

本轮一次持续任务完成：**保留几何输入 → 补充有界的跨视角诊断 → Omnidata 法线 → 实际训练读取核验 → 累计 30,000 次更新 → 验证/测试渲染与指标 → 原始 Poisson → 小报告推送**。完成后回传真实结果。此前 docs/08 的“停在几何准备、等审查”已达到终点；本轮不再按该历史停止条件暂停在法线、短训练或计划提交。只在数据身份、实现正确性或实际运行容量出现具体阻塞时停下并留下恢复入口。

这次保持历史 DN 方法；已有 VR Room 结果继续封存。新场景从自己的初始化点云重新优化高斯，不加载 VR Room 场景 checkpoint。不要为了宽基线残差重新做整个 bag 审计、SfM、IMU/VIO 路线，也不要临时开展消融或参数搜索。

## 2. 固定输入与身份

| 项目 | 本轮固定内容 |
| --- | --- |
| 几何提交 | d72f56836028d7a7885f21f02522e4f7cc310429 |
| 几何 run | 20261119-s01-5hz-3982cb26 |
| 原包 SHA256 | 3982cb269ea03ddf4234b45f74be27143eb66bcd807e4e471d0f3a9cd6b03b59 |
| 只读几何目录 | /root/autodl-tmp/DN-Splatter/.local/data/custom/capture-s01/20261119-s01-5hz-3982cb26 |
| 几何 manifest SHA256 | 2cc67f7af6ad6444131529ee392879c064bc6a76394220784e7b4b79a039b278 |
| 帧 / 划分 | 241；193 train / 24 val / 24 test；原清单逐项保留 |
| 正式训练网格 | 1280×720 去畸变 RGB 和同网格深度；不改成 640 |
| 深度 | float32 NPY；彩色光轴 Z；米；0.3–6 m；空洞为 0 |
| 固定尺度 | 1.464990352305482 m/SfM 单位，已经施加，不再乘一次 |
| 相机 | 已导出的 OpenGL c2w；零畸变；主点已做一次 +0.5 |
| 初始化 | 原 init_train.ply；436,361 个 XYZ/RGB 点；仅训练帧生成 |
| 建议训练 run ID | capture-s01-dn-30k-20261119-v1 |
| 训练 run 根目录 | .local/runs/capture-s01-dn-30k-20261119-v1 |
| 训练输入副本 | 上述 run 根目录下 inputs/ |
| 公开结果目录 | reports/capture-s01-dn-30k/20261119-v1/ |

run 目录已有内容时先核对身份与活跃进程，然后续接；不得删除重来或通过旧完成标记跳过阶段。若确实需要变更协议，另立有说明的版本，保留本基线。原 bag 不需要重新上传，本轮不访问 Windows H 盘。

实际训练深度正值覆盖在 241 帧上为 9.06%–22.69%，中位数 16.28%；约 95% 是 640 诊断网格未截断深度的覆盖率。稀疏传感深度、完整 RGB 和图像法线共同参与训练；不把深度空洞填成伪观测，也不把深度有效掩码当作 RGB 掩码。

## 3. 环境复用与专属 runner

继续使用 .local/conda/dn-splatter-bw，官方源码 .local/upstream/dn-splatter 固定 811f5cdf342d92d12a9d9a4be185d374c430c184。依据已有环境报告，预期 Python 3.10.22、Torch 2.7.1+cu128、Nerfstudio 1.1.3、gsplat 1.0.0、Open3D 0.18.0。先核对版本、实际导入路径、源码和既有补丁；没有具体不兼容就不重装环境、不重编译扩展、不重跑 VR Room。

复用 Omnidata checkpoint 与补丁，记录权重 SHA256；不重新下载可校验的现有资源。保留已有 Omnidata torch.load / backbone、gsplat 构建、Poisson 线程兼容修复。若遇到新错误，只做最小兼容修复并保存补丁，不能顺便切到官方新版或 AGS-Mesh。

先核对工作区、Git 远端和未提交改动；通过本项目子 shell 的 AutoDL 官方学术加速 fetch/pull/push，保留官方 HTTPS URL。只读检查实际 CPU quota/affinity、RAM、显存、磁盘和本项目进程。允许全部已分配 CPU 和整张 GPU；并行 worker 与内部线程总数按预算安排。只运行一个正式训练进程。不修改 base、系统 CUDA、驱动、全局配置，不终止其他项目。没有固定完成时限、自动关机或强杀 watchdog。

现有 scripts/autodl_env.sh、vr_room_run.py、vr_room_inputs.py、vr_room_evaluation.py、vr_room_publish.py 含 VR Room / MuSHRoom / Faro 的硬编码。**不能只替换 --data 后执行旧 runner。** 新增 capture 专用薄 runner / evaluator / 发布脚本，或先明确参数化公共部分，同时保持旧入口行为。尤其不能等待 VR 数据包、覆盖 VR 状态、加载其 checkpoint、把本场景映射成 long_capture/short_capture，或寻找 Faro GT。

新 runner 至少保存：独占运行锁、任务身份、阶段日志/PID/退出码、真实 checkpoint 路径与 step、正常退出/阻塞状态。只有输入 manifest、配置和源码身份匹配，才允许使用已有阶段产物恢复。长任务置于项目 tmux；不得在 ChatGPT 会话断开时重复启动。

## 4. 保留几何封存，建立训练输入副本

原 capture_dataset.validate() 对几何目录的全部成员做相等检查。直接在原目录生成法线会改变其成员集合。因此：

1. 校验原 manifest 本身的已知 SHA256，再核对全部成员、字节数与 SHA256。源目录仍应是原 488 个文件；发现不一致先定位原因。
2. 在本 run 的 inputs.building/ 建立复制或 reflink 副本，再原子改名。优先 reflink；不支持时普通复制足够。不要用可写硬链接让训练端改动传播到原件。
3. RGB、深度、transforms.json、splits.json、source_frames.json、init_train.ply、geometry.json 的原始内容保持逐字节相同。旧 manifest 作为 geometry_source_manifest.json 保存；新增 training_inputs.json 和最终训练 manifest，明确层次。
4. 在输入副本生成法线。最终训练 manifest 绑定：原几何 manifest、完整基础文件、全部法线文件、权重 SHA256、生成命令、源码/补丁及生成环境。训练开始后固定这个输入版本。
5. 原几何目录的 manifest 不覆盖、不增删成员；报告分别引用“几何输入指纹”和“含法线训练输入指纹”。

训练副本的基础文件按 geometry_source_manifest.json 原有成员逐项核对，新增成员按新训练 manifest 核对。副本不能再套用旧 capture_dataset.validate() 的全目录成员相等检查；旧验证器仍可用于不变的原几何目录。

训练输入副本不是新的位姿或尺度估计；不要利用副本机会重采样深度、过滤难帧、改变初始化或刷新划分。

## 5. 同步完成的跨视角补充诊断

目的：解释已观察的不一致并为训练结果提供上下文，不把诊断变成另一轮数据优化。复用原 640 网格、固定 poses.json 和原 20 对视角；增加反方向，共最多 40 个有向比较。只读几何中间目录，报告另存于本 run，**不覆写旧 cross_view.json**。

报告必须区分：

- 原定义：目标实测 Z 在 0.3–6 m，有源投影和目标观测；不保证同一可见表面。
- 双目标轴范围：目标实测 Z 和变换后的源 Z 都在 0.3–6 m。
- 原源相机范围也在 0.3–6 m 的支持子集，避免把双目标轴范围误称为训练输入完全相同的范围。
- 双方各自深度平滑、非 D2C 竞争区、远离空洞的支持子集。掩码只能由各自深度的局部性质生成，不使用被评估的跨帧残差来挑选“小误差像素”。
- 正、负分歧候选分别计数，阈值沿用 max(0.05 m, 0.03 × target_Z)，命名为候选，不能声称是已知遮挡真值。

z-buffer 应先接纳全部有效源点决定赢家，再携带赢家的原始源像素、源 Z、边界/竞争区标签进行分组。不要先删掉前景再把背景暴露出来。可沿用旧 edges() 断边阈值，在原 640 网格做 2 像素带状扩张；空洞邻域和竞争区的膨胀半径在运行前固定、写入报告。每个子集都报告总支持数、保留比例、signed/absolute P50、P90、P95 和正负分歧比例。不要把残差阈值过滤后的 visible_abs_m 当作独立精度证据。

三组代表图足够：164↔179、134↔149、224↔239；包括源/目标 RGB、目标 Z、投影 Z、有符号差和各类掩码。采用固定尺度和颜色范围，说明饱和上限。只有图像明确暴露帧映射、单位、坐标或网格实现错误才修正该错误；不得新设“P95 必须低于几厘米”门槛阻塞正式基线。

已有 val/test 深度参与过这项几何诊断，后续报告应披露；不把 test 称为完全未查看的盲测。不要按测试结果选 checkpoint、调损失、删难帧或裁剪网格。

## 6. Omnidata 与一次必要的真实读取核验

在最终 1280×720 去畸变 RGB 输入副本上使用历史版本的 Omnidata low 路径，内部 384×384 推理并还原输出分辨率。normal-format 为 opengl。所有 241 帧生成对应法线，以满足历史 parser 对全帧文件列表排序后再按 split 取索引的方式。

已对照历史源码的入口如下；在固定源码目录运行，DN_SCENE_DIR 必须是新 run 的训练输入副本，解释器和缓存来自既有专属环境：

    python dn_splatter/scripts/normals_from_pretrain.py \
      --data-dir "$DN_SCENE_DIR" \
      --save-path "$DN_SCENE_DIR/normals_from_pretrain" \
      --resolution low --model-type omnidata

该脚本从源码目录的 omnidata_ckpt/omnidata_dpt_normal_v2.ckpt 取权重；记录真实文件身份，避免无意回退到不同模型。

历史 NormalNerfstudio 从 normals_from_pretrain/*.png 取文件，不能仅给 transforms.json 增加 normal_file_path。零填充 frame_00000 至 frame_00240 的 stem 必须与 RGB 逐一对应；PNG 和 NPY 都保留在本地并入清单。生成全部帧先验并不意味着用 241 帧训练：训练 dataset 必须仍只有原 193 帧。

只补充本轮原验证没有覆盖的项目：

- 241 张 PNG 和对应 NPY 的数量、stem、尺寸、范围和有限性；编码可视化与向量数据的定义分别记录。
- 使用正式设置的 NormalNerfstudio、GDataset 和 DNSplatterDataManager（配置类 DNSplatterManagerConfig）读取 train/val/test，确认 193/24/24；RGB、sensor_depth、normal、K、c2w 同网格；不发生自动缩放或在线去畸变错位。
- load_normals=True、load_pcd_normals=True、load_3D_points=True。原 PLY 只有 XYZ/RGB；历史 parser 会以 radius=0.1 m、max_nn=30 估计并归一化点云法线。核对 436,361 个点与对应法线的 shape、有限值、非零范数；不改用随机点。
- normal 文件按 stem 验证准确对应关系，不能只验证“数量相等”。
- 记录实际 dataparser_scale=1、单位变换、scene_scale=15。scene_scale 是包围盒大小，不是新的米制缩放。
- 原几何读回已经覆盖全部基础输入，不重复建立另一组小训练。此读取核验通过后直接启动正式 30k，检查正式训练首批损失/梯度有限性，并记录预测 normal 的非有限像素数；不要另开 100/1k/3k 多轮 smoke。历史渲染法线直接除以范数，只有实测出现零贡献导致非有限值时才做必要的局部修复并记录，不能预先改变损失。

## 7. 固定训练参数和恢复

协议机读入口为 [capture_s01_train_protocol.json](../configs/capture_s01_train_protocol.json)。它是执行协议，不可直接当作 Nerfstudio config 读取。

| 设置 | 值 |
| --- | --- |
| method | dn-splatter，固定历史实现 |
| seed / 总更新数 | 42 / 30,000 |
| 最终 checkpoint step | 29,999 |
| 训练相机优化 | off，沿用历史 DN 默认 |
| pipeline.skip_point_metrics | True，没有本场景独立参考点云 |
| use_depth_loss / depth_loss_type | True / EdgeAwareLogL1 |
| sensor_depth_lambda / mono_depth_lambda | 0.2 / 0.0 |
| use_normal_loss / normal_supervision | True / mono |
| normal_lambda / use_normal_tv_loss | 0.1 / True |
| two_d_gaussians | True |
| num_downscales | 0，历史 DN 默认也是 0 |
| parser downscale_factor / camera_res_scale_factor | 1 / 1 |
| depth_unit_scale_factor | 1.0 |
| auto_scale_poses / scale_factor | False / 1.0 |
| orientation_method / center_method | none / none |
| scene_scale / normal_format | 15 / opengl |
| 保存间隔 / 仅最新 | 5,000 / False |
| 训练可视化 | tensorboard，完成即退出 viewer |

使用 normal-nerfstudio parser 及明确的 train_filenames/val_filenames/test_filenames；不要使用 mushroom parser 的 load-depths、depth-mode、normals-from 或 Faro 参数。正式命令及生成后的完整 config.yml 都保存；核对全部固定值后才进入长训练。未列出的模型优化器、学习率和高斯增密设置保持该历史 DN 默认及已有兼容修复，不能从 VR 的 MuSHRoom 数据参数整段照搬。

累计更新数通过 checkpoint 内容核对。Nerfstudio 1.1.3 恢复时本次 max_num_iterations 使用：

    remaining = 30000 - (saved_step + 1)

恢复同一新场景的模型、optimizers、schedulers，保留原 30k 调度终点；不得在中途 checkpoint 后再追加完整 30k，也不得因断线从头训练。若已达到 29,999，直接进入评估/网格。

复用现有 checkpoint 的局部 torch.load 兼容逻辑。新 evaluator 若直接调用 Python API，需要显式复用对应局部上下文，不能假设 vr_room_checkpoint.py 支持任意脚本入口。

## 8. 验证、测试与可看的效果

用同一个最终 step=29,999，分别输出 val 24 帧和 test 24 帧。训练期间可以记录 val 指标，但不按 test 筛选模型。本轮主模型始终是固定最终步。

Nerfstudio 1.1.3 的 ns-eval 默认 test。要评估 val，薄封装应调用 eval_setup(..., test_mode="val")；不要编造 --split 参数。eval_setup 可能继承恢复训练 config 中旧 load_step，必须在评估副本或 update_config_callback 中显式设 load_step=29999。它在 callback 后重新执行 load_dir=config.get_checkpoint_dir()，因此仅在 callback 内改 load_dir 不够；应核对 get_checkpoint_dir() 就是最终 checkpoint 目录，或通过合法输出目录/实验名/时间戳配置形成准确路径，再断言返回 checkpoint 路径、文件 SHA 和 step。原训练 config.yml 不覆盖，网格也使用绑定最终 checkpoint 的配置。

历史渲染 helper 会将 cwd 与输出字符串拼接；从固定源码 cwd 传相对渲染路径，落盘后核对目录和帧集合。可使用同一薄 evaluator 顺序加载 val/test，完成一个 split 后释放其 pipeline，避免无意义的双模型占用。

分别保存：

1. RGB：PSNR、SSIM、LPIPS，逐帧及每个 split 的等权帧均值；记录实际 torchmetrics / LPIPS 网络与是否 normalize，原始库输出保留。
2. 传感深度一致性：以原冻结的 0.3–6 m 正值深度像素作为参考支持，不填洞，不按预测误差剔除像素。记录参考像素分母、预测无效数与覆盖比例；非有限或非正预测不能静默丢弃。若有无效，明确报告该帧/指标无效，另列有限预测条件结果及保留率。同时记录参考支持内 accumulation==0 的像素数；历史模型把 alpha=0 处的 Z 填成本帧最大渲染深度，所以“有限且正”不保证有模型表面支持。保留原始定义，不静默按 alpha 筛选指标。
3. 在合法支持上报告 MAE、RMSE、AbsRel、绝对误差 P50/P95。RMSE 为 sqrt(mean((pred_Z-ref_Z)^2))；AbsRel 为 mean(abs(pred_Z-ref_Z)/ref_Z)。明确等权帧均值与全像素聚合，二者不可混用。不得用历史 normal/depth 指标中有问题的编码或定义冒充标准量。
4. 同一输入深度作为训练先验/评价参照，且来自同次采集，因此名称为 sensor consistency；没有外部 GT。不给本场景填 Faro F-score、Chamfer 或“绝对几何精度”。
5. 保存全部 48 帧 RGB、预测 Z 的 float32 原值和法线渲染在服务器。公开少量固定代表图：每个 split 的首/中/末帧，外加 149、179、239 所在 split 的诊断视角，避免只选最好画面。统一深度/残差色标，注明空洞与饱和。
6. 原始 normal 伪先验不是几何 GT，法线主要做可视化；若输出角度一致性，须从编码解码、归一化、裁剪 dot 后 acos，并明确仅为伪先验一致性。无需为此增加新训练损失。

可生成沿已有相机轨迹的短视频，优先训练/留出原位姿序列，避免把未观测的大幅自由飞行当可靠新视角能力。报告桌面/箱体/柱面是否连贯、椅腿是否断裂、重复表面/漂浮点、快速转向和远场的表现，不能仅交付均值。

## 9. Poisson 网格与适用范围

使用同一个最终 checkpoint 的历史 gs-mesh dn：total_points=5,000,000、poisson_depth=9，normal_method=normal_maps；沿用 filter_edges_from_depth_maps=False、outlier_removal=False。已有补丁将 Poisson n_threads 绑定实际 DN_CPU_BUDGET。固定网格采样随机种子为42并保存实际采样点数；若全部193视角有足够候选，历史整数分配可采样5,000,051点，最终以日志为准。

此 exporter 从 train_dataset 的 193 个视角采样模型渲染的深度与法线，不需要 Faro，也不使用 val/test 深度。核对实际访问的相机 ID 与冻结训练集一致，正常复用历史 exporter，不调用 MuSHRoom 专用裁剪/评价脚本。

**输入深度限定 0.3–6 m，不代表默认 Poisson 只输出该范围。** exporter 会使用模型渲染的其他非零深度，包括缺少传感器深度约束的远景。它默认还有密度最低 1% 顶点裁剪。保留这一历史原始导出作为基线，报告采样/过滤数、渲染深度范围、顶点与面数、世界坐标包围盒、有限性、连通组件和代表性局部预览。

优先交付原始 Poisson 与采样 PLY，核对采样法线有限且非零，并记录采样中 accumulation==0 的比例，不预先添加 alpha 过滤改变基线。若原始 exporter 因非有限法线等明确数值问题失败，保留错误与原始计数，仅修复定义良好的数值有效性条件，另存补丁和排除分母。用全局视图及桌面/柱体/细椅腿至少三个局部视图审查。Poisson 补出的闭合表面不自动等于被传感器观测过的真实表面。

若原始网格有明显远场浮点而需要可用性辅助产物，可在保留原件之后增加“仅训练观测支持”的独立裁剪版本；阈值、训练支持掩码、实际保留率和方法另存。不得用 val/test 深度或手调剔除困难区域后，把裁剪结果冒充原始方法/独立精度证据。本轮不自动增加第二组训练或改网格算法。

## 10. 完成条件、回传与下阶段

在 reports/capture-s01-dn-30k/20261119-v1/ 发布小型、可核查报告，至少包括：

- overview.md：是否完成、主要效果、局限、后续决定。
- run_manifest.json、config.yml、commands.sh、environment.json。
- inputs.json、normal_prior.json、reader_check.json。
- cross_view_diagnostic.json 与三组代表预览。
- training.json：实际累计 30k、step、Gaussian 数量、损失与资源/耗时。
- metrics_val.json、metrics_test.json、逐帧指标与评价定义。
- mesh_validation.json、网格全局及局部图、若有裁剪的独立说明。
- artifact_manifest.json：checkpoint、法线、完整渲染、PLY、日志的路径/大小/SHA256。
- latest 指针：实际完成状态，不把计划当运行结果。

报告与代表图目标 20 MiB、普通 Git 总新增不超过既有约定的 50 MiB；大文件留本项目目录。保留原几何报告与 VR Room 报告，推送到用户仓库 main，不 force push。GitHub 和服务器以提交 SHA 核对。Windows 本机由已有 Git 工作流同步小报告，不再搬运原包。

真正完成需要：法线/读取核验通过、最终累计 30k、48 留出帧结果、有效非空 Poisson、效果预览、产物清单与可重放代码已推送。只完成训练时不能写全流程完成。

首段结果回传后再决定：保留1280基线并扩大采集范围，或针对已证实的几何/噪声问题建立单独改进实验。全包约369秒尚未完成位姿与质量验证；不能从这48秒推断全部数据和未观测表面已可用。扩大时优先统一地图/尺度、重叠片段和独立评估轨迹，避免把多段独立 SfM 坐标直接拼在一起。

## 11. 实现依据

- [本次几何结果](../reports/capture-geometry/20261119-s01-5hz-3982cb26/overview.md)、[读取契约](../reports/capture-geometry/20261119-s01-5hz-3982cb26/dataset_manifest.json)、[生成脚本](../scripts/capture_dataset.py)。
- [历史 NormalNerfstudio](https://github.com/maturk/dn-splatter/blob/811f5cdf342d92d12a9d9a4be185d374c430c184/dn_splatter/data/normal_nerfstudio.py)、[DN 配置](https://github.com/maturk/dn-splatter/blob/811f5cdf342d92d12a9d9a4be185d374c430c184/dn_splatter/dn_config.py)、[DN 模型](https://github.com/maturk/dn-splatter/blob/811f5cdf342d92d12a9d9a4be185d374c430c184/dn_splatter/dn_model.py)。
- [历史法线脚本](https://github.com/maturk/dn-splatter/blob/811f5cdf342d92d12a9d9a4be185d374c430c184/dn_splatter/scripts/normals_from_pretrain.py)、[网格导出](https://github.com/maturk/dn-splatter/blob/811f5cdf342d92d12a9d9a4be185d374c430c184/dn_splatter/export_mesh.py)。
- [Nerfstudio 1.1.3 eval_utils](https://github.com/nerfstudio-project/nerfstudio/blob/v1.1.3/nerfstudio/utils/eval_utils.py)、[COLMAP 坐标/内参约定](https://colmap.github.io/faq.html)、[论文项目页](https://maturk.github.io/dn-splatter/)。

