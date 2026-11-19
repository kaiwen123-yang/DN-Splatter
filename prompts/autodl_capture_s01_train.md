# AutoDL Codex：capture-s01 法线、30k、评估与 Poisson

在现有 AutoDL 项目 /root/autodl-tmp/DN-Splatter 继续执行。先检查已有工作区、未提交修改、远端和本项目活跃进程，通过本项目子 shell 的 AutoDL 官方学术加速更新用户仓库 main。读取本提示词、AGENTS.md、docs/09_capture_training_plan.md、configs/capture_s01_train_protocol.json、reports/capture-geometry-review/2026-11-19/overview.md，以及 d72f56836028d7a7885f21f02522e4f7cc310429 的原几何报告。

几何审查已通过进入首个训练基线，带已知宽基线一致性局限。此前“geometry_prepared_pending_review，暂不运行法线/训练/Poisson”是上一阶段停止点，现已完成审查。本轮请实际完成下面整条链，不只写计划或停在环境检查、法线、短训练。已有持续项目授权和全 CPU/GPU 授权；无需为这些既定步骤反复确认。

## 固定输入

- 原始几何目录：.local/data/custom/capture-s01/20261119-s01-5hz-3982cb26
- 原 manifest SHA256：2cc67f7af6ad6444131529ee392879c064bc6a76394220784e7b4b79a039b278
- 来源几何提交：d72f56836028d7a7885f21f02522e4f7cc310429
- 241 帧，193 train / 24 val / 24 test。
- 正式输入 1280×720；米制 float32 彩色光轴 Z，0.3–6 m，空洞0；OpenGL c2w；436,361点训练帧初始化PLY。
- 固定已有划分、相机、米制尺度、图像、深度范围和点云，不根据留出残差改数据。
- 建议新 run：.local/runs/capture-s01-dn-30k-20261119-v1；已有同身份目录则按 checkpoint 和 manifest 续接，禁止重复启动或清空重来。

## 执行顺序

1. 只读检查现成环境和实际资源，复用 .local/conda/dn-splatter-bw、固定历史官方源码 811f5cdf342d92d12a9d9a4be185d374c430c184、Blackwell/Omnidata/checkpoint/Poisson 兼容补丁和已有权重。无具体错误不重装、不重编译，不重跑VR Room。允许全部已分配CPU/GPU，但不动其他项目、不设自动关机或任务deadline。
2. 先核对原几何 manifest 和全部成员。原目录保持封存；以reflink或普通复制建立新run的inputs/，原基础文件逐字节保持，旧manifest存为geometry_source_manifest.json。新增法线后另建训练输入清单并绑定两层哈希；不用可写硬链接污染原件。
3. 按docs/09补做原20对加反向的轻量跨视角诊断，固定640网格、位姿和尺度，最多40个有向比较；保留原始/双范围/源相机范围/局部平滑支持的完整分母与正负残差。输出164↔179、134↔149、224↔239代表图。不要按误差过滤测试帧，不把候选遮挡当真值，不引入“P95小于某值才允许训练”的新门槛。只有明确坐标、单位、帧映射等代码错误才修复后继续。
4. 在最终1280去畸变RGB副本上运行现有Omnidata low，输出241组PNG+NPY到normals_from_pretrain，记录权重和生成配置SHA。在源码目录运行以复用现有omnidata_ckpt。历史parser真正读取的是PNG；核对全体stem和尺寸。
5. 补一次新增法线路径的真实读取核验：load_normals=True、load_pcd_normals=True、load_3D_points=True。确认193/24/24逐帧对应、RGB/depth/normal同网格、单位尺度1，以及436361点的估计点法线有限、非零。不要再开多轮smoke训练。
6. 新建capture专属runner/evaluator/publication入口，复用已有公共逻辑。旧autodl_env.sh会覆盖VR路径，旧vr_room_run.py强制等待MuSHRoom/Faro数据；不能只改--data。新run独立锁、状态、日志、完成标记和产物身份。将最终可重放脚本提交到scripts/。
7. 直接正式训练累计30,000次更新，最终step=29999。dn-splatter、seed42、sensor_depth_lambda0.2、EdgeAwareLogL1、normal_lambda0.1、normal_supervision=mono、normal TV=True、two_d_gaussians=True、num_downscales0、camera_optimizer off、skip_point_metrics=True。normal-nerfstudio：depth单位1、auto_scale=False、scale_factor1、orientation/center none、downscale1、scene_scale15、normal_format opengl，camera_res_scale_factor1。其余历史算法默认保留。每5000步保存，保持必要中间checkpoint。
8. 若恢复，读取真实saved_step，remaining=30000-(saved_step+1)，恢复optimizer/scheduler且保持原30k调度；已到29999则直接评估，不重新训练。新场景不能加载VR Room checkpoint。
9. 同一最终checkpoint分别评估val24和test24，明确绑定step29999及SHA；不要让resume配置里的旧load_step误选模型。val需eval_setup(test_mode="val")等薄封装，不能编造ns-eval --split。按历史渲染helper要求从源码cwd传相对输出路径。交付逐帧与split均值PSNR/SSIM/LPIPS、固定参考支持上的深度MAE/RMSE/AbsRel/P50/P95、有效/无效预测分母、全部48帧本地渲染与固定代表图。深度叫同采集sensor consistency；不填Faro/Chamfer/F-score。
10. 从最终checkpoint运行历史gs-mesh dn，目标500万采样点、Poisson depth9、normal_maps，显式使用实际CPU预算。只采样193训练视角；保留原始mesh和带法线点云。默认exporter不会把输出限制在0.3–6m，不把远场模型表面称为传感深度验证过的几何。检查点/顶点/面、有限性、索引、米制范围和局部效果；不用val/test深度裁剪主网格。
11. 在reports/capture-s01-dn-30k/20261119-v1/提交小报告、协议/命令/环境/指标/图/完整大产物清单与哈希。大checkpoint、完整法线和渲染、PLY留服务器；原bag留H盘。保存原几何和VR结果，推送用户仓库main，不force push，并核对GitHub提交。

## 持续执行和回传

评估和网格同时记录 accumulation==0 的支持数/采样数：历史模型在零alpha处填入本帧最大Z，有限正深度并不等于有表面支持。保留原始指标和exporter行为，不静默加alpha筛选。网格采样seed固定42，实际点数以日志为准。正式训练首批检查loss/gradient和预测normal有限性；若实际出现数值错误，保留错误和排除分母，最小修复后继续，不预先改变算法。

长阶段使用项目tmux，完整记录PID/日志/退出码。不要因存在其他进程自动等待，也不得终止它们；只有实际资源不足才记具体阻塞。没有固定时限，不自动关机。常规实现选择直接完成；出现错误先做可定位的最小修复并继续。

完成后用中文回传：最终提交、阶段是否全完成、实际累计step、val/test指标及定义、法线/训练/评估/Poisson耗时与峰值资源、原始网格和局部预览、剩余误差与数据适用范围、大产物路径和SHA清单。没有执行的内容明确标未执行，不以漂亮预览代替指标，不声称无GT的绝对精度。

目标是得到这48秒自有场景的首个可审查完整基线。完成后保留数据等待效果复核；不自动扩展全包、其他场景、SLAM重构或超参数搜索。

