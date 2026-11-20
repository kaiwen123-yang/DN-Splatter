# AutoDL Codex：capture-full 全程30k、完整评估与原始Poisson

> 历史阶段已完成：完整结果固定在 `e26ae72a75997dcfd3a907aa26b6c5c0677597a5`。当前执行 `docs/11_capture_metric3d_plan.md` 与 `prompts/autodl_capture_metric3d.md`，不重新启动本文的提取/建图/训练。以下保留原执行协议。

请在现有 /root/autodl-tmp/DN-Splatter 中实际持续执行到最终交付。用户要求停止小规模验证与零碎修复，直接完成整段采集的第一次完整测试，再据完整结果决定改进。s01已完成，结果commit为ee3e87bf8a63486ed96df1c0d37b6a72d30fca88；本轮不是重跑s01。

先核对远端、未提交修改与本项目活跃进程，用项目子shell的AutoDL官方学术加速更新main。阅读AGENTS.md、docs/10_capture_full_test_plan.md、configs/capture_full_test_protocol.json、reports/capture-s01-review/2026-11-19/overview.md；遵循本轮入口。旧阶段的geometry_pending_review或“不自动扩全包”不再约束本轮，禁止加回中间审阅停点。

## 身份与复用

- 全程原件SHA256：3982cb269ea03ddf4234b45f74be27143eb66bcd807e4e471d0f3a9cd6b03b59。
- 输入紧凑包：.local/capture-geometry/20261119-full-5hz-3982cb26/transfer/。
- 全程数据：.local/data/custom/capture-full/20261119-full-5hz-3982cb26/。
- 独立run：.local/runs/capture-full-dn-30k-20261119-v1/。
- 小报告：reports/capture-full-dn-30k/20261119-v1/。
- 整段RGB跨度368.874747秒，原动态传感流约369.35秒；5Hz覆盖首尾，预计最多1846主帧，1478/184/184，实际清单为准。
- 固定历史源码811f5cdf342d92d12a9d9a4be185d374c430c184、已有项目Conda及Blackwell/Omnidata/Poisson兼容补丁；原始s01和VR数据、模型及报告保持封存。

只读检查现有资源与输入哈希，复用已工作的安装和权重；无具体错误不重建环境、不再跑独立梯度探针/短训练。允许全部已分配CPU/GPU，单个正式训练进程，防止线程嵌套；不终止其他进程、不自动关机、不设置训练deadline。已有同身份run时按manifest/锁/PID/checkpoint续接，防止本地协调器和服务器双重启动。

## 一次性实现与执行

1. 先核对全程包逐文件SHA、真实时间/源ID、唯一RGB-D和预冻结split。原候选索引mod10固定，缺失后不能重排；旧s01前241成员的val/test身份按源时间/哈希保留。无包时与正在执行的本地协调任务接续传输，不能拿s01包冒充全程包。
2. 把现有capture_segment/freezer、rectify、SfM/localize/metric/dataset与s01训练链作必要参数化或full薄封装，保存代码到scripts。重点移除旧241/193/24/436361/scene_scale15与s01路径硬编码，动态读取实际available集合、初始化点数和训练侧包围盒。不得只改--data后误复用VR/S01完成标记。
3. 全程离线去畸变RGB和D2C，使用既有真标定、毫米Z、depth→color变换、最近表面z-buffer、半像素约定。保留原值和未截深度，正式1280×720彩色轴Z为0.3–6m、float32米、孔洞0；640只用于已有几何汇总和点云初始化，不擅自把正式分辨率减半。
4. 对全部主train构建统一地图：现有CUDA SIFT、固定内参，默认全训练图exhaustive匹配（可复用先前顺序匹配结果），统一incremental mapper/全局BA。如果现场已有可直接用的COLMAP全局检索，可一次性选顺序+全局候选并记录；不新增检索工程。不把几个独立短片段简单拼接。注册失败才做一轮断口两端±2秒原生RGB辅助桥接，通过本地协调者补取；排除主val/test源ID，辅助图只用于SfM/BA，不入DN采样、尺度和初始化。冻结地图后留出只PnP，必要时扩大固定地图候选一次，不能留出BA。
5. 全程只用已注册主train的有效深度拟合统一米制尺度；不要复用s01的1.46499比例。点云仍是主train的640网格、0.3–6m、原遮挡/边缘处理、每轴4像素、0.03m体素，实际点数记入清单。scene_scale按docs/10用train相机和点云范围求得。所有地图分量、未定位帧、时间缺口保留；个别缺失不阻止可训练地图继续全链。无独立真值，不把残差达到某厘米阈值作为训练门槛。
6. 在独立训练输入副本生成或按像素/权重/配置身份复用Omnidata low法线。固定权重SHA38db3f7d952f48f900f0b32d6cce6749e41641045ebc5713cf29618f54aff3bb，真实读取normals_from_pretrain中的PNG，输出尺寸1280×720。按实际available帧和split核对RGB/Z/normal/K/c2w及法线有限性；正常输入检查随正式准备一次完成，不开smoke训练。
7. 若实际存在无0.3–6m传感器参考的帧，保留其RGB/normal/pose与全零Z。去掉旧coverage>0断言。EdgeAwareLogL1的x/y两个空归约分别取可微零，非空mean和所有权重原样保留；内部DepthMetrics无参考时不写NaN，记录触发帧和更新计数。不要用nan_to_num掩盖其它错误，不伪造深度，不把深度mask作用到全RGB。若没有实际空归约，不因假设风险改损失实现。
8. 正式从全程初始化训练累计30000更新，最终step29999。沿用s01全部核心方法参数：seed42，EdgeAwareLogL1，sensor depth 0.2，mono depth 0，mono normal 0.1，normal TV=True，two_d_gaussians=True，num_downscales0，camera optimizer off，skip_point_metrics=True。normal-nerfstudio：单位1、scale1、auto_scaleFalse、orientation/center none、downscale1、OpenGL，load_3D_points/load_normals/load_pcd_normals=True。15k停止split、30k调度保持，不自动变为120k/240k。记录实际train图数对应的增密候选步；pose-only辅助图不得算入DN训练数。
9. 每5000步保存；首次实际formal batch核对loss/gradient有限性，随后持续到终点。恢复按真实saved_step计算remaining=30000-(saved_step+1)，保留optimizer/scheduler及原30k调度。禁止从s01或VR checkpoint初始化全程场景；不得因断线再重复完整30k。
10. 同一个最终checkpoint与SHA分别评估所有可定位val/test，明确test_mode=val/test与最终load_step，避免旧resume配置误加载中间点。RGB逐帧与均值PSNR/SSIM/LPIPS；0.3–6m传感器Z给等权有效帧与合并参考像素的MAE/RMSE/AbsRel/P50/P95，公开requested/posed/evaluated、无参考/无效预测/alpha0分母。无参考帧的depth指标为null，RGB仍统计；整个split为空也如实给null而不崩溃。按30秒窗展开覆盖与表现。旧s01的48张留出仅在原身份和输入一致且仍留出时另作同帧端到端对照。
11. 渲染完整主帧采集轨迹的参考RGB/预测RGB对照视频，按原时间播放，标时间戳与split；未定位处标缺口，不插值造位姿。固定展示首中末、最差帧、s01的238/239源帧、约353秒动态段和末端侧倾。全部原尺寸结果留服务器，GitHub放小预览，不能只挑好看图。
12. 从同一最终checkpoint导出原始DepthAndNormalMapsPoisson：所有已注册主train视角，目标500万点，depth9，normal_maps，seed42，n_threads为实际CPU预算，不新加深度/alpha/边缘/离群过滤。保留点云和原始mesh；记录实际样点、每视角样点、超0.3–6m点数、alpha0/无效法线、顶点面数、分量及最大分量面占比、物理范围。完整全景加桌沿/柱面/椅腿局部，以及同区域模型深度/法线→点云→网格对照。原始基线完成前不做网格参数扫点。
13. 全部结果形成小报告，保存环境、实际配置、完整命令、输入manifest、源码/补丁身份、阶段耗时和资源、逐帧指标、覆盖缺口、视频/模型/mesh路径大小SHA。源数据和大产物不进普通Git。只stage本任务文件，经审阅diff后push用户main，不force push，核对远端commit与报告、状态指针一致。

## 终点和真实阻塞

正式流水线不以小图质量、某个未注册帧或某项指标不够好为理由退回新一轮等待审查。有可训练统一地图就完成正式训练、可评估留出和原始网格。存在注册缺口时最终写completed_with_coverage_gaps并公开所有分母；不能称全包成功。没有可训练地图、源身份矛盾、必要访问失效或实际资源不足且无法等价处理时，保留已完成阶段和明确恢复命令。

最终中文回传必须包含：真实完成状态与最终commit、全程覆盖与失败时间段、累计更新/最终step/checkpointSHA/高斯数、val/test指标与同帧对照、完整轨迹视频和固定坏例、原始网格全景与薄结构局部、阶段耗时资源、大产物清单，以及基于完整效果的下一步判断。没有执行的内容写未执行，不把“后台已启动”写成完成。本轮不自动启动下一项消融、其他场景或全文复现；先交付这一次全程结果。
