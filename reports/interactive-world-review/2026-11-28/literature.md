# 文献与借鉴范围：以灵波几何和可修订记忆为主线

来源校验截至 **2026-11-28**。本文为文献、官方代码及公开数据结构审计，服务于 [下一轮实施计划](../../../docs/17_interactive_world_plan.md)。读取论文、检查源码、列举压缩包和读取 metadata 都是本次研究活动，**不表示已完成新的训练、物理仿真、动态预测或真实机器人实验**。

本轮主线是：以 LingBot-Depth 提供有测量条件的几何候选，以 LingBot-Map / Map 2 的时序上下文与记忆修订思想组织证据，再将被接受的几何转换成能接受动作、可被新观测纠正的局部世界。其他论文补足候选校准、薄结构、碰撞、动力学辨识和数据评价；不把所有模型串起来当作算法贡献。

## 1. 日期与公开成熟度总表

下表将**预印本日期、版本日期、官方发布日、会议年份分开**。只有核实了原始来源才填写日期；会议年份不推导为预印本日期。“可作为复现起点”表示有实际源码或相应资源，不保证已经在本项目环境运行。

| # | 原始工作与日期依据 | 官方代码与成熟度 | 本轮用途 |
| --- | --- | --- | --- |
| 1 | [Masked Depth Modeling for Spatial Perception / LingBot-Depth](https://arxiv.org/abs/2601.17895)：v1 2026-01-25 | [推理、权重接口和特征接口](https://github.com/Robbyant/lingbot-depth)已提供；本项目已有冻结推理回执，不能据此称已复现完整预训练 | A 的条件遮挡、候选；C 的 R2/R3；E 的 RGB-D 归因 |
| 2 | [LingBot-Map](https://arxiv.org/abs/2604.14141)：v1 2026-04-15，v3 2026-09-08 | [官方仓库](https://github.com/Robbyant/lingbot-map)提供模型、推理和长序列工具；仓库另列 ECCV 2026 论文版本 | B 的流式几何、长期上下文和配对初值 |
| 3 | [Reconsolidatable Streaming 3D Reconstruction / LingBot-Map 2](https://raw.githubusercontent.com/Robbyant/lingbot-map-v2/078b54485830c74ab1d3797667073e3e8353f3b9/docs/lingbot-map-v2_paper.pdf)：核验官方技术报告；[固定 README](https://github.com/Robbyant/lingbot-map-v2/blob/078b54485830c74ab1d3797667073e3e8353f3b9/README.md)记录 2026-11-21 发布因果推理、11-23 增加 benchmark | 公开的是 **without GCR 的 geometric streaming base**；该 README 称权重仓库需有访问权限，本项目已实际取得配套权重并记录身份 | B 的实际推理；D/E 的显式可修订证据记忆启示 |
| 4 | [Infinite Worlds with Versatile Interactions / LingBot-World 2](https://arxiv.org/abs/2607.07534)：v1 2026-07-08；官方代码发布日 07-09，补充模型发布日 09-10 | [官方仓库](https://github.com/Robbyant/lingbot-world-v2)有推理及多个模型变体，README 明确不计划发布其部署代码 | 借鉴因果动作条件、局部事件标注和长时稳定性；本轮不训练 |
| 5 | [DN-Splatter](https://arxiv.org/abs/2403.17822)：v1 2024-03-26，v3 2024-11-07；WACV 2025 | [官方实现](https://github.com/maturk/dn-splatter)完整，本项目已有实际基线 | C 的损失、正则化与网格出口基线 |
| 6 | [AGS-Mesh](https://arxiv.org/abs/2411.19271)：v1 2024-11-28，v2 2024-12-16 | [项目页](https://xuqianren.github.io/ags_mesh_website/)和 DN-Splatter 仓库中的自适应深度/法线实现可读 | A/C 的先验过滤与 R5 平滑归因 |
| 7 | [ConfidentSplat](https://arxiv.org/abs/2509.16863)：v1 2025-09-21 | 已读论文方法；本次未核实到可运行的官方完整代码，不列为现成依赖 | A 的融合查重和拒绝机制对照 |
| 8 | [Improving Densification in 3D Gaussian Splatting](https://arxiv.org/abs/2508.12313)：v1 2025-08-17；[CVPR Findings 2026 条目](https://openaccess.thecvf.com/content/CVPR2026F/html/Deng_Improving_Densification_in_3D_Gaussian_Splatting_for_High-Fidelity_Rendering_CVPRF_2026_paper.html) | [Improved-GS](https://github.com/XiaoBin2001/Improved-GS)有实际增长、分裂、裁剪代码；Findings 与主会区分 | C/R4 的增长预算和边缘分裂基线 |
| 9 | [SurfFill](https://arxiv.org/abs/2512.03010)：v1 2025-12-02，v2 2026-05-28 | [官方项目](https://linus-franke.com/surffill/)与[代码](https://github.com/qy21gafy/SurfFill)可用，输入假设是 LiDAR 扫描及图像 | C/R4 的薄结构增长与区域正则化 |
| 10 | [MASt3R-SLAM](https://arxiv.org/abs/2412.12392)：v1 2024-12-16，v2 2025-06-02；CVPR 2025 | [官方实现](https://github.com/rmurai0610/MASt3R-SLAM)含匹配、跟踪和优化 | B 的新匹配证据；C/R6 的局部位姿候选 |
| 11 | [RE3Sim](https://arxiv.org/abs/2502.08645)：v1 2025-02-12，v4 2026-07-10；官方标 ICRA 2026 | [当前官方仓库](https://github.com/Sage-PhysicalAI/Re3Sim)有 IsaacLab/GS 管线、资源、示例数据；原 InternRobotics 地址重定向 | D 的外观/碰撞分层，E 的轨迹生成 |
| 12 | [TwinAligner](https://arxiv.org/abs/2512.19390)：v1 2025-12-22 | [官方代码](https://github.com/TwinAligner/TwinAligner)已有视觉重建、视角对齐等；**Dynamic Real2Sim 仍为 Coming soon** | D/E 借鉴分阶段系统辨识，不能承诺直接运行该辨识模块 |
| 13 | [PhysTwin](https://arxiv.org/abs/2503.17973)：v1 2025-03-23；ICCV 2025 | [官方代码](https://github.com/Jianghanxiao/PhysTwin)、真实 RGB-D 数据、处理数据与 checkpoint 已提供 | E 的 RGB-D + 控制点动态验证支线 |
| 14 | [Real-to-Sim Robot Policy Evaluation](https://arxiv.org/abs/2511.04665)：v1 2025-11-06，v2 2025-11-10；官方项目标 ICRA 2026 | [官方代码](https://github.com/kywind/real2sim-eval)有仿真、资产、motion replay、真实示教和权重；自定义 PhysTwin 训练仍引用原项目 | D/E 的接口、回放、配对初态和真实有效性评价 |
| 15 | [iTACO](https://arxiv.org/abs/2506.08334)：v1 2025-06-10，**v2 2025-09-25**，v3 2025-11-17；v3 标 3DV 2026 camera-ready | [官方代码](https://github.com/3dlg-hcvc/video2articulation)有粗关节预测、联合细化、网格和交互渲染 | 以后扩展门/抽屉；本轮不把它加入默认刚体路径 |
| 16 | [Real-is-Sim](https://arxiv.org/html/2504.03597v2)：v2 2025-07-02；[官方项目页](https://real-is-sim.github.io/)标 ICRA 2026 | 本次核查论文与项目页，不承诺其全栈部署代码已可直接运行 | D/E 的真实观测持续校正；严格分开在线状态同步与开环物理预测 |
| 17 | [MapAnything](https://arxiv.org/html/2509.13414v2)：v2 2025-09-18 | [官方代码](https://github.com/facebookresearch/map-anything)提供可选 K、位姿、深度等几何输入 | B 的可选米制条件对照，不替代灵波主线 |
| 18 | [Boba](https://jianxiapyh.github.io/Boba-project-page/)：官方标 ECCV 2026，核查日期 2026-11-28；未将会议年伪造为预印本日期 | 项目页提供论文与源码入口；本次未在目标机器验证其完整栈 | 共享静态资产、独立动态状态及批量 rollout 的系统启示；主线验证后再考虑迁移 |

## 2. 灵波主线：从模型机制到本项目可验证变化

### 2.1 LingBot-Depth：学习有测量条件的空间表征

[MDM 正文 §2.1–2.2](https://arxiv.org/html/2601.17895v1)将 RGB 和深度分别 patch embedding，再让完整 RGB 与未被遮挡的深度 token 共同参与编码。它的遮挡具有传感器语义：全失效 depth patch 必遮，混合有效/无效 patch 以较高概率遮挡，训练示例概率为 0.75，再补随机有效 patch 达到 60%–90% 总遮挡率。ConvStack 从编码后的上下文表征恢复稠密几何。

这一机制直接支持 A 线的三个实施选择：按原生量测足迹、连续块和真实几何边界构造隐藏；在所有候选生成前隐藏测量及其投影后代；把可见传感器几何保留为条件。它不能支持“所有空白画布像素都是传感器失败”或“填满率就是正确率”。本项目的二维 NN 是必要基线，因为重采样空隙与真实大孔洞具有不同难度。

**实际复用范围：**固定 [源码 commit](https://github.com/Robbyant/lingbot-depth/tree/f3a237e434ae987bc38281476d6cfb5df3e4d739) 和 [v0.5 权重](https://huggingface.co/robbyant/lingbot-depth-pretrain-vitl-14-v0.5)，导出冻结候选及表征；本轮的轻量风险头属于新增实现。mask/置信输出不可直接解释为“误差小于 10 cm 的概率”。单帧网络的平稳视频效果，也不自动提供严格跨帧一致性或对象身份。

**关联：**A1–A4；C 的 R2 对比 R1 测候选贡献，R3 对比 R2 测校准与拒绝贡献；E 的 RGB-D 支线检验状态修订后的未来预测。

### 2.2 LingBot-Map：不同记忆承担不同几何职责

[Map 正文](https://arxiv.org/html/2604.14141v1)的 Geometric Context Transformer 组织 anchor context、pose-reference window 和 trajectory memory，分别服务于坐标锚定、近期稠密关联和长程上下文。官方仓库提供 keyframe/windowed 等长序列工具，但模型置信分数、显示点云和与 B2 的接近程度，都不是独立几何真值。[代码和使用说明](https://github.com/Robbyant/lingbot-map)

对本项目最有用的迁移是：将流式模型作为新的几何/姿态信息源，保留原始时序与状态边界，用 RGB 检索和学习匹配提出候选，再用没有参与拟合的传感器几何裁决。不能把未知尺度的整条轨迹直接覆盖度量 B2；也不能把每帧拟合的深度倍率直接乘绝对相机中心。

**关联：**B1 尺度与位姿误差分解、B2 新匹配/新初值、B3 有序预取；R6 只消费独立验收通过的局部位姿。此处不要求重新训练 Map。

### 2.3 LingBot-Map 2：修订旧上下文的思想与公开 base 的边界

[Map 2 技术报告](https://raw.githubusercontent.com/Robbyant/lingbot-map-v2/078b54485830c74ab1d3797667073e3e8353f3b9/docs/lingbot-map-v2_paper.pdf)讨论 Geometric Context Reconsolidation：保留可修订的中间表征，在新的已观测信息到来后重算相关历史上下文，并让后续流式推理读取修订后的状态。这涉及模型层次、缓存改写与相应训练，不能仅凭更改 attention mask 获得。

公开边界以[本项目固定 README](https://github.com/Robbyant/lingbot-map-v2/blob/078b54485830c74ab1d3797667073e3e8353f3b9/README.md)为准：发布的是 **pure causal geometric streaming base，without GCR**；原代码及 base 权重为 CC BY-NC 4.0；README 明确权重仓库当时需有权限的账号。项目已取得 `robbyant/lingbot-map-v2-base-streaming` 的配套配置与权重，历史回执记录 `GCR=false`。已有下载成功不能推导为公众无需认证可下载。

本轮 D/E 可实现的是**外部显式证据记忆**：对象 ID、来源组、可见性、假设/确认/矛盾状态、局部版本与回滚。M0/M1/M2 的价值应由残影、错误删面、静态背景误更新及未来预测判断。它受 reconsolidation 思想启发，但不是官方 GCR 网络复现；KV cache 也不会自动变成碰撞资产或物理状态。

### 2.4 LingBot-World 2：借鉴因果交互，暂缓生成大模型训练

[World 2 正文 §2–3](https://arxiv.org/html/2607.07534v1)将未来视觉序列建模为受历史上下文与当前动作条件驱动的因果生成；动作形式包括相机位姿和按 chunk 对齐的文本提示。其数据管线区分全局背景、局部事件与控制时序，并防止未来事件泄露到当前条件；训练采用因果设计和后训练蒸馏以改善长程生成。

本轮可直接借鉴的是日志与训练样本的因果边界：动作发生/生效时间、当前可观察证据、局部事件和未来标签分开；不能以未来对象姿态纠正当前预测后再计“预测准确率”。模型的视觉交互能力不提供本项目所需的显式米制碰撞、质量、接触力真值。

[当前 README](https://github.com/Robbyant/lingbot-world-v2)列出 14B causal-fast、causal-pretrain、bidirectional 和 1.3B causal-fast，提供分块 KV inference；1.3B 包还需要共享 T5/VAE/tokenizer 资产。README 明确不计划发布官方部署代码，项目许可为 CC BY-NC-SA 4.0。论文/演示吞吐不直接作为本地硬件性能承诺。**本轮暂缓 World、VLA、VA 大模型训练**，先完成 D/E 的可解释状态与真实动态证据。

## 3. 几何、融合和薄结构：哪些已经有人做过

### 3.1 DN-Splatter 与 AGS-Mesh：先验权重与平滑归因已有基线

[DN-Splatter 正文](https://arxiv.org/html/2403.17822v3)已有 RGB 梯度调节的深度损失、法线监督和法线平滑。对应 [losses.py](https://github.com/maturk/dn-splatter/blob/97588b4290128ce7ba6fdbfaac3020b42b17de4c/dn_splatter/losses.py) 的 `EdgeAwareLogL1` 是直接实现证据。因此“按图像边缘给深度加权”本身不构成新方法。

[AGS-Mesh §4](https://arxiv.org/html/2411.19271v2)用深度/法线之间的一致性过滤低质量先验，并在训练中自适应减弱不可靠法线约束；还研究不同网格提取方式。同一 [losses.py](https://github.com/maturk/dn-splatter/blob/97588b4290128ce7ba6fdbfaac3020b42b17de4c/dn_splatter/losses.py) 有 `AdaptiveDepth`、`AdaptiveNormal`。这支持 R5 的必要性：若只关闭 normal TV 就恢复薄结构，不能把全部收益归给更复杂的边界模块。模型深度和最终 mesh 必须分别评价。

### 3.2 ConfidentSplat：多视角支持加权已有，绝对风险仍需验证

[ConfidentSplat §III-C](https://arxiv.org/html/2509.16863v1)以多视角一致支持数构造权重，将多视角深度与单目先验混合，单目权重取其补数，并结合姿态优化及 Gaussian 地图更新。没有多视角支持时单目权重增加，属于相对来源选择，不能自动解释为该像素具有较低绝对误差风险。本次未核实完整官方代码，因此只作方法对照。

本项目应检验的新增差异是：候选风险是否由独立原始观测校准；证据不足时能否拒绝所有来源；遮挡前后景是否保留互斥模式；与 NN/硬阈值相比，在相同覆盖率上是否降低错误接纳。共享 RGB 的多个模型互相同意，不能被计算为多个独立传感器证据。**关联 A、R2/R3。**

### 3.3 Improved-GS：边缘驱动增长与长轴分裂已存在

[论文](https://arxiv.org/html/2508.12313v1)已有 edge-aware score、long-axis split、恢复感知裁剪和增长控制。已检查 [densification_stage.py](https://github.com/XiaoBin2001/Improved-GS/blob/20b5db343a87c2f9fb4b70c6e39ff1f042a8a47a/scene/methods/densification_stage.py) 与 [gaussian_model_densification.py](https://github.com/XiaoBin2001/Improved-GS/blob/20b5db343a87c2f9fb4b70c6e39ff1f042a8a47a/scene/gaussian_model_densification.py)：它按图像边缘证据选择分裂，并控制增长预算。

R4 不能只重新命名“沿边缘加点”。实际差异需是由独立几何支持的深度不连续/轮廓驱动、抑制跨断层假连接，同时使用与 R3 相同总 Gaussian 预算。RGB 图案边缘与真实物体边界要分开。代码可作为增长调度基线，不应整套替换本项目而丢失归因。

### 3.4 SurfFill：薄结构缺失区定向补点已有相近工作

[SurfFill v2 方法](https://arxiv.org/html/2512.03010v2)围绕 LiDAR 在细结构/边缘的缺失，以局部密度变化识别模糊区域，将优化和 densification 集中到这些区域，并对局部正则化、保留原始扫描与新点提取做专门处理。[官方 train.py](https://github.com/qy21gafy/SurfFill/blob/4849b5b6bd5fb5279913f7cec8e68de09f808e90/train.py)可以核对相应 mask 和损失。

这与“低置信区域加密、减弱平滑”高度接近，不能预先宣称本项目首创。可借鉴其对可信原扫描与待补区域的分工；本项目要额外解决的是传感器投影 lineage、候选风险校准和真实视角证据。其高精度 LiDAR 输入假设不能直接等同于大厅重采样后的低分辨率 RGB-D。**关联 A1、R4/R5。**

### 3.5 MASt3R-SLAM：新匹配可带来增量，局部优化本身不新

[MASt3R-SLAM 方法](https://arxiv.org/html/2412.12392v1)包含学习匹配、相机跟踪、局部融合和全局优化；跟踪使用对深度误差较稳健的射线方向项，并处理相应退化。[tracker.py](https://github.com/rmurai0610/MASt3R-SLAM/blob/e6f4e3d474fad0e11f561482012be864ba8c3f17/mast3r_slam/tracker.py)提供实际实现。

B3 已做 SIFT、EPnP、联合重投影/极线优化及独立 holdout，新轮不能只再次做“局部 BA”。可检验的增量是学习对应、基于 RGB 而非可疑 B2 朝向生成邻接、Map 独立初值以及米制传感器锚点。原方法的 Sim(3) 优化设计也不能直接当成已标定 RGB-D 的逐帧尺度自由度。**关联 B2、R6；R7 仅在两个分支各自验收后测试组合。**

## 4. 动作、物性和数据：补足几何模型没有提供的接口

### 4.1 RE3Sim：外观与碰撞分开，静态视觉不自动给出物性

已读 [v2](https://arxiv.org/html/2502.08645v2) 并核对 [v4 §III](https://arxiv.org/html/2502.08645v4)。它分别恢复背景和前景网格，背景用 GS、前景可用 mesh 渲染，通过深度合成保持遮挡一致；机器人使用已有资产。论文明确从静态视觉获取质量/摩擦困难，采用默认值。专家利用真值物体位姿及 RRTConnect 产生轨迹，并筛掉失败样例。

[当前官方代码](https://github.com/Sage-PhysicalAI/Re3Sim)依赖 IsaacLab/GS，仓库说明许可为 CC BY-NC 4.0；[仿真资源](https://huggingface.co/datasets/RE3SIM/sim-resources)和 [ACT 示例数据](https://huggingface.co/datasets/RE3SIM/act-dataset)可取得。D 可借鉴其资产分层及 GS 背景 + mesh 前景；E 借鉴动作/观察记录。使用默认物性得到的 hall sandbox 应标 assumed，不能称经真实交互辨识的 twin。

### 4.2 TwinAligner：先对齐控制响应，再拟合物体响应

[论文 §III-C](https://arxiv.org/html/2512.19390v1)把视角和动力学分开处理。Control–Hit–Slide 先比较同控制输入下的机器人关节轨迹，再利用物体运动和 ADD/ADD-S 优化摩擦、质量、质心等。它提醒我们：控制器增益/延迟、初态误差和碰撞形状错误都可能被错误地吸收到摩擦参数中。

但 [README 的 Dynamic Real2Sim](https://github.com/TwinAligner/TwinAligner#dynamic-real2sim) 截至核验仍为 Coming soon。**可借鉴实验分解，不能承诺调用已公开的完整动力学辨识模块。**本项目 C0/C1/C2/C3 是自己的实现设计；只拟合数据可辨识的参数。单纯 Coulomb 滑行的平移减速度为 μg，不能据此唯一恢复质量。**关联 D/E。**

### 4.3 PhysTwin：有真实 RGB-D 和控制点的动态验证资源

[官方 PDF p3–5](https://jianghanxiao.github.io/phystwin-web/phystwin.pdf)明确使用三路 RGB-D 交互视频；手部轨迹升到 3D 后作为控制点，物体采用 spring–mass，分阶段优化拓扑/物理，再用 Gaussian 外观及 LBS 渲染。它既有重现也有未来预测和未见交互实验。动作语义是控制点位置，不是机器人关节命令或测得外力。

[MIT 官方代码](https://github.com/Jianghanxiao/PhysTwin)与[数据](https://huggingface.co/datasets/Jianghanxiao/PhysTwin)可取得；本报告第 5 节给出实际文件审计。软体支线保留其 spring–mass/Warp，不强行塞入 D 的 MuJoCo 刚体接口。论文中的形状先验与多视角处理也可能引入隐藏观测，严格归因须重新界定哪些资产是共同先验、哪些数据只作评价。**关联 E 的补充 RGB-D 动态实验。**

### 4.4 Real-to-Sim Robot Policy Evaluation：把仿真可用性与真实有效性分开

[正文 §III](https://arxiv.org/html/2511.04665v2)结合 GS 场景/机器人分层、URDF/FK、颜色对齐和扩展 PhysTwin 接触，用配对真实/仿真初态比较多个策略及 checkpoint。它的机器人接触通过物理求解处理，而非简单把对象节点固定在夹爪上；报告的策略相关性是独立于视觉相似度的证据。

[官方代码](https://github.com/kywind/real2sim-eval)含 `reset/step`、键盘交互及 [motion replay](https://github.com/kywind/real2sim-eval/blob/8bd70918307aa23d0ebdd3c231aac18300533dab/experiments/replay.py)，[集合](https://huggingface.co/collections/shashuo0104/real-to-sim-policy-eval)公开资产、权重和真实示教。其自身 release plan 的 PhysTwin training 未勾选，创建新软体仍需原 PhysTwin 流程。本轮只借接口、配对回放与评价，不训练 VLA/VA，也不因小状态预测器改善就宣布机器人策略成功率改善。

### 4.5 iTACO：有运动观测才建立关节，运动学不等于物性

[v2 §3.1–3.3](https://arxiv.org/html/2506.08334v2)输入初始表面和手持 RGB-D 交互视频，先分离静态/动态区域，再联合优化相机、关节与部件。它假设一次只有一个部件运动；多关节可通过连续多段交互逐个处理。输出关节类型、轴、部件等显式运动学结构。[代码](https://github.com/3dlg-hcvc/video2articulation)可作为以后门/抽屉分支的起点。

日期需要特别保持准确：v2 是 **2025-09-25**，v3 是 **2025-11-17**；3DV 2026 是会议标识，不能改写成“2026 年新预印本”。静态关门画面不能独立辨识真实轴、限位、摩擦和动力学。当前 D 默认单刚体，iTACO 暂缓到实际取得开合观测后。

### 4.6 Real-is-Sim：让新观测持续纠正世界状态

[Real-is-Sim v2](https://arxiv.org/html/2504.03597v2)以 Embodied Gaussians 的物理粒子和视觉高斯构建动态孪生，利用真实 RGB 反馈持续纠正模拟状态，并让策略以模拟器状态为统一接口。它与“随交互校准”的目标直接相关：[项目页](https://real-is-sim.github.io/)给出 PushT 的同步、虚拟评价和数据增强演示。

这里最该借鉴的是观测—预测—残差—状态修订闭环。视觉反馈把当前对象拉回正确位置，不能单独证明质量、摩擦或未来动力学已经被辨识；也不能每一步读取真值后把累计轨迹误差称开环预测准确率。因此 D/E 分别评价有相同当前观测的在线更新，以及无中途真值校正的冻结世界预测。本轮不复刻其策略训练/实机跟随系统。

### 4.7 两个补充参照：几何条件与批量吞吐

[MapAnything](https://arxiv.org/html/2509.13414v2)允许图像配合 K、位姿、深度等可选几何条件，输出分解的相机/深度/尺度。这为 B 线提供 RGB+K 与加入部分真实 Z 的独立对照，有助于判断失败来自单目尺度还是局部几何；不因模型声称米制就跳过本场景标定和 fit/audit。[官方实现](https://github.com/facebookresearch/map-anything)仅作可选扩展，避免又建立一个阻塞灵波主线的大系统。

[Boba](https://jianxiapyh.github.io/Boba-project-page/)专门研究物理 Gaussian twin 的批量 rollout：共享静态拓扑/外观模板，每个环境只保存独立动态状态，并共同优化物理、形变和渲染。它对 96 GB 显存的启示是减少复制大资产，把资源花在更多独立轨迹上；论文的 aggregate FPS 是许多环境合计吞吐，不能解释为单环境帧率或承诺本机速度。本轮先保证 D/E 对照和状态隔离正确，之后才将模板共享/批处理作为等价性能优化；不默认替换已验证物理求解器。

## 5. 公开动态数据的真实审计

### 5.1 Public PushT：有二维动作，无 RGB-D 和现成对象真值

读取 [xarm7_pusht 数据卡](https://huggingface.co/datasets/shashuo0104/xarm7_pusht)及 [meta/info.json](https://huggingface.co/datasets/shashuo0104/xarm7_pusht/resolve/main/meta/info.json)确认：60 episodes、47,582 frames、30 Hz、两路 240×240 RGB；`action` 与 `observation.state` 均为 shape `[2]` 的 `xy waypoint`，另有 timestamp/frame/episode 索引。这些 xy 不是 T-block 对象位姿。没有深度字段、对象姿态真值或力矩/接触力标签。

[replay.py](https://github.com/kywind/real2sim-eval/blob/8bd70918307aa23d0ebdd3c231aac18300533dab/experiments/replay.py)的 `load_robot_json` 默认读 `action` 前缀，再逐帧组成动作执行；支持 xy、qpos、Cartesian 或末端位姿等格式。这证明提供了动作重放接口，不能替代对具体数据的 command/执行位置/时间语义审计。[xarm_pusher.yaml](https://github.com/kywind/real2sim-eval/blob/8bd70918307aa23d0ebdd3c231aac18300533dab/cfg/env/xarm_pusher.yaml)提供相机参数和 pusher/URDF 配置，但 848×480 原相机 K 不能直接用于 240×240 训练视频，需恢复缩放/crop。

建议按 episode 分组冻结 36 fit / 12 calibration / 12 test，先对近重复起点/轨迹分组。建立独立、冻结的对象跟踪参考，并带可见性和不确定性；不能拿 C3 的状态估计给 C3 自己评分。真实执行器轨迹作为未来条件时，应称条件回放预测，不与仅给 command 的完整系统预测混为一谈。

### 5.2 `phystwin-T-block/data` 是静态生成资产

读取 [完整文件树](https://huggingface.co/shashuo0104/phystwin-T-block/tree/main)后，实际资产只有约 337 kB：

| 文件 | 字节数 | 能证明什么 |
| --- | ---: | --- |
| `data/T_0001/final_data.pkl` | 80,648 | 静态 object_points/colors/surface/interior 数据 |
| `experiments/T_0001/train/best_0.pth` | 255,054 | 用于模拟的参数资产 |
| `experiments_optimization/T_0001/optimal_params.pkl` | 94 | 少量初始化参数 |

本次安全读取 pickle 的字段字符串，确认没有 controller 或真实动态轨迹字段。[官方 create_rigid_phystwin.py](https://github.com/kywind/real2sim-eval/blob/8bd70918307aa23d0ebdd3c231aac18300533dab/experiments/utils/create_rigid_phystwin.py)从 mesh 采样并写入 `object_points[None]`，即 T=1，颜色全白，刚度/摩擦等有硬编码。**不能把此 data 目录当成 PushT 缺失的真实 RGB-D/对象运动数据。**使用已知 T-block 资产能验证桥梁与校准，但不直接证明灵波恢复了碰撞几何。

### 5.3 PhysTwin 两条真实 RGB-D 数据：已核实际文件和帧数

官方 [data.zip](https://huggingface.co/datasets/Jianghanxiao/PhysTwin/resolve/main/data.zip)大小为 8,075,884,740 B，LFS SHA-256 为 `586a237a0f870b2a0a144ee69c3f04b636cac4fe251df969c1fd1feaf67eeae2`。本次使用 HTTP Range 读取 ZIP 中央目录及指定 metadata/split；没有下载整个包，也没有运行模型。

已确认两条 case 均有 `color/{0,1,2}/*.png`、`depth/`、`metadata.json`、`calibrate.pkl`、`split.json`、`final_data.pkl`、`track_process_data.pkl`、`gt_track_3d.pkl`：

| ZIP 内路径 | 采集参数 | 官方划分，半开区间 | 可支撑的最小预测长度 |
| --- | --- | --- | --- |
| `data/different_types/single_push_rope_1/` | 三路 RGB-D，848×480，30 Hz，92 帧 | train `[0,64)`，test `[64,92)` | 约 0.5 秒；留未来总长不足 1 秒 |
| `data/different_types/single_push_rope_4/` | 三路 RGB-D，848×480，30 Hz，83 帧 | train `[0,58)`，test `[58,83)` | 约 0.5 秒；不能套用 2 秒 horizon |

[data_process_pcd.py](https://github.com/Jianghanxiao/PhysTwin/blob/81c718790a37e5e0102eb77af2c6edd34a9db25f/data_process/data_process_pcd.py)从 `metadata.json` 读 K、从 `calibrate.pkl` 读外参，将原深度除以 1000 转米。[real_data.py](https://github.com/Jianghanxiao/PhysTwin/blob/81c718790a37e5e0102eb77af2c6edd34a9db25f/qqtt/data/real_data.py)读取 object_points 与 controller_points。动作来自手部观测，不是机器人 command。

不能原样继承的两处评价细节：

- [data_process_track.py](https://github.com/Jianghanxiao/PhysTwin/blob/81c718790a37e5e0102eb77af2c6edd34a9db25f/data_process/data_process_track.py)根据整段视频可见性筛选控制点。现成控制点适合“已知记录作用轨迹”的离线条件预测，不能冒充严格因果在线动作估计。
- [evaluate_track.py](https://github.com/Jianghanxiao/PhysTwin/blob/81c718790a37e5e0102eb77af2c6edd34a9db25f/evaluate_track.py)在某帧没有有效点时把误差设 0。本项目应记 missing 并报告有效分母。`gt_track_3d` 是视觉/深度派生的跟踪参考，仍需保留不确定性，不升级为高精度独立运动捕捉真值。

## 6. 最小归因实验与现实边界

### 6.1 RGB-D 支线的拟议实验，尚未执行

研究问题限定为：**在已知物体及共同动力学先验下，灵波条件深度是否改善当前对象状态修订，并改善给定控制点后的未来预测？**

1. 用上述两条 case，分别保持官方 test 不动；如果需要调阈值，从各自 train prefix 内再划 calibration，不看未来 test。不未经物体身份核验就将两段视为可互换的同一物体多次交互。
2. 三个条件共用同一拓扑、物理参数、节点数、控制点路径和状态更新预算。共同先验的训练来源必须披露；如预训练资产包含不允许的未来对象信息，重建允许前缀或标 blocked。若保留这类资产作调试，只可另标 privileged-model sanity，不计入严格未来预测结果。
3. 在 test 起始时刻，只给状态修订器 camera 0 的 RGB 和已结构化隐藏的 depth。比较 sensor/NN、LingBot 直接补全、LingBot 候选与证据拒绝三组。
4. camera 1/2、隐藏读数及后续对象观测只用于评价。动作条件可以是给定未来 controller_points，但绝不把未来对象坐标送入修订器。
5. 当前状态修订后开环预测约 0.5 秒，评价共同有效帧上的跟踪、独立视角几何及失败数。绳的物理仍用 spring–mass；这不是把柔性物体近似成一个 MuJoCo 刚体。
6. 这一实验只支持状态校正和条件预测结论。若要进一步主张“灵波几何改善物性辨识”，必须从允许 prefix 重新构建并同预算拟合几何分支，不能直接复用含所有视角几何的 `final_data.pkl` 来补回本应隐藏的结构。

若真实 RGB-D、标定、控制点、独立对象参考或同物理先验的状态修订接口不足，H3 的灵波真实动态归因应标 blocked；hall 几何、显式记忆、PushT 回放和合成数据接口继续，不用自仿真成功替代缺失证据。

### 6.2 D/E 的默认物理桥梁与验收

建议独立 MuJoCo CPU 进程管理局部平面、一个刚体和 pusher，现有 GS 保持外观渲染；这是项目适配方案，不是 RE3Sim 或 PhysTwin 原生接口。MuJoCo 的 [simulation loop](https://mujoco.readthedocs.io/en/stable/programming/simulation.html)可无 GUI 执行；[mesh 文档](https://mujoco.readthedocs.io/en/stable/XMLreference.html#asset-mesh)说明碰撞用凸包，因此不能把整厅 mesh 当单一碰撞体而封死门洞/桌下空间。视觉表面、碰撞代理和物性分别记录来源与误差。

C0/C2/C3 应在 fit/cal 后冻结，用同一初态清单和预生成动作数组执行 512 条配对 rollout；只固定 seed 不足以保证随机调用分支不同后仍得到相同动作。C3 的在线更新另在真实观测 episode 或独立观测生成器的合成 sanity 上评价，不能用它自己的渲染确认自己的补全。模拟 GT 只给 evaluator，不能直接喂校准器。

小型对象状态预测器保持架构、历史/horizon、采样比例、优化预算和评价标签相同，并报告真实样本暴露次数；20k updates 本身不保证同预算。至少比较持久/恒速、RealOnly 和合成增强；若分别生成 C2/C3，应提前明确使用哪一组或分别报告。它能够检验数据对对象预测的价值，不能替代视觉政策或真实机器人成功率实验。

## 7. 本轮取舍与贡献边界

| 类别 | 本轮处理 | 对应产物 |
| --- | --- | --- |
| 直接复用并核身份 | LingBot-Depth 候选、Map 2 causal base、DN 基线；公开 PushT/PhysTwin 数据读取与回放接口 | A/B 候选与来源、C 完整对照、D/E 可重放数据 |
| 借鉴并实现可证伪差异 | MDM 结构化隐藏；候选绝对风险与拒绝；几何支持的预算分配；显式对象证据版本；控制器/物性分阶段校准 | R1–R5、M0/M1/M2、C0–C3 及负面结果 |
| 有条件扩展 | 新匹配/Map 初值通过独立验收后进入 R6/R7；PhysTwin 严格 RGB-D 动态归因 | 独立几何残差及真实留未来预测 |
| 暂缓 | Map 2 GCR 网络复现、World 2/VLA/VA 大模型训练、全厅物性自动识别、未观测关节的可靠恢复、机器人策略提升结论 | 明确缺失信息和下一步数据需求 |

已经存在的机制——深度/法线权重、置信融合、边缘 densification、局部 BA、仿真参数拟合——不应改名后当成新贡献。可研究的主线差异是：**以灵波提供候选，保留来源和互斥几何假设，在独立观测下校准接受风险，用可修订对象记忆管理动作后的证据，并验证这些变化是否改善真实条件预测和数据有效性。**新颖性、改进幅度及可发表性均由实验和进一步严格查重判断，本文不预先保证。
