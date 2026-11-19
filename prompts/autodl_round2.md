# 给 AutoDL Codex：第二轮执行提示词

请在我的 DN-Splatter AutoDL 项目中实际执行第二轮。仓库：
https://github.com/kaiwen123-yang/DN-Splatter.git

上一轮 VR Room 已完成，结果提交为 c747cc92d70055d5d5ceb70699e85b77faad87da。目标机 RTX PRO 6000 96GB、120GB 内存、25 个有效 CPU；我允许使用全部已分配 CPU 和整张 GPU。继续独立 Conda/源码/缓存/输出，不修改其他项目和全局驱动；GitHub 优先在项目子 shell 中使用 AutoDL 官方学术加速。

先确认你已在目标服务器，保留工作区修改，获取仓库最新计划。阅读 AGENTS.md、reports/review_2026-11-19.md、docs/04_round2_plan.md、docs/06_custom_data.md、configs/round2_protocol.json，以及首轮 run_manifest/config/patches。如果已有第二轮任务，核对归属并接续，不重复启动。

【本轮目标】

A. 不重训 VR Room，使用原 checkpoint、渲染和网格补齐诊断。
B. 固定 coffee_room/iPhone 作为第二场景，完整 30k、双轨迹评估、Poisson 和几何评价。
C. 准备自有数据入口；有我明确提供的数据目录时处理，没有时仅将 C 标为 waiting_for_dataset，继续完成 A/B。

【先解决跨场景运行器问题】

现有 autodl_env.sh 和 vr_room_*.py 写死 VR Room 输入、输出、Faro 和论文参考值，不能只改 --data。请做最小参数化，使 dataset、scene、run_id、seed、parser 和 GT 可用性明确。每个 run 有独立目录、锁、阶段标记；恢复要核对输入/配置/源码身份。保留原 VR Room 产物，不用它的 checkpoint 或旧 mesh-export.ok 跳过新场景。复用已经实际验证过的环境和兼容补丁，不从头重装。

【A：现有模型诊断】

从原日志提取训练曲线和 checkpoint 读取失败/恢复的精简记录；把正式训练时长、后处理时长、中断空档分开。独立梯度探针未跑不等于 Blackwell 失败，不为这个布尔字段另堆 smoke tests。

保存 42 帧 within 和 186 帧 with 的逐帧 RGB/深度指标，使用同 checkpoint 真实预测或注明来源的完整渲染，不从缩略 JPEG 反算正式分数。固定展示最好/中位/最差及已知薄弱视角。

同视角对照原始 Poisson、官方裁剪 mesh 与 GT，区分原几何缺陷和裁剪边界。保留所有原始 JSON/PLY，将新增审计写独立目录。

新增定义明确的有效深度 MSE/RMSE 和标准 log-RMSE；报告无效预测，不靠删除失败像素改善指标。法线先确认编码/坐标，解码并单位化后计算“与 Omnidata 先验的一致性”，不能叫几何真值角度精度。

官方 cull_mesh 接收 test_ids 却没有过滤，原结果用了全部418个long_capture注册视角。保留official口径，另存只用376训练视角的strict_train_only结果，对GT和预测使用同样视角/算法。记录两套帧列表。需要量化F1末位波动时，对同一已有网格使用评价采样seed 0/1/2，保存全部结果，不重训或挑最好结果覆盖原值。

【B：第二场景】

使用独立 run_id coffee-room-iphone-30k-s42。复用官方固定源码811f5cd及已登记补丁、Omnidata low/opengl；保持首轮损失和初始化协议，训练seed42累计30000更新，5000步保存，最终step29999。单独记录网格采样和几何评价seed。

先核对coffee_room的真实帧划分、RGB/深度/法线、单位和仅训练帧初始化，再完整执行。目标5M网格采样、Poisson depth9，显式使用全部有效CPU预算。没有作者coffee_room单场景值时，不把VRRoom或六场景均值当其达标线。

【C：自有数据】

目前我未在此任务中提供自有数据目录。不要把其他项目数据自动当作本项目输入。先准备独立入口和输入说明；若工作区已有我明确提供的数据，按docs/06进行只读盘点、转换和必要的5–10帧几何核验。

静态RGB-D、内参、逐帧相机位姿齐全时用历史normal-nerfstudio路径，显式预计算Omnidata，注意单位/轴约定/深度配准/初始化点云。新场景重新初始化并优化，不使用VRRoom checkpoint直接推理新场景。输入满足后进行一个自有场景完整30k；没有几何GT就不计算或伪造Faro/F-score。

【持续执行与交付】

合理并行CPU准备工作，GPU重阶段按容量安排，本轮默认一个新场景正式训练进程。不要为100%利用率改变论文参数。使用tmux，不设置训练/网格完成deadline，不自动关机，不结束其他项目。

推进到A/B完成并保存C的真实状态。推送最小代码改动、配置、逐帧指标、代表图、原始/审计结果、结论及大产物哈希清单；不做Windows归档，不删除服务器数据。阶段状态必须来自真实执行，不能把这份计划当成已完成。
