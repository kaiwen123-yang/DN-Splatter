# AutoDL Codex：Metric3D v2 全程 30k 对照

你正在用户的 AutoDL 服务器上继续 `kaiwen123-yang/DN-Splatter`。用户最新补充“考虑跑一下 metrics”，本任务暂按 Metric3D v2 理解。执行这一具体、完整的对照；不要重复已经结束的 capture-full 或只再跑一次原有 metrics。用户不希望反复小验证和逐项等待确认，允许使用全部分配 CPU 和整张 RTX PRO 6000 GPU，环境和数据仍独立。

## 先读并确定身份

读取 AGENTS.md、reports/capture-full-review/2026-11-20/overview.md、docs/11_capture_metric3d_plan.md、docs/12_metric3d_normal_adapter.md、docs/13_capture_supported_mesh.md 和 configs/capture_metric3d_protocol.json。报告与实际行为按这些文件执行；旧 docs/08/09/10 的停止点和已完成任务不再启动。已有本地修改先保留，不 force reset。

- B0 已完成结果：e26ae72a75997dcfd3a907aa26b6c5c0677597a5。
- B0 最终 checkpoint SHA256：b89ad55fb5781582a6306a8c1f006ae5afe1e7f7ee5704c1594c8b7bbe6b8094。
- 现成全程几何：.local/data/custom/capture-full/20261119-full-5hz-3982cb26。
- 原输入：.local/runs/capture-full-dn-30k-20261119-v1/inputs。
- B1 新 run：.local/runs/capture-full-metric3d-30k-20261120-v1。
- 新报告：reports/capture-full-metric3d-30k/20261120-v1。
- 固定1846主帧，1478 train /184 val /184 test；原100辅助图只属于建图。

本轮所有必需输入已在 AutoDL，不重新读取 H 盘、不重新提取或上传 bag、不重做D2C/SfM/尺度，不向用户索要小样例。保留 B0、s01、VR 所有文件。只读核对实际资源、输入manifest/配置/最终checkpoint，复用既有Blackwell训练环境和下载缓存；已有同身份run按锁、PID、manifest和checkpoint续接，避免重复启动。

## 一次持续完成

1. 直接抽取 B0 已存在的训练日志、TensorBoard 和中间checkpoint，形成真实高斯数/loss曲线。没有记录的旧剔除原因明确 unavailable，不能从文件大小猜测。把 s01 与 full 的现成米制 c2w 小JSON导出作几何对照，不重求解轨迹、不另开审核任务。
2. 固定 Metric3D 官方 eb5b6fac0dc155e4e52f576e304fbf11655ff339，`metric3d_vit_large`、`metric_depth_vit_large_800k.pth`、RAFT8。记录权重实际SHA、字节、来源和实际运行精度。GitHub操作优先在项目子shell启用AutoDL官方 `/etc/network_turbo`，不写全局配置。Hugging Face按官方域独立下载，不能把未下载的权重哈希写成已核验。
3. 不在已工作的训练环境照装Metric3D旧requirements或降级Torch/CUDA。实际缺依赖时使用独立`.local/conda/metric3d-bw`，复用支持Blackwell的Torch组合，最小兼容改动保存到仓库。已可用就继续，不重新搭建整套DN环境。
4. 全部1846图一次流式推理。严格按docs/12处理RGB、616×1064画布、去padding、恢复1280×720、前三通道单位化，原CV XYZ编码RGB PNG。保留现有normal_format=opengl，让历史loader完成唯一D=diag(1,-1,-1)变换；不要提前再翻YZ，不用OpenCV的BGR可视化图代替训练法线。保存原始法线、κ、诊断深度与预处理元数据。κ不新增权重/筛选。训练只读train法线，val/test预测不参与训练或网格筛选。
5. 诊断深度使用真实去畸变内参，按官方公式只做一次resized_fx/1000校正；不额外重复半像素偏移，不用默认九种焦距，不用留出数据拟合尺度。本轮继续sensor depth0.3–6m、mono_depth_lambda=0，不用Metric深度替换、补洞或扩范围。留出先验深度指标另存。
6. 创建独立B1输入副本，逐帧核对RGB/Z/K/c2w/split/PLY与B0一致、只有normal改变。训练法线PNG实际经历史reader读回并与数值公式比较；这项全量转换核验直接嵌入准备，不开小训练。写配置语义diff，保证seed42、1478训练图、2001906初始化点、scene_scale21、分辨率、损失/权重、采样/增密/剔除/学习率调度不变。
7. 从相同初始化完整训练累计30000更新，最终step29999。保留历史首遍随机/后续顺序采样和15k停止增密，以隔离法线变量；本轮不同时修shuffle、延长240k或改cull阈值。正常训练首个正式batch核对有限性后持续到终点，不另做smoke。每5k保存，断点按真实saved_step算remaining，保留optimizer/scheduler。正式记录每次采样帧、Gaussian计数和split/duplicate/cull事件；原因重叠给union，不影响随机数或计算路径。
8. 用同一B1最终checkpoint/SHA完成184val和184test，逐帧及均值PSNR/SSIM/LPIPS，深度等权帧与参考像素合并MAE/RMSE/AbsRel/P50/P95，公开完整分母与无效预测。输出相对B0的逐帧差值/改善退化数量、30s时间窗和s01同源48帧对照。没有外部几何GT，不能伪造Faro/Chamfer/F-score。
9. 生成1846位置全程 reference/B0/B1 RGB视频，按采集时间播放；B0已有匹配渲染可按身份复用。保留固定首中末、238/239、1409、最大深度误差帧、353秒和末端倾斜。不要只挑最好图。
10. B1先导出同原协议的原始Poisson：1478train、目标5M、depth9、不新增筛选。随后依次在B0和B1最终checkpoint上运行docs/13的同一套完整sensor-supported导出：预筛选、每图最多25907点、实际1cm体素融合、depth11、固定多视角支持裁剪。保留裁前和裁后，记录所有过滤/覆盖/资源。不要按val/test调阈值，不把B1适配网格相对B0原始网格的收益全归给Metric3D。
11. 模型/PCD/mesh评价给完整有效参考分母、命中覆盖、条件MAE及共同支持像素，给5/10/20cm正确覆盖率、伪前景比例；全场全部留出和原三处局部都报告。保存每点源帧/像素和投影获胜来源，区分自身反投影、跨视角融合遮挡、Poisson补面。
12. 整理中文报告、实际配置与diff、法线/深度接入证据、训练曲线、全部指标和视频/网格对照、耗时资源、源码/权重/输入/产物SHA。新增复用脚本必须进scripts，不能只藏.local。更新本轮latest.json为实际状态，原B0结果不改写。只提交本任务小报告/代码/图/清单，检查diff排除凭据及大文件，再push用户main并核对远端提交和报告一致。完整视频沿用既有授权回传路径并校验SHA。

## 资源、终点和输出

全量先验推理、正式DN训练和大点云网格顺序执行。B0进程峰RSS约79GiB，不能因96GB显存有余就并发复制两个全图缓存；分块处理最多约38M候选点。允许全部实际CPU/GPU，但不结束其它进程、不改系统环境，不设置自动关机或训练完成时限。

持续执行到全部交付，不在环境、法线、首批训练或某项质量较差后停下来询问是否继续。真实错误做最小修复并保留偏差；只有访问/身份/实际运行容量等无法推进的阻塞才交代原因、已完成工作和恢复命令。不要自动启动其它模型、超参数搜索或下一轮240k。

最终回传：真实状态和commit；Metric3D权重/法线数值约定；训练实际更新/最终step/高斯数和曲线；B0/B1全量val/test及时间段、同源对照；先验深度诊断；两种导出在两模型上的覆盖/准确度/伪面和局部；完整视频与大产物清单。明确回答法线替换是否有效、网格改善来自什么、后续是否该调整完整训练调度。未执行的内容写未执行。
