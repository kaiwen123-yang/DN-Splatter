# 本地 Codex：接续 AutoDL 的 Metric3D v2 完整对照

继续用户的 `kaiwen123-yang/DN-Splatter` 项目，保留工作区已有修改，取得并阅读最新 AGENTS.md、reports/capture-full-review/2026-11-20/overview.md、docs/11_capture_metric3d_plan.md、configs/capture_metric3d_protocol.json 和 prompts/autodl_capture_metric3d.md。

本轮暂按用户所说的“metrics”为 Metric3D v2，进行一个新的全程30k法线替换对照。现有capture-full已经完整结束，B0结果固定在e26ae72a75997dcfd3a907aa26b6c5c0677597a5。全部1846帧几何、深度、原始输入和checkpoint都已在AutoDL，本地无需再读取H盘、重新提取bag、上传数据或回到小样例验证。

复用上一轮已经授权并正常工作的AutoDL连接，检查服务器该项目状态，接续执行 prompts/autodl_capture_metric3d.md 的完整任务。只运行一个协调任务，避免本地和服务器各自重复启动。保持持续日志和阶段状态；若已有同身份run就续接。允许全部分配CPU/GPU，按内存约束顺序安排推理、训练和网格。

任务包括：Metric3Dv2 ViT-Large全量法线；保留真实sensor深度及原位姿/初始化/划分/调度的B1完整30k；全部val/test、同源和时间段评价；全程reference/B0/B1视频；B1原始Poisson以及B0/B1同参数支持约束网格；代码和中文报告推送并核验。

无需中间审阅或小训练，不重做已完成的Blackwell/VR/SfM。具体参数与法线坐标转换按服务器提示词及docs/12/13。只有真实连接/权限/身份或运行阻塞无法处理时报告，不索要已经掌握的信息，不声称尚未完成的运行成功。

结束时经原授权连接把完整对照视频和小报告同步到本项目既有本地结果位置，核对哈希及本地/GitHub/AutoDL提交身份。无需额外Windows全盘归档，不删除服务器原数据、模型或网格。最终中文回传真实状态、commit、全量对比指标和效果判断、大产物路径及恢复入口（若实际受阻）。
