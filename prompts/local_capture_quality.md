# 本地 Codex：接续 AutoDL 完成 capture-full B2

继续 `kaiwen123-yang/DN-Splatter`。保留本地已有修改，取得最新main，阅读AGENTS.md、reports/capture-quality-review/2026-11-21/overview.md、docs/14_capture_quality_plan.md、configs/capture_quality_protocol.json及prompts/autodl_capture_quality.md。

用户现在要求提高完整重建质量。B0、Metric3D B1已全部结束；下一项是有明确证据依据的完整B2：局部错误位姿纠正、可靠训练SfM初始化补充、保护和增长模型容量、从头240k、完整val/test/全程视频，以及传感器TSDF/模型TSDF/支持Poisson对照。不是再次运行Metric3D推理对照，也不是单做检查报告。

复用上一轮已授权且正常工作的AutoDL连接，核对该项目状态，实际执行服务器提示词。输入/先验/环境已在服务器，原bag无需重新读取上传。本地只开一个协调者，服务器已有相同run则续接，不重复启动。允许全部分配CPU/GPU，文件和环境继续隔离，AutoDL GitHub优先官方学术加速。

按docs/14及JSON实现必要可复用脚本，并持续到一轮完整结果交付。现成B1六checkpoint全val和B2正式每30k全val属于整轮工作，不新增短训练或参数扫描。不设阶段审批、deadline、自动关机或GPU必须空闲门槛；不要结束其他项目进程。只有真实无法处理的访问/容量/执行阻塞才报告实际原因和恢复入口。

末尾将完整对照视频、小报告与用户既有授权的大产物回传本项目原结果位置，核对SHA及本地/AutoDL/GitHub提交身份。不要新增全盘归档、删除服务器旧数据或模型。公开内容遵守原有遮挡。最终中文回传完整指标、固定局部效果、质量目标是否达到、实际commit、产物位置和剩余问题。不要只说“已启动”或“计划已写”；未实际运行的步骤始终保留planned/null状态。
