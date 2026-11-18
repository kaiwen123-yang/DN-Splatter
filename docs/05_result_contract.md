# 结果、证据与交付约定

本轮开始时没有目标服务器结果。完成任何阶段后，将实际证据写入本文件约定的目录。`reports/latest_status.json` 的 null 表示尚未获取，不表示零。

## 1. 小文件优先进入用户仓库

建议的公开产物：

| 文件 | 内容 |
| --- | --- |
| `reports/latest_status.json` | 当前阶段、实际状态、run_id、完成/阻塞项、下一步 |
| `reports/host_summary.md` | 硬件、实际容器配额、全 CPU/GPU 授权及各阶段资源预算、复用结论的脱敏摘要 |
| `reports/reuse_inventory.json` | 本项目复用的工具链、包/权重/数据来源、用途、只读/复制方式 |
| `reports/environment_actual.txt` | 实际安装版本和编译信息；不含凭据、私有URL |
| `reports/<run-id>/run_manifest.json` | 下述可复现元数据 |
| `reports/<run-id>/commands.sh` | 实际命令，去除凭据，项目路径使用变量 |
| `reports/<run-id>/config.yml` | 实际训练配置的脱敏副本 |
| `reports/<run-id>/rgb_depth_official.json` | 官方原始图像与深度指标 |
| `reports/<run-id>/faro_official_raw.json` | 如计算，保留有已知聚合缺陷的原始结果 |
| `reports/<run-id>/faro_audited.json` | 如计算，经审计的独立分轨迹结果及计算定义 |
| `reports/<run-id>/mesh_metrics.json` | 官方可见性裁剪后的mesh指标 |
| `reports/<run-id>/summary.md` | 结论、与论文差距、实际缺陷与限制 |
| `reports/<run-id>/previews/` | 适量RGB/深度/法线拼图及mesh预览 |
| `reports/<run-id>/artifacts.json` | 大文件名称、位置、大小、哈希、上传状态 |
| `patches/` | 固定源码的实际兼容、显式资源配置或评估修复，附说明 |

这里的文件名是交付接口，不代表全部已存在。程序原始输出名称不同可以复制为上述名称并记录原路径，不能修改数值伪装符合协议。

## 2. run manifest 必填信息

- `run_id`、Asia/Shanghai 与 UTC 起止时间。
- 用户仓库提交；官方源码 URL、base SHA、patch SHA256 与是否有未记录 diff。
- 实际 Python/torch/torchvision/Nerfstudio/gsplat、驱动、torch CUDA、nvcc 和 GPU compute capability。
- 显式环境前缀、缓存归属；公开信息只保留本项目相对路径。
- dataset、sequence、sensor、数据来源URL及下载/输入清单哈希。
- 训练/同轨迹测试/独立轨迹测试的准确帧数和文件清单；位姿和深度单位、parser变换与尺度。
- 先验网络及权重哈希、法线编码/轴约定、RGB与深度/法线分辨率。
- 初始化点云来源、是否仅训练帧、实际点数；最终高斯数。
- seed、完整命令与最终config、关键loss实际实现、已知论文差异。
- 初始/恢复checkpoint的step、每次追加更新数、累计完成更新数、最终checkpoint step。
- 各阶段耗时、峰值显存和系统内存、实际 CPU quota/affinity、各阶段 worker/native/编译/Poisson 线程数及 GPU 使用情况；如确有资源等待，记录具体容量阻塞和等待时长。全资源授权不等于实际利用率始终为 100%。
- 网格目标/实际采样点数、Poisson depth、过滤设置、可见性裁剪、F-score阈值及单位。
- 官方raw和audited评估的区别，异常帧及排除规则，不能在得到结果后选择性删差帧。
- 成功、失败或资源等待的依据；任务进程与恢复信息的公开脱敏版本。

## 3. 和论文对比

VR Room 的 Table 4 完整配置参考：PSNR 26.8 dB，SSIM 0.879，F-score 87.1%。这些值来自原论文，不是本项目已得结果。

比较前先核对相同场景、输入、训练帧和评估轨迹、初始化、正常/深度先验、步数、mesh方法和F-score阈值。Table 4 未充分明确所有设置时标注不确定性，不将六场景均值或另一轨迹指标混用。报告实测差值，禁止凭空规定“已经成功”的误差容忍门槛。

单场景流程完成但指标有差距，应该报告“流程完成，指标未对齐”。使用已修改算法、分辨率或不同网格方法时，报告偏差及原因。当前历史候选并非作者认证的论文提交。

## 4. 大文件与存储

完整 checkpoint、完整渲染、Poisson mesh、裁剪mesh和原始数组保存在本项目 `.local/runs/<run-id>/`。环境、缓存和数据集放各自 `.local` 子目录。普通Git不存这些大文件。

必须为重要产物计算 sha256，记录实际字节数与项目相对路径。可存储状态包括 `local_only`、`uploading`、`uploaded_verified`、`upload_failed`；不要把没有实际上传验证的文件标成远程备份。

若用户已有本项目配置好的外部大文件目的地，使用其既定方式继续上传；否则先保留本项目文件和可复现证据，不创建新的付费存储，不擅自设置Git LFS或额外账号。训练结果先完成，上传大文件不阻挡小文件、结论和清单的Git推送。

## 5. 失败也需要可继续

记录首个实质错误、最小相关堆栈、最后成功阶段、环境变更和补丁、最后checkpoint、当前进程状态与下一条执行命令。不要把全部其他项目进程参数、环境变量或整份终端历史推上公共仓库。

全资源模式不因其他 GPU 进程活跃或利用率非零而自动排队。只有实际显存/RAM/设备等资源不足时才标记 `waiting_for_resources`，写明测得的占用、具体阻塞阶段与持久队列/恢复说明；不能仅写“其他任务优先”。不把等待标成训练失败或完成。若离开交互，必须说明队列或任务是否真的仍在运行。
