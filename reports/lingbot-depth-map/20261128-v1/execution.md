# 大厅离线实验的执行与恢复

运行目录是 `/root/autodl-tmp/DN-Splatter/.local/runs/lingbot-depth-map-20261128-v1`。各阶段日志和退出码分别保存在 `logs/` 和 `state/`。用 `artifacts.json` 定位视频、点云与逐帧预测索引。

## 数据和环境恢复

本机 main 从 `1f2b487` 快进到 `aa979ae`，两个既有未跟踪文件留在原工作区。新实例的数据盘为空。用 H 盘 11 月 25 日归档的冻结清单和内容哈希恢复了 B2 相机、划分、配准参数、1,846 张 RGB 和 1,846 张全距离配准 Z。恢复脚本逐个验证源文件和传输字节。将全距离 NPY 按 B2 历史导出规则置零，1,846 个生成文件的 SHA-256 全部与 B2 的训练清单相同。全距离输入另保留 282,027,099 个区间外读数。所有正式参考分母使用 0.3–6 m。

服务器首次通过 AutoDL 学术加速克隆 main 时传输中断。Git 清理了本次克隆目录，包括刚创建的两个环境前缀。随后从已核验的本机 main 传输 Git bundle，重新建立并验证项目专属的两个 Conda 前缀，两套官方源码也按固定 Git 对象身份检出。输入和权重在独立恢复目录中，完整性通过哈希核对。PyPI 直连下载 OpenCV 超时后复用了项目 pip 缓存中的完整 wheel。GitHub 和 Hugging Face 加速仅作用于项目子 shell。权重使用作者提供的 ModelScope 发布，实际文件与固定提供方 revision 的 Git LFS 指针逐项核对。Map 的官方下载器另外验证了发布配置中的 SHA-256。

## 运行与适配

正式 Depth 官方例 0 初次调用遇到 Hopper FlashAttention 3 内核的 CUDA invalid argument。仅在 SM120 进程中关闭 xFormers 的 FA3 分派，实际调用 FA2，保留 BlockDiagonalMask 和默认深度 masking。例 0 成功输出 303,593 个有限正点。另取同一示例首层 8 个查询，将 8,192 个输出标量与禁用 autocast 后的 FP32 分块注意力比较，最大绝对差 0.003777，RMS 0.0003804。原失败的 Hopper 实现无法提供整网数值对照。

30 个样例按训练时间段、深度支持和边缘选取，包括 B3 预定的桌沿、椅腿、柱面和泡沫齿边。散点与矩形均在推理前保存输入遮盖与独立计分掩码。全训练集改变量、固定遮挡恢复、两个其他视角共同支持的孔洞分开评估。近邻投影用首个正 Z，投影在前、相符与可能被遮挡的位置分别计数。

Map 三个序列使用独立进程，每次完整读取原 image-list。正式调用官方 infer.py，显式传入发布 checkpoint 和 config。分段计时在同步 GPU 后记录输入、拷贝、网络调用和保存时间。首 8 帧批量返回，后续按 raw_indices 逐项核对。时间图保留原采集时间戳；视频使用 5 Hz 观看时钟。模型输出保留原尺度。对齐使用 32 张预先选定的训练图，每个序列重新拟合自身的单一尺度与刚体变换。

本次利用率还受 CPU 读图和串行到帧影响。短序列稳定阶段中位数为每帧输入处理 28.9 ms、同步模型步骤 32.5 ms。Depth 的大厅中位 GPU 推理时间为 55.1 ms，端到端为 96.2 ms。未设 GPU 算力或显存配额。已完成序列的评估和成片与下一序列并行。实例驱动提供 NVENC API 13.0，环境中的 FFmpeg 需要 13.1，实际视频以 libx264 编码。全量指标、点云和视频生成后，查看器与回放服务继续在 tmux 中运行。

## 可离线播放的大厅全程录像

四画面录像采用完整 1,846 帧探索序列。上排是实拍 RGB 和截至该帧的累积三维点云，下排是共用 0.3–6 m 色标的传感器深度与 Map 深度。30 个既有遮挡位置按原时间戳保留。06:09 起为完整点云的 10 秒旋转回看，共展示 4,266,403 个分数高于 1.5 的显示采样点。成片为 1920×1080 H.264 MP4，时长 6 分 19.075 秒。[录像的服务器路径、文件大小、SHA-256 与逐帧解码核验](hall_recording.json)

录像使用 Map 环境、单独安装的 PyAV 15.1.0 和该版本支持的 NVENC 接口。CUDA 深度缓冲为同一屏幕位置选择最近点，编码使用 NVIDIA H.264 NVENC。可复用入口为 `scripts/lingbot_record_hall.py`，`labels` 生成中文画面标签，`render` 从冻结结果直接生成并验证 MP4。

Windows 上的成片已与服务器 SHA-256 比对，本地解码器读完全部 2,096 帧。北京时间 2026-11-28 16:33 已按用户要求关闭 `hall-viewer`、`hall-replays` 及对应端口转发。离线 MP4 可直接播放。

## 断点与查看

从服务器项目根目录执行。已有 Depth 帧按身份和 SHA-256 复用。Map 遇到中断时以完整原列表建立新的 attempt，保留原输出、8 帧前缀和 Uniform320 选择。

```bash
cd /root/autodl-tmp/DN-Splatter
bash scripts/launch_lingbot_depth_map.sh depth
bash scripts/launch_lingbot_depth_map.sh map
bash scripts/finish_lingbot_delivery.sh
```

当前实例的 tmux 可执行文件为 `/root/autodl-tmp/lingbot-20261128-bootstrap/bin/tmux`。使用 `-L lingbot ls` 查看本项目会话。重新建立环境可用 `bash scripts/bootstrap_lingbot_env.sh depth` 和 `bash scripts/bootstrap_lingbot_env.sh map2`。输入恢复脚本的 send 子命令接受归档目录、项目专属接收目录和用户已有的 SSH 客户端。

官方三维查看器读取同一批已落盘的预测，默认训练序列。服务器用 `bash scripts/view_lingbot_map.sh full_train_1478 18788` 启动。视频服务用 `python scripts/serve_lingbot_media.py` 启动。将服务器回环地址的 18788 和 18789 转发到本机同号端口即可旋转点云、播放视频。服务运行时可打开[三维查看器](http://127.0.0.1:18788/)和[训练序列完整视频](http://127.0.0.1:18789/full_train_1478.mp4)。
