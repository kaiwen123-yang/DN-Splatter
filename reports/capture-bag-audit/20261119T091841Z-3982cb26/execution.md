# 审计执行记录

此次在 Windows 11 本机的项目检出中执行。系统可见 24 个逻辑 CPU、31.43 GiB RAM。venv 位于仓库 `.local/venv/bag-audit`，只读复用已有 NumPy、OpenCV、Matplotlib 和 Pillow，并在 venv 内安装 rosbags。完整版本在 [summary.json](summary.json) 中。像素处理顺序执行，OpenCV 使用一个线程。原始盘只通过只读文件句柄访问。

## 原始执行

先用 Git 快进到审计协议提交 `73981d9`。检查了指定目录中的唯一普通文件，用 Python `Path.open('rb')` 读取 magic，并用 `hashlib.file_digest(f, 'sha256')` 从文件头顺序散列至 EOF。实测路径、字节数、mtime_ns 与散列保存在项目私有的 fingerprint.json；扫描器核对路径、字节数与 mtime_ns 后复用该次散列。正式消息扫描于 2026-11-19 09:18:41 UTC 启动。以下是本次环境及分析入口，在仓库根目录执行。

```powershell
python -m venv --system-site-packages .local/venv/bag-audit
.local/venv/bag-audit/Scripts/python.exe -m pip install --no-cache-dir rosbags
.local/venv/bag-audit/Scripts/python.exe scripts/capture_bag_scan.py --source 'H:\capture.bag\capture.bag' --private-root .local/bag-audit --fingerprint .local/bag-audit/fingerprint.json
$env:MPLCONFIGDIR = (Join-Path (Get-Location) '.local/cache/matplotlib')
.local/venv/bag-audit/Scripts/python.exe scripts/capture_bag_report.py --private .local/bag-audit/20261119T091841Z-3982cb26 --config configs/capture_bag_audit.json --output reports/capture-bag-audit
.local/venv/bag-audit/Scripts/python.exe scripts/capture_bag_report.py --private .local/bag-audit/20261119T091841Z-3982cb26 --config configs/capture_bag_audit.json --output reports/capture-bag-audit --review .local/bag-audit/20261119T091841Z-3982cb26/review.json
```

先生成数字结果和联系表，再逐格复核，最后将观察与该 bag 的 SHA-256 绑定写入报告。完整的每条消息时间表、厂商帧号、连接、物理块与偏移在 `.local/bag-audit/<run-id>/times.sqlite`。源码定义和序列化顺序按固定的 Orbbec SDK 参考提交核对；未用 SDK 实时播放接口。静态资料中的设备序列号、网络字段等没有导出。[厂商读写实现](https://github.com/orbbec/OrbbecSDK_v2/tree/f5d59930ccb9a189f200e58fba4e2ae42664792d/src/media/ros)

## 对同一原件独立复查

新执行时省略 `--fingerprint` 即会从原件重新计算 SHA-256 并创建新的 run-id。将报告命令的 `--private` 指向新目录。要复用本次已发布的场景观察与遮挡范围，可将 `--review` 指向此报告目录，使用单独的 `--output` 做比对。该审计入口核对 Gemini 335 原生流布局、深度属性与标定；其他型号或数据模式需按其真实定义复核。脚本内容的 LF 规范化 SHA-256 已随结构化报告保存。
