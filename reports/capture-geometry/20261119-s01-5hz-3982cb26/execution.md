# 首段执行记录

运行身份为 `20261119-s01-5hz-3982cb26`。本地工作区从 `a13bb40` 快进到 `d2887a3`，服务器工作区从 VR Room 结果提交快进到同一计划提交。Windows 本机 24 个逻辑 CPU，约 31.4 GiB 可见内存，盘点时约 8.8 GiB 可用。选帧使用一个进程并按块顺序读取，OpenCV 设 1 线程。[版本、脚本指纹与资源](versions.json)

## Windows

工作目录为 `C:\Users\ykw\Documents\ChatGPT\DN-Splatter`。复用项目的 bag-audit venv、厂商消息解码和原审计的 SQLite。以下是本次运行的命令入口，变量缩短了相同路径。提取、冻结和数据发布会检查既有目录或清单，本次完整产物以清单校验接续。

```powershell
$runId = '20261119-s01-5hz-3982cb26'
$py = '.local/venv/bag-audit/Scripts/python.exe'
$runDir = ".local/capture-geometry/$runId"
& $py scripts/capture_segment.py --run $runId
# 逐张检查 screen_*.jpg 和低清晰度裁片，保存 quality_review.json 后才冻结划分
& $py scripts/capture_freeze.py "$runDir/transfer" --review "reports/capture-geometry/$runId/quality_review.json"
& $py scripts/capture_rectify.py "$runDir/transfer" "$runDir/d2c"
```

通过本项目已有的认证连接，将 `transfer/` 作为 tar 字节流从本机送到服务器本次运行的 `incoming/`，成功解包后移入 `transfer/`。本地未生成长期归档。服务器核对发送方清单的 SHA256，再枚举全部成员及其长度、摘要。Windows 原包未复制或传输。[双端核验](transfer_manifest.json)

## AutoDL

运行根目录为 `/root/autodl-tmp/DN-Splatter/.local/capture-geometry/20261119-s01-5hz-3982cb26`。原始派生包、两套网格、SfM 数据库、度量结果和持久日志分别在 `transfer/`、`rect/`、`sfm/`、`metric/`、`logs/`。新脚本先送到本次运行的 `code/`，与提交到仓库的脚本对照。复用了已有 DN 解释器中的 NumPy、OpenCV、Open3D 和历史读取器。在单独的 `.local/conda/capture-geometry` Python venv 中安装 PyCOLMAP CUDA 版及其依赖，固定 NumPy 1.26.4。原 CPU 版下载在安装前由本次任务停止，改用可用的 GPU 版本。

PyPI 官方 CUDA wheel 及依赖以 SHA256 校验后存放在本次运行的 `wheels/`。OpenCV 使用项目已有 wheel。导入时发现容器缺少 libSM 和 libICE，从 Ubuntu 软件源下载 `libsm6` 和 `libice6` 两个 deb，以 `dpkg-deb -x` 解到本次运行的 `runtime/`，仅在预处理进程设置动态库搜索路径。安装入口如下。

```bash
cd /root/autodl-tmp/DN-Splatter
run="$PWD/.local/capture-geometry/20261119-s01-5hz-3982cb26"
dn="$PWD/.local/conda/dn-splatter-bw/bin/python"
sfm="$PWD/.local/conda/capture-geometry/bin/python"
dataset="$PWD/.local/data/custom/capture-s01/20261119-s01-5hz-3982cb26"
# 以下安装命令对应本次已建好的独立环境
"$dn" -m venv .local/conda/capture-geometry
"$sfm" -m pip --disable-pip-version-check install --no-index --find-links="$run/wheels" pycolmap-cuda12==3.13.0
"$sfm" -m pip --disable-pip-version-check install --no-index --no-deps .local/cache/wheels/opencv_python-*.whl
```

实际 CPU 配额 25 核，容器内存上限 120 GiB。盘点时项目所在数据盘约 432 GiB 可用，RTX PRO 6000 Blackwell 有 97,252 MiB 可用显存。SIFT 提取和匹配日志均确认使用 GPU。特征及建图阶段在 25 个 CPU 的 affinity 内运行，PyCOLMAP 设 25 线程，BLAS 和 OpenMP 的嵌套线程数置 1。每个长阶段使用项目 tmux 会话，退出码保存在日志目录。几何和复读入口如下。

```bash
export DN_CPU_BUDGET=25
export LD_LIBRARY_PATH="$run/runtime/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR="$run/cache"
"$dn" "$run/code/capture_rectify.py" "$run/transfer" "$run/rect" --all
for stage in features match queries; do
  taskset -c 0-24 "$sfm" "$run/code/capture_sfm.py" "$run" "$stage" --device cuda
done
taskset -c 0-24 "$sfm" "$run/code/capture_sfm.py" "$run" map
"$sfm" "$run/code/capture_localize.py" "$run"
"$dn" "$run/code/capture_metric.py" "$run"
OMP_NUM_THREADS=4 "$dn" "$run/code/capture_dataset.py" export "$run" "$dataset"
OMP_NUM_THREADS=4 "$dn" "$run/code/capture_dataset.py" validate "$run" "$dataset"
```

训练数据库内只有 193 张训练图。顺序匹配使用 overlap=10、quadratic overlap，未使用词表回环。留出 RGB 的特征存放在独立的 queries.db，训练地图增量求解后才用于固定点 PnP。地图在主路径完整连通，本轮未追加映射帧，也未运行全图 SfM、穷举回退或 RGB-D odometry。为检查宽基线残差，最后复算同一批 20 对视角，并增加两边都在窗口内及投影超过 20 m 的统计，复算前后位姿文件及尺度文件的 SHA256 一致。

历史读取核验以 `normal-nerfstudio` 的 `load_normals=False`、`load_pcd_normals=False` 执行，保留 `load_3D_points=True`。使用零目标畸变、`depth_unit_scale_factor=1`、`auto_scale_poses=False`、`scale_factor=1`、`orientation_method=none`、`center_method=none`、`downscale_factor=1` 和相机分辨率系数 1。初始化点云最远坐标绝对值为 14.5934 m，据此设读取器的 `scene_scale=15`，保留米制坐标。待审查配置记录 `num_downscales=0`。上述核验可在已有数据集上单独重跑最后一条 validate 命令。[派生数据清单](dataset_manifest.json)
