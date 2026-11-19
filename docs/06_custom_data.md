# 自有数据接入：复用已验证环境，为新场景重新优化

更新日期：2026-11-19（Asia/Shanghai）。审查基于用户仓库提交 `c747cc92d70055d5d5ceb70699e85b77faad87da` 和官方历史源码 `811f5cdf342d92d12a9d9a4be185d374c430c184`。

**当前状态：`waiting_for_dataset`。用户尚未提供本项目自有数据的目录或样例，未确认其 RGB/RGB-D 类型、内参、逐帧相机位姿、深度单位与配准、场景静动态；本文件没有声称已经转换或训练自有数据。** 不将其他机器人项目的数据或相机自动认作本项目输入。

## 1. 现在能否直接处理

已经可以开始自有数据的接入工作。VR Room 本轮运行留下了 [实际环境](../reports/environment_actual.txt)、[最终配置](../reports/vr-room-iphone-30k/config.yml)、[运行清单](../reports/vr-room-iphone-30k/run_manifest.json) 和 [阶段状态](../reports/latest_status.json)。Blackwell 环境、DN-Splatter 实现、Omnidata 权重及 Poisson 工具可复用，不必重新安装整套环境。

DN-Splatter 为每个场景优化一套高斯位置、尺度、方向、颜色和不透明度。**VR Room checkpoint 是 VR Room 的场景模型；新场景需要重新初始化并优化。** 可以复用的是代码、环境、先验网络和执行流程。已有训练场景中的新视角可以直接渲染，换到另一场景则应创建新的训练记录。

现有 [vr_room_run.py](../scripts/vr_room_run.py) 固定了 `mushroom --mode iphone`、VR Room 下载包、long/short 轨迹、Faro 参考深度和 MuSHRoom 几何评估；[vr_room_inputs.py](../scripts/vr_room_inputs.py) 与 [vr_room_evaluation.py](../scripts/vr_room_evaluation.py) 也有相同假设。**不能仅替换 `DN_SCENE_DIR`，就把它们当作通用自有数据脚本。** 应新增独立的自有场景入口，沿用正式训练和产物管理约定，保留本轮基准结果。

## 2. 根据真实输入选择路径

| 已有输入 | 接入方式 | 开始完整训练前还需完成 |
| --- | --- | --- |
| 静态多视角 RGB-D、内参、逐帧相机位姿；深度已配准且尺度明确 | `normal-nerfstudio` + 传感器深度 + Omnidata 法线 | 文件转换、固定划分、初始化点云及几帧几何核对 |
| RGB-D 和内参，没有逐帧位姿 | 先通过 RGB-D SLAM/VIO/SfM 恢复相机轨迹，再使用上述路径 | 位姿估计、时间对应、坐标和尺度核对 |
| 多视角 RGB 或移动相机视频，没有传感器深度 | COLMAP + `coolermap`；可加入对齐至 SfM 的单目深度及 Omnidata | COLMAP 重建、法线/单目深度准备及对应分支的实际兼容性验证 |
| 已有 COLMAP 模型与 RGB | `coolermap` | 核对注册帧、内参顺序、稀疏点云、需要的深度/法线及尺度 |
| 单张图，或固定机位下大量重复帧 | 不能作为当前多视角三维重建协议直接使用 | 补充具有空间视角变化和重叠的采集 |
| 背景静态，少量行人或机械臂运动 | 静态区域可作为目标；必要时剔除受影响帧或提供一致掩膜 | 明确哪些表面属于重建目标，验证掩膜和有效深度 |
| 重建主体持续移动、形变或被操作 | 当前固定静态场景模型不适合直接描述整个过程 | 单独设计静态片段、物体坐标系或动态重建方法；不能宣称现有方法已支持动态场景 |

RGB-only 路径在历史源码中存在，但当前成功运行的证据来自 sensor RGB-D 的 MuSHRoom 分支。不要把它写成已经验证的 RGB-only 完整流程。COLMAP 的尺度未必是米；将单目深度对齐至 SfM 可以统一重建尺度，并不自动取得物理尺度。[S1][S3][S5]

## 3. 最少需要用户提供什么

请用户提供 **服务器数据目录，或者几组代表性 RGB/深度文件与已有标定、位姿文件**。如果用户不清楚格式，先给目录结构也足够开始盘点。执行器从真实文件中读取能确认的信息，再针对缺失项询问，不要求用户先自行填写全部技术细节。

只读盘点用户指定的目录，生成以下事实记录：

| 信息 | 记录内容 |
| --- | --- |
| 采集与范围 | 场景名称、静态/动态、图像数量、分辨率、相机是否移动、可用时间戳 |
| RGB | 原始图像路径、畸变是否已经校正、文件名与帧时间对应关系 |
| 深度 | 是否存在；原始数据类型、单位、无效值、轴向 Z 深度或沿射线距离；与 RGB 的配准情况 |
| 内参 | `fx/fy/cx/cy`、对应图像尺寸、相机模型和畸变参数；是否逐帧变化 |
| 位姿 | 是否逐帧存在；c2w/w2c、相机坐标轴约定、世界坐标单位、来源与时间对应关系 |
| 参考真值 | 是否有独立几何扫描、尺寸测量或未参与训练的参考深度；没有则明确记为缺失 |

机器人机身、末端、GNSS 或 IMU 位姿不是直接可用的相机位姿。若用这些轨迹，需要相机外参和时间关系；不要将位姿列名改成 `transform_matrix` 就视为转换完成。

## 4. 首选 RGB-D 接入协议

### 4.1 建立独立工作副本

沿用已有项目 Conda 和缓存，创建 `.local/data/custom/<scene-id>/` 及 `.local/runs/<new-run-id>/`。原始采集只读保留；预处理、法线和初始化点云写入本项目工作副本。用户数据未提供前不生成假的标定、位姿或训练完成记录。

推荐文件组织为 `images/`、`depth/`、`normals_from_pretrain/`、`transforms.json` 和训练帧初始化点云。各图像使用统一补零编号，例如 `frame_00001`，避免图像与法线的排序不同。

### 4.2 构造真实的 transforms.json

| 字段 | 要求 |
| --- | --- |
| `camera_model` | 与输入实际相机模型一致；校正后的针孔图像使用相应针孔模型和正确内参 |
| `fl_x`, `fl_y`, `cx`, `cy`, `w`, `h` | 来自真实标定，匹配当前 RGB 尺寸；变化时逐帧提供 |
| 畸变字段 | 使用真实畸变参数，或在完整校正 RGB/深度后记录校正后的模型，不能只把参数置零 |
| `frames[].file_path` | 当前场景目录内的 RGB 路径 |
| `frames[].depth_file_path` | 与该 RGB 对应的有效深度路径；此路径需要在所有 RGB-D 帧中存在 |
| `frames[].transform_matrix` | 4×4 的相机到世界变换，采用 OpenGL/NeRF 相机轴约定 |
| `ply_file_path` | 由允许的训练帧建立的初始化点云路径；颜色与点坐标须对应 |
| `train_filenames`, `val_filenames`, `test_filenames` | 固定帧列表，路径与 `file_path` 一致；一旦使用显式划分，提供运行所需的所有 split |

位姿的平移、深度和点云应使用相同世界尺度。Nerfstudio 的相机轴是 X 向右、Y 向上、相机朝向 −Z。已有 OpenCV c2w 可右乘 `diag(1,-1,-1,1)` 改变相机轴约定；若原文件是 w2c，先求逆。记录转换过程，并通过实际反投影验证，不凭格式名称推断坐标正确。[S1][S6]

优先保留真实米制坐标。初次接入显式使用 `auto_scale_poses=False`、`scale_factor=1`、`orientation_method=none` 和 `center_method=none`，防止默认归一化造成尺度混淆。需要变换时，统一变换位姿、点云及评价真值并记录变换。

毫米深度设置 `depth_unit_scale_factor=0.001`，米制深度设置为 `1`；历史 `GDataset` 实际乘数是 `depth_unit_scale_factor × dataparser_scale`。输入必须是此分支使用的轴向 Z 深度；沿射线距离应先正确转换。按传感器定义处理无效值，并记录转换前后的有效像素比例。[S2]

**将深度 resize 到 RGB 分辨率不等于完成 RGB-D 配准。** 应确认深度像素已投影到 RGB 成像平面；相机分辨率、裁剪或去畸变变化后，相关内参和深度也须一致。历史代码中的 resize 不能代替两相机之间的几何重投影。[S2][S6]

训练、验证、测试划分在优化前确定。优先保留独立轨迹或不同视角的测试帧；同一短轨迹的相邻留出帧与独立轨迹指标分别说明，不混称泛化结果。初始化 PLY 只使用训练帧，不能用留出帧或评价几何真值辅助初始化。

### 4.3 生成 Omnidata 法线

`normal-nerfstudio` 读取 `normals_from_pretrain/*.png`，不会像 `mushroom` 一样自动补生成法线。下面命令已对照固定历史源码的 CLI 字段；执行前须将 `DN_CUSTOM_SCENE` 设置为上述真实工作副本，并激活已有项目环境。[S1][S4]

```bash
cd "$DN_SOURCE_DIR"
python dn_splatter/scripts/normals_from_pretrain.py \
  --data-dir "$DN_CUSTOM_SCENE" \
  --transforms-name transforms.json \
  --model-type omnidata \
  --resolution low \
  --save-path "$DN_CUSTOM_SCENE/normals_from_pretrain"
```

复用本轮已经处理好的 Omnidata checkpoint 加载兼容补丁及权重。`low` 是先验网络的输入处理模式，不表示将训练 RGB 任意改为 384×384。生成后逐帧核对文件名、尺寸和法线方向。

### 4.4 做一次真正影响结果的输入检查，然后完整训练

选择 **5–10 帧**覆盖近远距离、相机移动及主要表面，完成：

1. RGB、深度、法线和位姿按同一帧对应，检查实测尺寸与内参。
2. 深度转为真实单位，反投影到世界坐标，叠合相邻及间隔视角，查看表面、尺度和方向是否一致。
3. 检查训练/测试划分、初始化点云来源及实际点数，确认损失所需输入存在。

这些检查通过后直接开展独立的新场景 **30,000 次更新**，不额外堆叠短训练。保留现有 checkpoint 保存与累计恢复规则，目标最后 step 为 29,999；禁止加载 VR Room checkpoint 当作新场景初始化。

训练配置应从本轮 [实际配置](../reports/vr-room-iphone-30k/config.yml) 的模型和优化器部分生成，改用 `normal-nerfstudio`，并填写真实场景路径、显式划分、深度单位和已核对的初始化信息。首个 RGB-D 实验继续采用传感器深度权重 0.2、Omnidata `mono` 法线监督及本轮记录的历史实现行为。不要把 `mushroom` 专有的 `--mode`、`--normals-from`、`--use-faro-scanner-*` 等参数搬进这个 parser。

训练入口使用哪些 parser 参数，需在已安装环境中通过 `ns-train dn-splatter normal-nerfstudio --help` 核对。`normal-nerfstudio` 本身没有 `load_depths` / `depth_mode` 配置开关，不能把 MuSHRoom 命令整段替换名字后运行。帮助检查用于核实真实 CLI，不启动额外训练。[S1][S2]

用户已经授权使用实例全部 CPU/GPU。仍按阶段安排 CPU 并发、保持单场景主训练流程，并防止多个内部线程池相乘。使用项目独立 tmux 或等效持久进程，不设训练完成期限，GitHub 操作优先 AutoDL 官方学术加速。

## 5. 历史 parser 和模型的具体注意事项

| 核查发现 | 自有数据入口应如何处理 |
| --- | --- |
| `normal-nerfstudio` 中深度列表可能为 `None`，而 `GDataset` 默认加载 sensor 深度 | 无传感器深度时不要只关模型 depth loss 后沿用该入口；改走经过核实的 RGB-only parser 或明确修复其 metadata |
| `normal_file_path` 不被该 parser 读取，法线来自固定目录 | 显式生成 `normals_from_pretrain/*.png`，不要只在 JSON 里添加一个未被使用的字段 |
| 图像按字典序，法线按自然序排序，当前断言只检查 stem 是否存在于集合 | 文件名统一补零，并检查逐项对应，而非仅检查总数或名称集合 |
| 默认 `load_pcd_normals=True`，无可加载点云时会访问不存在的 `points3D_xyz` | 提供训练帧初始化 PLY；如明确关闭该选项，记录初始化变化，不能静默退成随机初始化 |
| 点云法线在已变换点云上估计后又乘 transform，且用齐次分量 1 | 初次米制接入使用恒等 parser 变换；需额外旋转/平移时审查此处并记录最小修复，法线不应附加平移 |
| `normal_supervision=depth` 从 `depth_out.detach()` 计算目标法线 | 它不等同于传感器 GT 法线监督；首轮自有数据沿用 `mono` + Omnidata |
| 传感器深度损失使用 `depth > depth_tolerance`，默认阈值 0.1 | 核对该阈值所对应的真实尺度；无穷值和传感器特殊大值需按真实定义处理 |
| `coolermap` 读取 `mono_depth/*_aligned.npy`，对已有但不完整目录不保证自动补齐 | RGB-only 入口显式准备并核对深度文件，使用 mono 的权重设置，不照搬 sensor 权重 |

以上内容根据固定历史文件静态核查，不代表所有自有数据分支已经在 RTX PRO 6000 上运行过。[S1]–[S5]

## 6. 没有 Faro 或网格真值时如何评价

自有数据不要求必须拥有 Faro。评价可以按实际条件分层：

| 可用参考 | 可以交付 | 不能据此宣称 |
| --- | --- | --- |
| 留出 RGB 图像与相机位姿 | 留出 PSNR/SSIM/LPIPS，固定视角与连续视角渲染 | 仅凭图像指标证明表面几何准确 |
| 留出 RGB-D | 上述图像指标，加传感器深度一致性；说明有效像素、单位和统计方式 | 将传感器深度一致性当作独立高精度几何真值 |
| 独立参考网格/点云，且尺度和坐标已配准 | 按预先确定的范围、可见性、采样和距离阈值计算几何指标 | 未配准或不同可见区域下直接套用论文的 F-score 数字 |
| 只有少量真实尺寸测量 | 对指定尺寸报告重建误差，补充网格局部检查 | 外推成全场景统一精度保证 |
| 没有几何真值 | 完整网格、代表局部、覆盖与孔洞/漂浮物/薄结构检查；明确几何精度未定量验证 | 伪造 Faro、Chamfer 或 F-score |

`ns-eval` 和 `gs-mesh dn` 可以沿用通用入口，但 `eval_mesh_mushroom_vis_cull.py` 及本轮 Faro 审计不能直接用于任意自有数据。自有数据应使用自己的配置、划分和评价入口；没有的指标写为未提供参考，而非零。

## 7. 交付与下一步

先新增参数化的自有数据转换/运行入口，输入至少包括真实场景目录、parser 类型和新 run-id，不修改已完成的 VR Room 运行。转换脚本应保存 RGB/深度/法线/位姿映射、标定来源、单位、所有坐标变换、train/val/test 清单、初始化来源及实际点数。

首个自有场景完成后，交付完整 30k checkpoint 清单、实际配置、留出评价、RGB/深度/法线代表图、Poisson 网格和产物哈希；存在独立几何真值时再报告相应几何指标。所有真实运行结论写入新的 run 目录，不覆盖首轮基准数据。

**当前可立即执行的下一步是获取用户指定目录或样例，确认输入属于第 2 节哪一条路径。** 不必先完成全文所有数据集再做接入，也不在输入尚缺时凭空启动“自有数据训练”。

## 来源

- [S1：历史 normal-nerfstudio parser](https://github.com/maturk/dn-splatter/blob/811f5cdf342d92d12a9d9a4be185d374c430c184/dn_splatter/data/normal_nerfstudio.py)
- [S2：历史 GDataset 深度、法线加载](https://github.com/maturk/dn-splatter/blob/811f5cdf342d92d12a9d9a4be185d374c430c184/dn_splatter/data/dn_dataset.py)
- [S3：历史 coolermap parser](https://github.com/maturk/dn-splatter/blob/811f5cdf342d92d12a9d9a4be185d374c430c184/dn_splatter/data/coolermap_dataparser.py)
- [S4：历史 Omnidata/DSINE 法线预计算入口](https://github.com/maturk/dn-splatter/blob/811f5cdf342d92d12a9d9a4be185d374c430c184/dn_splatter/scripts/normals_from_pretrain.py)
- [S5：历史模型损失与逐场景参数](https://github.com/maturk/dn-splatter/blob/811f5cdf342d92d12a9d9a4be185d374c430c184/dn_splatter/dn_model.py)
- [S6：Nerfstudio 官方数据和坐标规范](https://docs.nerf.studio/quickstart/data_conventions.html)
- [历史 README 的 Custom Data/COLMAP 说明](https://github.com/maturk/dn-splatter/blob/811f5cdf342d92d12a9d9a4be185d374c430c184/README.md)
- [本轮实际运行证据的固定提交](https://github.com/kaiwen123-yang/DN-Splatter/tree/c747cc92d70055d5d5ceb70699e85b77faad87da/reports/vr-room-iphone-30k)
