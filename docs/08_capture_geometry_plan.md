# capture.bag 下一阶段：配准、相机轨迹与训练数据准备

日期：2026-11-19（Asia/Shanghai）。依据为用户实际审计提交 `a13bb40445c1c049c2931094e533a5ed43472297`。**本文件是下一阶段执行计划，不是已完成几何重建或已完成训练的记录。**

## 1. 已知事实与本轮终点

原始文件实际位于 Windows `H:\capture.bag\capture.bag`；上一层 `H:\capture.bag` 是目录。文件 SHA-256 为 `3982cb269ea03ddf4234b45f74be27143eb66bcd807e4e471d0f3a9cd6b03b59`，大小 5,919,666,852 字节。数据来自 Orbbec Gemini 335，包含原生 RGB、毫米轴向 Z 深度、双相机 SDK 标定、加速度和角速度，未发现逐帧世界相机位姿。[A1]

原始 RGB 为 1280×720，深度为 848×480，硬件 D2C 关闭；现有深度不能仅 resize 后作为 RGB 网格深度使用。时间同步审计已找到 10 ms 内的一对一 RGB-D 关联，不需要重新猜话题或尺度。原始审计、样例和复现结果保持原样。

本轮机器可读计划为 [capture_geometry_protocol.json](../configs/capture_geometry_protocol.json)，执行入口为 [本地导出提示词](../prompts/local_capture_geometry.md) 和 [AutoDL 几何准备提示词](../prompts/autodl_capture_geometry.md)。输入目录、阶段与报告字段按这份协议统一，不用过去的“等待首次 bag 审计”状态替代本轮状态。

**本轮做到：有证据的同帧 D2C、一个短片段的相机轨迹和米制尺度、通过几何核对的派生数据集，以及完整回传报告。到此等待报告审查，不自动运行 Omnidata、30k 训练、Poisson 或新的基准实验。** 正在运行的 AutoDL 任务保持原状态。

执行顺序：

1. 利用已回传的 4 组原数值样例核对同帧 D2C。
2. 在 Windows 本地从指定原包按索引只读导出首段约 5 Hz 的配对帧。
3. 将带哈希清单的精简输入目录传至已有 AutoDL 项目。
4. AutoDL 用共享固定内参的 RGB SfM 恢复相机姿态，以训练帧传感器深度求单一全局米制尺度。
5. 做相邻和跨时间的几何核对，生成与固定 DN-Splatter 历史实现一致的数据布局并回传。

这一路线是针对本采集的工程选择，不是论文宣称已经验证的 Orbbec 数据协议。后续新场景需要重新优化高斯；VR Room checkpoint 不是该采集的初始场景模型。

## 2. 执行位置和资源

### 本地

复用本次审计的独立 Python 环境、`capture_bag_scan.py`/`capture_bag_samples.py` 和已生成的 `times.sqlite`。先确认其源路径、源 SHA-256、字节数、mtime 和审计 run 身份；不递归扫描其他盘来猜输入，不全量复制或重新审计约 6GB 的原包。已有索引确实不可用时，才按同一原包重新建立必要索引并记录原因。

新增导出和几何转换脚本放受 Git 跟踪的 `scripts/`；大数据、中间索引和日志放本项目 `.local/`。本机资源以现场读取为准，不把 AutoDL 的 25 核/120GB 当作 Windows 配置。

### AutoDL

先读取真实项目工作区与现有后续任务。复用已验证的 DN 训练环境，不向它随意安装 COLMAP/Open3D 及相互冲突的依赖。若现场没有合适的预处理工具，建立单独的项目预处理环境或使用已有兼容二进制，并记录版本、编译配置和命令。

用户允许使用分配给实例的全部 CPU 和 GPU；CPU 预算按实际 quota/affinity 确认，避免进程数与库内部线程数相乘。GPU SIFT/匹配在实际可用时使用。Blackwell 上若预处理二进制的 CUDA 后端不兼容，先将该阶段切换 CPU，不修改驱动、共享 CUDA 链接或 DN 已验证环境来解决 SIFT。CPU 路径也能完成这个几百图规模的 SfM。

AutoDL 上 GitHub 操作仍优先在本项目子 shell 使用官方学术加速；Windows 使用本机现有连接，不拷贝 AutoDL 代理到全局配置。长任务保存阶段记录并可接续，不设完成期限、不自动关机、不结束其他项目进程。

## 3. 先做同帧 D2C：不需要世界轨迹

已发布的原数值样例为 `pair_01`、`pair_13`、`pair_24`、`pair_25`，原生 JPEG 和 uint16 深度均在 [审计目录](../reports/capture-bag-audit/20261119T091841Z-3982cb26/) 中。先检查这些样例及哈希；本次审查若已提供对应 D2C 诊断，复用其代码与结果，现场仅核对实际数据接口。

本次已新增 [4 样例数值检查](../reports/capture-bag-review/2026-11-19/numeric_review.json) 与 [投影预览](../reports/capture-bag-review/2026-11-19/sample_projection_review.jpg)，可复用 `scripts/capture_registration_review.py`。原生 1280 网格单点 z-buffer 的覆盖约 32.76%–34.67%，640×360 去畸变辅助网格约 90.67%–95.97%；这主要反映投影采样密度与网格选择，不能直接解释为标定优劣。naive resize 与标定投影的像素位置差中位数约 37.03–39.90 px，忽略 RGB 畸变的投影差 P95 约 7.12–7.42 px。**这些是算法投影位置之差，不是相对真值的标定残差；4 个样例也不能代替整个首段的配准验收。**

### 3.1 已核实的参数含义

SDK 保存的外参为深度到彩色，旋转按 row-major 解释：

`p_C = R_(C<-D) @ p_D + t_(C<-D)`。

平移以米使用：

`[-0.013604243278503418, 0.00016550984978675842, -0.0017640596628189086]`。

平移范数约 13.719 mm，旋转约 0.462°。SDK 原平移数组以毫米保存，不能与已经换成米的深度点直接相加。其方向与排列已有 Writer、Reader 和坐标变换实现支持，不凭视觉效果反复尝试转置或取逆来挑“看起来更好”的版本。[S1][S2][S3]

深度 profile 畸变为零；RGB 是 Brown-Conrady K6，SDK 数组顺序是 `k1,k2,k3,k4,k5,k6,p1,p2`。OpenCV 使用 `k1,k2,p1,p2,k3,k4,k5,k6`，必须显式重排。本采集 k4/k5/k6 为零，k3 非零，不能丢掉 k3。每个原深度计数为 1 mm，代表光轴 Z；零值和 65535 不作为有效测量。[A2][S1][S3]

### 3.2 固定一种目标图像网格

本轮默认用**去畸变后的 1280×720 RGB 针孔网格**作为 SfM 和后续 DN 输入。保存原始 `K_cv,D_sdk,D_cv`、目标 `K'_cv`、图像尺寸及所有裁剪/缩放参数。可以保持原尺寸和原 `K_cv` 作为去畸变目标内参，以减少额外裁剪；若选择新的 `K'`，必须让全部相机、图像、深度和像素采样使用同一记录。

将每个原生深度像素反投影成米制 `p_D`，经记录外参转换成 `p_C`，再投影到目标无畸变 RGB 网格。目标像素保存 **`p_C.z`**，不能原样复制 `p_D.z`，也不能改成沿射线距离。目标像素冲突使用最近表面的 z-buffer。

原始深度约 40.7 万像素，RGB 约 92.2 万像素；仅按像素中心投影自然会出现空洞。区分“没有原观测支持”与“错误配准”，不把空洞比例直接当作标定误差。可评估遵守遮挡和深度边界的像素 footprint 覆盖，但不要跨物体边缘插值或为提高覆盖率涂满深度。基准输出优先保留无覆盖像素为无效值。

**SDK 的两个实现细节需显式处理：**

- 固定版本 `Align.cpp` 的 `TargetDistortion` 默认是 0；它生成的目标深度可以是无畸变网格，而原生 RGB 仍带畸变。不能只调用默认 Align 就认定两个输出像素网格相同。
- 同版本 `CoordinateUtil.cpp` 的某个投影 helper 仅在 Brown enum 3 分支应用畸变；本 RGB 为 K6 enum 4。真正 `AlignImpl.cpp` 支持 K6。选择匹配实现或显式 Brown/OpenCV 投影，不能把不匹配的 helper 作为数值金标准。[S3][S4]

### 3.3 像素中心约定只转换一次

SDK/OpenCV 几何计算使用整数坐标作为像素中心。COLMAP/Nerfstudio 的相应相机约定使用角点原点、中心位于半像素处，手动导出时按本项目源码审查做一次主点转换：

`cx_colmap = cx_cv + 0.5`，`cy_colmap = cy_cv + 0.5`。

保留分别命名的 `K_cv` 与 `K_colmap/ns`，不要因为先经过 COLMAP 再导出 Nerfstudio 而重复加 0.5。几何核对时使用各自一致的像素坐标；COLMAP 特征坐标回到 OpenCV 图像采样也要按同一约定转换。

若为 Open3D 或诊断另建 640×360 辅助图像，OpenCV resize 对应：

`fx' = sx*fx`，`fy' = sy*fy`，
`cx'_cv = sx*(cx_cv+0.5)-0.5`，`cy'_cv = sy*(cy_cv+0.5)-0.5`。

随后才做该辅助网格的半像素约定转换。不能将 640 网格的内参复制给 1280 图像。

### 3.4 D2C 交付

保存同一视野的去畸变 RGB、有效深度覆盖、深度边界叠加、若干前景边缘放大图；说明遮挡、无效值与采样密度。输出目标有效比例、投影越界/后方/冲突计数，并对可明确对应的近物边缘给出实际残差或可复查标记。

公式一致、矩阵正交和程序读回通过仅说明实现与记录自洽；没有独立距离真值时不称为深度精度认证。没有世界位姿不阻止上述同帧 D2C，跨时间几何核对留到轨迹恢复后。

## 4. 本地按索引导出首段，先取约 5 Hz

首段的边界必须来自 `samples_metadata.json` 的 #01/#04，不能把舍入显示值、厂商帧号和零基数组索引混用：[A3]

| 边界 | RGB 设备/header 纳秒 | RGB 零基 index | 厂商 frame number | 相对首张深度 |
| --- | --- | --- | --- | --- |
| 起点 #01 | `2427228619000` | 0 | 1 | 0.465019 s |
| 终点 #04 | `2475353026000` | 1442 | 1443 | 48.589426 s |

这段约 48.124407 s。默认以起点加 `n × 200000000 ns` 生成不超过终点的时间格，约 241 个目标时刻；选择最近的可用 RGB 且不重复。若改为固定含两端的均匀采样，记录真实间隔和选择规则，不能仍声称精确 5 Hz。

对选中的 RGB 复用审计的一对一深度关联，保留原时差；缺匹配或冲突时报告并按既有索引寻找候选，禁止无说明重复使用同一深度帧。不必先导出完整 30 Hz 第一段，更不导出全部 11,041 帧后再删减。

`times.sqlite` 已有 connection/index/chunk/offset 等身份，可按必要 chunk/offset 读取选中 payload。按 chunk 分组顺序读取，避免每取一帧就重复扫描原包。源数据开只读句柄，导出结束复查大小/mtime。

精简输入目录至少保存：

- 选中帧原始 JPEG payload；
- 配对的原生 uint16 深度，可用无损 NPZ；保留原 0/65535，此时还不是 DN 的训练深度；
- 对应 SDK 标定、设备单位依据、完整选帧表和唯一同步关系；
- 源 SHA-256、每个导出文件 SHA-256/字节数、脚本版本、精确时间窗、抽样规则和环境版本。

选帧表同时记录 source connection、零基 index、vendor frame number、record/header/device 时间字符串，以及派生的统一补零文件名。`frame_00000` 只是本次导出编号，不能覆盖原始身份。

先按事先声明的图像质量与同步规则筛选候选，记录被剔除帧和依据；随后固定 train/val/test，再运行 SfM、估计尺度和生成初始化点云。默认将时间排序后的选帧零基序号模 10：0–7 为 train，8 为 val，9 为 test。尾部使用同一规则。实际数量以选帧表为准，不预填成功帧数。划分固定后不能因为 pose 失败把留出帧改为训练，或反复换掉较难测试帧。

相邻时间留出是同轨迹评价，不包装为独立采集的泛化测试。本轮主线用训练 RGB 建 SfM 地图、训练深度定尺度，再在固定地图上定位留出 RGB，禁止留出图像参与重建地图、BA、尺度或初始化。

## 5. 传输的是精简输入目录

将导出目录放本项目 `.local/` 下，以 manifest 识别版本；“输入包”指带清单的文件集合，不要求新增长期 Windows 归档或上传原始 bag。尽量复用已有目录传输方式；仅当现有传输渠道需要时才做临时打包。

优先使用用户已经配置且获授权的本地→AutoDL 连接及真实目标项目目录。执行前确认连接目标与目的目录，传输后按完整 manifest 校验文件数、字节和哈希，只有匹配才记为 `transferred_verified`。不猜 SSH 主机、端口、密码或实例身份。

如果本地会话没有可用的 AutoDL 连接，完成本地导出、清单和小型报告即可；向用户返回精确本地目录、总大小、manifest 哈希以及现场实际缺少的连接信息。不要因不能传输而改为把大数据塞入普通 Git，也不要要求上传约 6GB 原包到 ChatGPT。后续 AutoDL Codex 从用户真实提供的位置接续。

普通 Git 只保存脚本、配置、缩略图、选帧/同步/划分清单、版本/哈希和结论。大图像集、完整模型和点云留项目数据区；日志说明是否已传输，不能将“本地准备完成”写成“服务器已收到”。

## 6. AutoDL 主线：已标定 RGB SfM

### 6.1 固定相机模型，先解决轨迹

对精简输入的选中帧批量执行与第 3 节完全相同的去畸变和 D2C，保存固定目标网格。**仅用已经固定的训练 RGB 建主 SfM 地图**，使用同一共享 PINHOLE 相机、已知 `fx/fy/cx/cy`。全部图像来自同一录制的固定 profile；不让每幅图自由优化一套内参，也不把 factory 标定丢弃后从任意焦距重新估计。

核对现场 COLMAP 版本和命令帮助，明确关闭 BA 对 focal length、principal point、extra parameters 的优化，固定内参；名称以真实版本支持的选项为准。提交最终使用的配置和命令，不照抄过时 CLI。稀疏点与相机外参可以优化。

约几百幅图可以使用 exhaustive matching，以相机重访和非相邻观测帮助形成一个模型；若使用 sequential matching，必须保存邻接窗口和闭环候选策略，不能只连接相邻一对并忽略回访。GPU 特征后端失败时使用兼容 CPU 路径，不借此更换 DN 训练栈。[S5]

输出相机注册率、连通分量、未注册帧清单、稀疏点数、track length、重投影误差分布、轨迹图及其与原始帧时间的对应。不能把多个彼此独立的 SfM 子模型拼在同一世界坐标里冒充完整轨迹。

训练地图确定后，对 val/test RGB 提取/匹配特征，仅用训练地图已有 3D 点进行 PnP/图像定位。固定训练相机、内参和三维点，不用留出图像继续三角化新点或做全图 BA。保存定位内点数、重投影误差、失败帧和训练地图前后身份；失败如实报告，不换图。如果现场实现只能做全图 SfM，则另记 `all_rgb_sfm_diagnostic` 并说明看过留出 RGB，不能冒充主协议。

### 6.2 失败时只针对实际原因推进

主线不连通或关键区段跟踪断裂时，先检查去畸变、曝光/模糊、静态纹理、重复结构误匹配和选帧间距。证据表明 5 Hz 过稀时，在同一原始片段按索引加密至 10 Hz；仍有对应证据才补全速关键区段。保留原 split 的身份与用途，为新增帧补明确标签，不把之前的留出帧改为训练来改善结果。

SfM 经具体修复仍不适合时，再启用 Open3D RGB-D odometry 与 pose graph 作为备选。使用相同已注册的针孔 RGB-D、正确米制深度和已说明的距离 mask；先核对 odometry 的相邻变换方向，再组装相机 c2w。后端需回环/全局优化和可复查的约束，不能将逐帧 ICP 连乘当成无漂移轨迹。Open3D 路线不是与 COLMAP 并行展开的第二套庞大实验，也不先安装多个 SLAM 系统。[S7]

有 IMU 数据不意味着可以直接积分出稳定姿态/位置。本轮不将全零四元数当有效姿态，不无依据估计 bias、时间偏移或手眼外参来拼装 VIO。

## 7. 用训练帧深度确定单一米制尺度

### 7.1 先将 COLMAP 位姿解释正确

COLMAP 的 `images` 外参定义为 world-to-camera。对于 `p_C = R_CW p_W + t_CW`，相机中心为 `-R_CW^T t_CW`，c2w 为其逆；四元数分量顺序按真实文件格式读取。不能直接将 w2c 的 `tvec` 填入 Nerfstudio c2w 平移。[S6]

已知内参并不使单目 RGB SfM 自动成为米制。先保留原 SfM 无量纲模型，以训练帧的传感器深度估计一个公共比例。

### 7.2 估计与核验分开

对于训练帧中可追踪的稀疏三维点，将点变换到相机，取得正的 `z_sfm`；在对应像素取得同网格的传感器 `z_sensor`。像素中心约定按第 3.3 节转换；拒绝无效/越界/遮挡边界、混合前后景和明显动态区域，记录排除数量，不能通过删除所有较大误差点隐藏问题。

以比值 `z_sensor/z_sfm` 估计稳健单一全局比例 `s`。允许中位数、稳健回归或有明确残差定义的 RANSAC；记录方法与阈值，并限制单帧/单track的大量重复观测支配拟合。要求多个训练视角和多个深度层提供支持；只凭单点、单帧或一个平面不能建立强尺度结论。

将 `s` 同时作用于 SfM 稀疏点和所有相机平移/中心，保持旋转不变。**不逐帧拟合不同尺度，不再做深度的任意 affine shift 来掩盖轨迹或标定错误。** 保存原模型、尺度值、样本数量、每帧支持数、尺度比值分布，以及未参与拟合视角上的一致性检查。

默认主线已按训练图建图、留出图在固定地图中定位执行；留出 RGB 参与自身 PnP，不参与地图优化。若因具体限制运行全图 SfM，必须单独保存并明确“地图和位姿估计见过留出 RGB”，不能混入主协议结果。两条口径都禁止用留出深度求尺度、初始化或改写训练输入。

### 7.3 远深度与局部目标分别检查

原审计的 94.04% 有效数值比例仅排除了哨兵值；合并抽样深度 P95 为 24.799 m、最大 41.848 m，不是对应距离上的精度保证。不要自动把 20m 以上数据当可靠训练监督，也不要将 3m 外全说成无效。

第一轮几何报告至少比较 `0.3–3m`、`0.3–6m`、`0.3–10m` 的保留像素/视野覆盖、尺度拟合支持和跨帧残差。这些是工程敏感性区间，不是传感器官方合格边界。根据目标桌椅/箱体的实际深度和视角覆盖选择首个数据集的派生 mask，保存原数值和 mask 规则，不能改写原始深度。

派生范围和尺度拟合筛选规则用训练帧确定并冻结，再查看留出深度上的最终一致性；不根据留出结果反复调范围来提高该批测试分数。

尺度拟合只使用训练深度。训练帧以外的深度可用于清楚标明的几何验证，不回流到比例拟合、点云初始化或掩盖失败的参数选择中。

### 7.4 多帧几何检查

检查相邻、间隔较大和重访视角的相对变换、重投影、遮挡处理与米制点云叠加，重点看桌面、箱体和柱体是否重合、是否出现双层墙/倾斜地面/轨迹跳变。保留逐对 overlap、有效数、深度残差分布和失败对，不能只展示最好的几对。

没有几何 GT 时，这些是传感器和多视图一致性；不计算伪造 Faro、Chamfer 真值误差或 F-score。静态场景和标定都可能影响残差，要同时展示 RGB/深度/轨迹证据。

## 8. 生成 DN 可读取的派生数据集

### 8.1 图像、深度和 mask

训练 RGB 使用已经固定的去畸变网格。历史 Nerfstudio FullImageDatamanager 的图像去畸变流程不会自动同步处理本项目 `sensor_depth`；因此必须离线把 RGB、深度和相机模型对齐，不能把原始 RGB+depth 交给 datamanager 后期待它一起修复。[S8][S9]

推荐派生深度为**米制 float32 `.npy`**，保存 D2C 后的 `Z_C`，后续 `depth_unit_scale_factor=1`。原始样例的 `.npz` 是回传容器，不能直接当作历史 `GDataset` 输入。无测量、65535、投影未覆盖或被派生距离 mask 排除的位置写 0；不通过毫米 uint16 二次量化来省这一步。

深度有效 mask 与 RGB mask 分开：缺深度的像素仍可能是有效 RGB 监督，不要把深度 valid mask 直接写到 `mask_path`。用于 RGB 去畸变边界或真实动态物体的图像 mask 需要独立定义。某幅图所有深度都无效时必须报告并按明确规则排除/处理，不能进入空集合 depth loss 的 mean。

历史深度损失主要以深度 `>0.1` 选择有效像素，没有本采集的合理距离上限；远景 mask 应在派生深度中落实，不能只在文档里声称裁切。保留算法损失实现，`num_downscales=0` 保持该 DN 配置默认值，不引入另一套分辨率调度。

### 8.2 相机、单位与初始化

输出 `images/`、`depth/`、`transforms.json`、`splits.json`、`source_frames.json` 和可选的训练帧初始化 PLY。文件使用相同补零编号，所有映射可追溯至原包消息。

`transforms.json` 写目标针孔图像的真实尺寸/内参、同一数据尺度、对应的 RGB 与深度路径和相机 c2w。OpenCV 光学轴 c2w 到 OpenGL/NeRF c2w 通过右乘 `diag(1,-1,-1,1)` 改变相机轴；全局原点/朝向变换必须一致作用于位姿和点云，不重复转换。

初始自有数据配置保持 `auto_scale_poses=False`、`scale_factor=1`、`orientation_method=none`、`center_method=none`，避免米制数据再次被隐藏归一化。若为了公开相对轨迹而归零，记录变换定义和相对参考帧，不公开可还原绝对地理位置的原点。

初始化 PLY 仅使用已固定训练帧的有效深度和相机位姿，记录采样/体素化规则、点数、颜色和范围；不使用测试帧或评估几何真值。若尚未生成有效 PLY，不伪填已完成初始化，也不借旧 VR Room 点云跳过该步骤。

后续 Omnidata 必须基于**最终去畸变 RGB**生成 `normals_from_pretrain/*.png`，文件顺序和名称逐帧对应。历史 parser 不使用任意 `normal_file_path` 来替代该目录机制。本轮只把这一依赖写入数据清单，不自动推理法线或开始训练。

数据阶段状态最多记为 `geometry_prepared_pending_review`；不要写 `training_ready=true`，因为法线、初始化和后续运行配置仍需按实际完成情况逐项记录。

## 9. 回传与本轮完成条件

公开几何报告建议放 `reports/capture-geometry/<run-id>/`，真实时间、原包 SHA 和输入 manifest 身份组成 run 身份，不覆盖原始审计。报告与状态入口在同一个提交发布。小型图文和 JSON 优先回传 GitHub，完整图像集、COLMAP database、稀疏/稠密点云留本项目并登记位置、大小与 SHA-256。

按照 `configs/capture_geometry_protocol.json` 至少包含：

- `overview.md` 与 `summary.json`：实际进度、推荐场景、能否进入训练准备、未解问题；
- `selection.json` 与 `split_manifest.json`：原包身份、精确窗口、选帧/关联、剔除规则及冻结的划分；
- `registration.json`：目标网格、单位、畸变、外参、像素约定、Z-buffer、无效/冲突/越界和样例诊断；
- `pose_quality.json` 与 `scale_alignment.json`：版本、固定相机参数、匹配策略、训练地图/留出定位口径、注册/连通性、尺度支持、相对轨迹及跨帧残差/失败对；
- `dataset_manifest.json`：真实本地/AutoDL 路径与传输校验、相机/图像/深度一致性、距离敏感性、缺少的 normals/PLY/后续设置；
- `artifact_manifest.json`：公开文件和本地大产物的身份/体积/哈希范围；清单本身不做递归哈希；
- 代表性 D2C 边缘图、相对轨迹图和多帧点云叠加预览；脚本、真实命令、最少依赖记录。

核验实际文件可读、映射一致、单位/位姿方向无误、轨迹无明显断裂、多个视角的几何对齐有支持即可；不要堆叠无关 smoke tests。达不到几何要求时明确 `partial` 或 `blocked`，保留失败证据和下一步，不以“COLMAP 退出码 0”替代场景几何通过。

最后返回本地/AutoDL 已完成阶段、实际帧数和划分、传输状态、D2C/轨迹/尺度结果、数据目录、报告分支/commit，以及是否满足进入下一阶段的条件。**当前计划终点是报告和准备好的派生数据，不自动训练。**

## 10. 依据与核查入口

- [A1] [原始审计 overview](../reports/capture-bag-audit/20261119T091841Z-3982cb26/overview.md)、[summary](../reports/capture-bag-audit/20261119T091841Z-3982cb26/summary.json)。
- [A2] [calibration](../reports/capture-bag-audit/20261119T091841Z-3982cb26/calibration.json)、[depth_quality](../reports/capture-bag-audit/20261119T091841Z-3982cb26/depth_quality.json)、[frame_graph](../reports/capture-bag-audit/20261119T091841Z-3982cb26/frame_graph.json)。
- [A3] [samples_metadata](../reports/capture-bag-audit/20261119T091841Z-3982cb26/samples_metadata.json)、[sync](../reports/capture-bag-audit/20261119T091841Z-3982cb26/sync.json)。
- [S1] OrbbecSDK 固定提交：[RosbagWriter.cpp](https://github.com/orbbec/OrbbecSDK_v2/blob/f5d59930ccb9a189f200e58fba4e2ae42664792d/src/media/ros/RosbagWriter.cpp)、[RosbagReader.cpp](https://github.com/orbbec/OrbbecSDK_v2/blob/f5d59930ccb9a189f200e58fba4e2ae42664792d/src/media/ros/RosbagReader.cpp)；定义录制字段和外参方向。
- [S2] [ObTypes.h](https://github.com/orbbec/OrbbecSDK_v2/blob/f5d59930ccb9a189f200e58fba4e2ae42664792d/include/libobsensor/h/ObTypes.h)、[Frame.h](https://github.com/orbbec/OrbbecSDK_v2/blob/f5d59930ccb9a189f200e58fba4e2ae42664792d/include/libobsensor/h/Frame.h)、[PlaybackDevice.cpp](https://github.com/orbbec/OrbbecSDK_v2/blob/f5d59930ccb9a189f200e58fba4e2ae42664792d/src/media/playback/PlaybackDevice.cpp)；畸变结构、毫米尺度和播放时恢复。
- [S3] [CoordinateUtil.cpp](https://github.com/orbbec/OrbbecSDK_v2/blob/f5d59930ccb9a189f200e58fba4e2ae42664792d/src/shared/utils/CoordinateUtil.cpp)；3D 坐标与 Z 定义，注意本文件中投影 helper 的模型分支。
- [S4] [Align.cpp](https://github.com/orbbec/OrbbecSDK_v2/blob/f5d59930ccb9a189f200e58fba4e2ae42664792d/src/filter/publicfilters/Align.cpp)、[AlignImpl.cpp](https://github.com/orbbec/OrbbecSDK_v2/blob/f5d59930ccb9a189f200e58fba4e2ae42664792d/src/filter/publicfilters/AlignImpl.cpp)；target distortion 默认值及实际 K6 分支。
- [S5] COLMAP 官方 [相机模型](https://colmap.github.io/cameras.html)、[常见问题](https://colmap.github.io/faq.html)、[命令行](https://colmap.github.io/cli.html)；现场版本可能不同，执行参数以已安装版本帮助为准。
- [S6] COLMAP 官方 [输出格式](https://colmap.github.io/format.html)；四元数、world-to-camera、相机中心和点轨迹。
- [S7] Open3D 官方 [RGB-D odometry](https://www.open3d.org/docs/release/tutorial/pipelines/rgbd_odometry.html)、[Multiway registration](https://www.open3d.org/docs/release/tutorial/pipelines/multiway_registration.html)；本任务只作为明确失败后的备选。
- [S8] 固定 DN 历史实现：[normal_nerfstudio.py](https://github.com/maturk/dn-splatter/blob/811f5cdf342d92d12a9d9a4be185d374c430c184/dn_splatter/data/normal_nerfstudio.py)、[dn_dataset.py](https://github.com/maturk/dn-splatter/blob/811f5cdf342d92d12a9d9a4be185d374c430c184/dn_splatter/data/dn_dataset.py)、[dn_model.py](https://github.com/maturk/dn-splatter/blob/811f5cdf342d92d12a9d9a4be185d374c430c184/dn_splatter/dn_model.py)、[camera_utils.py](https://github.com/maturk/dn-splatter/blob/811f5cdf342d92d12a9d9a4be185d374c430c184/dn_splatter/utils/camera_utils.py)。详细接口见 [自有数据接入说明](06_custom_data.md)。
- [S9] Nerfstudio 官方 [数据约定](https://docs.nerf.studio/quickstart/data_conventions.html)、固定 v1.1.3 [full_images_datamanager.py](https://github.com/nerfstudio-project/nerfstudio/blob/v1.1.3/nerfstudio/data/datamanagers/full_images_datamanager.py)、[cameras.py](https://github.com/nerfstudio-project/nerfstudio/blob/v1.1.3/nerfstudio/cameras/cameras.py)；其中 `depth_image` 与本插件的 `sensor_depth` 不能混为同一处理路径。
