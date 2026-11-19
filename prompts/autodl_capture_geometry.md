# AutoDL Codex：恢复 Gemini 335 首段位姿并准备 DN 输入

请实际执行数据准备。仓库：
https://github.com/kaiwen123-yang/DN-Splatter.git

已有 VR Room 30k/评估/Poisson 环境和结果保留。本轮输入来自本地 H:\capture.bag\capture.bag 的指定首段紧凑数据包；服务器不能直接访问该Windows路径。

获取最新仓库，先读 AGENTS.md、reports/capture-bag-review/2026-11-19/overview.md、docs/08_capture_geometry_plan.md、configs/capture_geometry_protocol.json，以及最新 reports/capture-geometry/ 报告和传输清单。

## 输入与环境

核对本次实际收到的包、包哈希、source hash、帧清单、标定、时间关联和固定划分，使用独立scene/run目录。如果数据包尚未到达，只读确认现有工具、准备执行脚本和恢复命令，并报告具体缺少的包；不能自动替用VR Room或其他项目数据。

允许使用AutoDL实例全部已分配CPU和整张GPU，实际budget按quota/affinity确认。复用已验证的DN环境与权重，新增COLMAP/pycolmap等放独立预处理环境；不修改驱动、系统CUDA、base或其他项目，不停止现有进程。GitHub优先在本项目子shell使用AutoDL官方学术加速。

检查现有COLMAP、Python binding、Open3D版本与各命令--help，记录实际版本再生成命令。GPU SIFT不支持Blackwell时可使用CPU，不能为此拆改已经工作的DN CUDA环境。不为占满GPU而启动无关实验。

## 几何处理主线

1. 按docs/08离线生成去畸变针孔RGB和同网格D2C深度，保存两套网格的真实K：1280×720供RGB SfM，640×360供RGB-D几何检查。若已有派生文件，先核对来源与参数再复用。depth uint16原值乘0.001得到米，经R,t变换后存Z_C，输出float32 .npy；原始0、65535、未覆盖与明确拒绝的测量在派生图中为0。不要用resize或跨边界插值填深度。
2. 从SDK/OpenCV整数像素中心K转COLMAP/Nerfstudio约定时cx/cy各加0.5，且只加一次。Brown系数顺序正确转换，去畸变后模型使用PINHOLE并固定已知共享内参。
3. 保留已固定的train/val/test清单。主方案只用训练图建立COLMAP稀疏地图和BA；随后将val/test的RGB定位到固定地图，不用它们的深度参与尺度拟合或初始化，不通过定位阶段改动训练地图。若现场工具只能完成全图SfM，将其单独标为all_rgb_sfm_diagnostic并披露，不能把它冒充主方案。
4. 先做SIFT、顺序匹配和增量mapper；首段仅数百图时可依据连通性增加长程候选或该小集合的exhaustive匹配。无需COLMAP dense stereo或其Poisson。保存输入/注册/未注册图、连通组件、track长度、重投影误差及真实相机轨迹。
5. 用训练帧中静态、非遮挡边缘、可信距离范围的registered sensor Z，与SfM稀疏点在该帧相机中的Z对应，稳健拟合一个正的全局尺度。按帧/点平衡权重，分别检查早中晚和距离分段。全部相机平移及世界点统一缩放；不能逐帧自由scale，不能把sensor depth当欧氏距离。
6. 用恢复位姿在5–10个分布于片段的视角做遮挡感知重投影，检查桌面、箱边、椅腿的多层表面、漂移、镜像和尺度。保留帧深度可用于独立于尺度拟合的诊断；这是传感器一致性检查，不是几何真值准确率。
7. 实测SfM失配时，先对具体缺口加密帧和补必要长程匹配；需要更多原帧时通过同一已授权连接请求本地索引提取，或输出明确帧清单。仍不能形成可靠模型时才启用Open3D离线RGB-D odometry加pose graph/回环备选。不得把失败odometry边填成identity强约束。无需本轮同时搭建完整ROS SLAM/VIO体系。

## 导出与本轮完成条件

只有位姿、尺度和目标网格验证成立时才输出实际prepared dataset。文件名统一补零；COLMAP w2c先求逆得到OpenCV c2w，再右乘diag(1,-1,-1,1)得到OpenGL c2w。共同世界归零变换同步作用于相机和点云。

导出images、depth、transforms.json、完整train/val/test列表及真实训练帧初始化PLY。使用米制.npy时depth_unit_scale_factor=1；保留auto_scale_poses=False、scale_factor=1、orientation_method=none、center_method=none、downscale_factor=1、num_downscales=0。不依赖datamanager自动去畸变sensor_depth，也不把depth valid mask误用作全RGB mask_path。

本轮可以准备上述入口，但还没有Omnidata法线时必须标为geometry_prepared_pending_normals_and_training，不能称完整训练输入已验证。法线以后从最终选定网格RGB生成normals_from_pretrain/*.png，历史parser不读取逐帧normal_file_path。

报告放 reports/capture-geometry/<run-id>/，记录local extraction、transfer、registration、SfM、metric scale、geometry validation、dataset export的实际状态。提供逐帧/逐阶段数量、失败原因、轨迹与融合点云视图、尺度残差、相机/深度/划分身份、完整大文件清单及恢复命令。没有GT不生成Faro/F-score。

只推小报告、代码、配置和代表样例；大数据留本项目独立目录并登记hash。latest.json与报告同提交发布，不force push、不覆盖原始审计或VR Room结果。使用项目持久会话，不给完整几何任务设置人为完成deadline；遇到真实阻塞保存可继续状态。

本轮到几何报告和实际可准备的数据集交付为止，不自动执行法线推理、DN训练或Poisson。返回报告commit、实际数据目录和下一步是否具备开始自有场景30k的条件。
