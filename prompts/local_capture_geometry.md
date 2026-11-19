# 本地 Codex：提取 Gemini 335 首段并准备几何验证

请实际执行，不要只返回计划。仓库：
https://github.com/kaiwen123-yang/DN-Splatter.git

本地只读审计已经完成，结果提交 a13bb40445c1c049c2931094e533a5ed43472297。实际原包为 H:\capture.bag\capture.bag；H:\capture.bag 是目录。不要重新猜相机型号、单位或文件结构，也不要重复做整轮审计。

## 先读并接续

获取仓库最新内容，保留现有修改。阅读 AGENTS.md、reports/capture-bag-review/2026-11-19/overview.md、docs/08_capture_geometry_plan.md、configs/capture_geometry_protocol.json。

复用本机项目 .local/venv/bag-audit、四个 capture_bag_*.py 的真实厂商解析，以及 .local/bag-audit/20261119T091841Z-3982cb26/times.sqlite。先核对源文件身份、索引所属 SHA256、大小和 mtime；索引缺失或身份不匹配时才有依据地重建。原包只读，不复制或改写，不向 Git/LFS 上传原包。

## 本轮执行

1. 使用 metadata 中样例 #01 与 #04 的精确设备时间确定首段，分别为 2427228619000 和 2475353026000 ns。相对首张深度为 0.465019–48.589426s。不能用四舍五入后的 0.47 当精确下界，也不能把厂商帧号直接当数组下标。
2. 从起点每 200000000ns 取一个目标时刻，目标时刻不超过终点，约241个目标。选最近且不重复的真实 RGB，关联既有10ms容差内的一对一深度，记录真实时差。先提取这些选定原JPEG、原uint16深度NPZ及完整帧映射、真实标定和哈希，不预先把整段或整包全部解压。
3. 创建覆盖所选片段的连续缩略预览，检查四张旧均匀预览之间是否存在严重模糊、人员遮挡、时间缺段或视角突变。实际剔除必须登记理由，不能为了后续指标好看删帧。
4. 对分布在首段的5–10组数据做同帧D2C，复用并适配 scripts/capture_registration_review.py 的数学逻辑。该脚本已经对4组公开样例执行，但不是完整导出器；补充必要实现并保存到scripts，不能只留.local。
5. 使用真实 depth→color 变换、毫米到米、完整Brown畸变。统一去畸变RGB目标网格，输出彩色光轴 Z_C，最近表面z-buffer，空洞保留0。优先保留1280RGB供SfM，同时准备640×360同网格RGB-D供几何核验。内参和像素中心按docs/08显式记录，不能resize深度替代配准。
6. 报告0.3–3m、0.3–6m、0.3–10m的覆盖与边缘差异，保留远景原数值；这些是诊断范围，不是已经验证的传感器可靠范围。无独立参考时不输出“配准精度达标”或“94%深度准确”的结论。
7. 质量筛选完成后固定顺序与train/val/test清单，按协议的确定性规则划分。不要在得到SfM或训练结果后移动帧。补帧用于恢复几何时登记辅助建图身份，保留原评价清单。

## 紧凑数据包与 AutoDL

把所选原JPEG、原值深度、标定、时间映射、划分、manifest和版本信息整理为本项目 .local/capture-geometry/<run-id>/transfer/ 下的紧凑输入包；run-id包含source SHA前缀并记录实际manifest身份。不要重复打包大中间数组，先计算实际体积和SHA256。GitHub只推小报告、代码与必要样例。

如果本机已有本项目明确可用、已授权的 AutoDL 连接，则复用它传输紧凑包，在服务器独立目录核对包哈希和内容清单；不终止现有任务，不写全局代理、不改DN训练环境。遵循 prompts/autodl_capture_geometry.md 继续位姿恢复和准备数据；不要重复开启相同运行。

如果没有可用服务器连接，仍完成提取、D2C核验、包与报告，把完整本地待传路径、字节数、SHA256、所需目标目录与准确阻塞返回。不要猜SSH地址、密码或端口，也不要退回要求上传6GB原包。

本机资源按实际CPU/RAM和可用内存选择并行；25CPU/96GB是AutoDL的配置，不是本机的默认资源。GitHub学术加速只在AutoDL项目子shell应用，本地使用现有GitHub连接。

## 回传与结束

公开回传统一放 reports/capture-geometry/<run-id>/，更新同目录层级的latest.json，至少提供本地阶段的frame_selection、registration_checks、transfer_manifest摘要、代表图与阶段状态，缺失服务器步骤明确未执行。已有完整过程表保留.local。

与报告同一次提交推送最新指针；只stage本任务文件，保留其他工作，不force push。最后返回报告分支/commit、输入包的真实位置/哈希、选帧实际数量、配准检查结论、服务器步骤完成或待传状态。

本轮目标是首段正确输入、相机位姿与米制几何验证，以及条件满足时的prepared dataset。到几何报告回传为止，不自动生成Omnidata法线，不开启DN训练或Poisson；后续根据实际报告决定首次自有场景30k。
