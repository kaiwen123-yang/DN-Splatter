# 2026-11-23：B2 训练及 B3 实现复核

复核冻结仓库：`d6405240df30c49e71365f87fdce4ec7ce7c7462`。本报告只读取代码及已发布结果，没有访问 AutoDL 进程或启动训练。所有“已实现”均指此提交中的代码；“已运行”只在有结果收据时成立。

## 1. 结论

**上一轮针对训练的主要修改已经取得实质收益。现在继续照着旧诊断调低删点阈值、重新跑同一 B2 或直接延长训练，并不是优先路线。** 应先交付正在开发的 B3 冻结模型成面比较，再由实际位姿与深度可靠性诊断决定一次有明确改变的完整训练。B3 的纹理分裂目前没有实现生效，主脚本也只串到了 A4，不能把计划中的各项写成已完成。

## 2. B2 已经从 30k 训练到 240k，210k 是按全量验证选出的主模型

B2 的八个完整验证节点如下，每个184帧。RGB 按帧等权，深度为同采集0.3–6m有效传感器 Z 的全部参考像素合并；这不是独立真值几何误差。[验证选点](https://github.com/kaiwen123-yang/DN-Splatter/blob/d6405240df30c49e71365f87fdce4ec7ce7c7462/reports/capture-full-quality-v2/20261121-v1/selection.json)

| 完成更新 | Val PSNR dB | SSIM | LPIPS | 深度 MAE cm |
|---|---:|---:|---:|---:|
| 30k | 23.2027 | 0.78777 | 0.22508 | 12.573 |
| 60k | 23.5606 | 0.79321 | 0.21695 | 12.812 |
| 90k | 23.7402 | 0.79548 | 0.21118 | 12.615 |
| 120k | 23.7280 | 0.79484 | 0.21337 | 13.177 |
| 150k | 24.3581 | 0.80210 | 0.20413 | 12.390 |
| 180k | 24.4563 | 0.80342 | 0.20063 | 12.537 |
| 210k | 24.6442 | 0.80451 | 0.19902 | 12.039 |
| 240k | 24.6381 | 0.80473 | 0.19782 | 12.245 |

八个节点均通过与 B1 的 RGB 非劣门槛。210k 的验证 MAE 最小，240k 未继续改善深度，选210k符合原协议。不能为提高事后测试 PSNR 改选240k。

| 固定184张开发测试图 | PSNR dB | SSIM | LPIPS | 深度 MAE cm | RMSE cm | 深度绝对误差 P95 cm |
|---|---:|---:|---:|---:|---:|---:|
| B1 30k | 16.8851 | 0.66411 | 0.41303 | 33.645 | 78.518 | — |
| B2 30k 留存节点 | 23.2654 | 0.78838 | 0.22650 | 12.964 | 46.968 | 42.700 |
| B2 210k 主模型 | 24.6807 | 0.80445 | 0.20104 | 12.358 | 38.930 | 44.130 |
| B2 240k 最终模型 | 24.7033 | 0.80453 | 0.19986 | 12.724 | 39.252 | 47.495 |

来源：[B1](https://github.com/kaiwen123-yang/DN-Splatter/blob/d6405240df30c49e71365f87fdce4ec7ce7c7462/reports/capture-full-metric3d-30k/20261120-v1/metrics_test.json)、[B2 30k](https://github.com/kaiwen123-yang/DN-Splatter/blob/d6405240df30c49e71365f87fdce4ec7ce7c7462/reports/capture-full-quality-v2/20261121-v1/posthoc_30k/test_step-000029999.json.gz)、[B2 210k](https://github.com/kaiwen123-yang/DN-Splatter/blob/d6405240df30c49e71365f87fdce4ec7ce7c7462/reports/capture-full-quality-v2/20261121-v1/test_step-000209999.json.gz)、[B2 240k](https://github.com/kaiwen123-yang/DN-Splatter/blob/d6405240df30c49e71365f87fdce4ec7ce7c7462/reports/capture-full-quality-v2/20261121-v1/test_step-000239999.json.gz)。B2三个节点共用29,100,252个参考像素，均无无效预测。

B2前30k已取得B1→B2最终选点全部PSNR dB增益的 **81.8%**、MAE降幅的 **97.2%**。随后180k更新再提升1.4153dB，MAE仅降低0.606cm（4.67%）。184帧中170帧PSNR进一步提高、141帧MAE进一步降低，但深度P95和前方冲突比例反而略升，不能说所有几何尾部都变好。这支持降低“继续加步数”的优先级，不说明长训练无用。

B2 30k是240k总调度中的留存节点；同更新数对照仍包含位姿、初始化、采样和增密策略、学习率时程等改变。它不是纯删点消融、也不是与论文完全相同的30k配置。没有B1同条件210k，不能将全部收益分解为某个单因素因果效果。30k没有同口径网格结果，不能推断其成面质量已等同210k。

## 3. 旧容量退化机制已缓解，不能继续按“最终17万高斯”分析现在的模型

独立复算300条真实细化记录：115次满足细化条件，每条均满足 `before + split_children + duplicates - cull_union = after`。初始化2,071,347个点；首次有效细化step5600后为1,211,003个，净减少41.535%。删除仍然明显，但不再是旧B1首轮约80%的规模。120k以后150次回调全部没有增删点或重置，最终/210k主模型均为 **761,513个高斯**，约为B1最终171,417个的4.44倍。[真实事件](https://github.com/kaiwen123-yang/DN-Splatter/blob/d6405240df30c49e71365f87fdce4ec7ce7c7462/reports/capture-full-quality-v2/20261121-v1/refinement.jsonl.gz)

代码确实实施了alpha阈值0.005、4000步warmup、800步细化、24000步重置周期、重置后3个完整训练epoch恢复、120k停止增密且之后关闭持续裁剪；不是只修改了计划表。[训练器](https://github.com/kaiwen123-yang/DN-Splatter/blob/d6405240df30c49e71365f87fdce4ec7ce7c7462/scripts/capture_quality_train.py) · [实际配置差分](https://github.com/kaiwen123-yang/DN-Splatter/blob/d6405240df30c49e71365f87fdce4ec7ce7c7462/reports/capture-full-quality-v2/20261121-v1/configuration.diff)

因此，低点数与后段持续删点已不再是当前最有力的解释。高斯总数也不能代表椅腿或纹理区域实际分配到多少有效容量；没有局部归因，不能据总数再要求无差别增密。

报告陷阱：本目录 `curve.json` 虽带B2 run_id，其内容是B1六个固定checkpoint的全验证曲线。B2应读 `selection.json` 与 `val_step-*.json.gz`。B1真正全验证 MAE 在15k为0.433m、20k为0.188m、末点为0.313m，不能再概括成“15k后单调变差”或“15k最佳”。这些名称问题不影响B2实际的八节点选点逻辑。

## 4. 真实论文/历史实现差别：有披露项，但不能当成已证实失败原因

从本地上游git对象读取固定 `811f5cdf342d92d12a9d9a4be185d374c430c184`，核对 `dn_model.py` SHA256为 `00781070b9f17b9739f18a32a324586ed7461409b26a69d2624a3f0083bebc60`，与B2 `loaded_sources.json`逐字一致；datamanager SHA也相符。因此可以确认实际训练沿用历史目标：

```text
L_rgb + 0.2 L_sensor_depth + 0.1 (L_normal_L1 + L_normal_TV + L_flatten)
```

历史代码将扁平化损失放入 `normal_lambda` 乘法内，论文式(12)的外部扁平化系数为1。内部损失定义和归一化也需一并考虑。此差别值得单独定义实验，不能静默将系数放大10倍后仍称同一历史复现，更不能预言它必然修好细杆或粘连。[已记录的源码审计](https://github.com/kaiwen123-yang/DN-Splatter/blob/d6405240df30c49e71365f87fdce4ec7ce7c7462/docs/03_source_audit.md) · [实际载入源码](https://github.com/kaiwen123-yang/DN-Splatter/blob/d6405240df30c49e71365f87fdce4ec7ce7c7462/reports/capture-full-quality-v2/20261121-v1/loaded_sources.json) · [历史上游代码](https://github.com/maturk/dn-splatter/blob/811f5cdf342d92d12a9d9a4be185d374c430c184/dn_splatter/dn_model.py)

`use_scale_regularization=false` 关闭的是Nerfstudio额外scale项，**不等于** `two_d_gaussians=true` 的DN扁平化项没启用。当前上游main重构中的深度权重/法线参数问题也不能套到这份历史源码。B2已使用Metric3D v2法线和1280×720全分辨率，不应再把“换Metric3D/开全分辨率”当作下一轮尚未尝试的建议。

## 5. B3到底实现到哪一步

| 项目 | 冻结提交中的实际行为 | 能否声称质量已提高 |
|---|---|---|
| 深度可靠性权重 | 原生深度样本到冻结RGB-Z映射核验；独立训练组投票；权重 `(1+S)/(1+S+C)`；另留一组诊断；训练损失逐方向带权并验证全1损失/梯度等价 | 有实现，无发布的启用收据/训练结果 |
| 条件训练 | 绑定B3独立输入清单、源数据哈希、完整恢复状态；复用B2的240k训练器；对不完整诊断拒绝启动 | 有入口，无发布的实际训练结果 |
| 纹理增强 | 每块 `eligible=False`，结果固定 `texture_split_enabled=False`、`status='incomplete'`；真实贡献Gaussian ID未验证 | 尚未实现启用；不能说已定向增密 |
| 纹理训练接入 | 若 `texture_split_enabled=True`，训练器直接raise并说明缺少已验证的贡献ID和可恢复原生分裂实现 | 未接入 |
| 持久主链 | `launch_capture_surface_quality.sh` 等A3结束，做model/sensor两支全184帧val，写 `frozen_pair_val.json` 后结束 | 当前尚未串B诊断、C训练与最终交付 |

来源：[权重](https://github.com/kaiwen123-yang/DN-Splatter/blob/d6405240df30c49e71365f87fdce4ec7ce7c7462/scripts/capture_surface_weights.py)、[训练](https://github.com/kaiwen123-yang/DN-Splatter/blob/d6405240df30c49e71365f87fdce4ec7ce7c7462/scripts/capture_surface_train.py)、[纹理](https://github.com/kaiwen123-yang/DN-Splatter/blob/d6405240df30c49e71365f87fdce4ec7ce7c7462/scripts/capture_surface_texture.py)、[主入口](https://github.com/kaiwen123-yang/DN-Splatter/blob/d6405240df30c49e71365f87fdce4ec7ce7c7462/scripts/launch_capture_surface_quality.sh)。

公开 `launch_verified.json` 是2026-11-22 05:58:46 UTC的观察：A1完成34/187组、272训练帧，训练开关及收益全为null。此后代码多次更新，但仓库没有新的最终B3结果报告。这份旧观察不能证明11月23日服务器进程仍在同一阶段，也不能证明B3训练已经跑过。[发布状态](https://github.com/kaiwen123-yang/DN-Splatter/blob/d6405240df30c49e71365f87fdce4ec7ce7c7462/reports/capture-full-quality-v3/20261122-v1/launch_verified.json)

这里的完成条件应明确，且不修改已冻结协议：**实现能力不足（implementation_unavailable）与实测诊断证明本轮不应启用是两种不同结论。** 若固定协议允许在实际检查贡献ID能力后关闭可选项，应保留所检查接口与训练块证据并明确能力限制；这不等于已经证明纹理无需增强。若协议要求的诊断仍缺失，就继续如实标记未完成，交付A链已有结果并说明具体未完成项，不能绕过现有门。尤其不能把 `incomplete` 改成某个“完成”字符串就当完成，也不能仅因尚未实现而将整轮标成 `skipped_equivalent`。当前两视角比较使用两者均值作为清晰度中位阈值，两个非相等数不可能都超过该均值；源码已诚实地标记不完整，需要多视角清晰度依据，或有事实依据地关闭这项。

## 6. 下一步的优先级

1. **先完成既有B3冻结210k成面比较。** 这利用已训练好的模型，直接检验当前最明显的多视角表面损失。保留全184帧的命中率、10cm全参考正确覆盖、前方冲突、共同命中MAE和固定物体原尺寸图；不能只看新增面数量或命中率。RGB冻结不变，不把导出收益称成训练收益。
2. **将B诊断做成有限、实际、可执行的决定。** 重点是1304/1305等坏位姿与跨视角可靠深度。可靠性权重已有训练接入，实测验证支持才启用；不要因少量远距像素分歧直接推断整个相机标定错误。B2的4–6m层MAE28.848cm、10cm正确覆盖33.979%，这一层23.36%的像素贡献54.54%的绝对误差，应单列验证。
3. **只在改变了有效输入/损失证据后执行一次新的完整训练。** 接受的位姿修改、有效的非恒定深度权重，或真正实现并验证的局部分裂，至少有一项；否则诚实交付冻结模型导出改善。保持1846位置和原拆分，不删除难帧提高平均值。不另开短训练、扫描多个参数。
4. **在这一完整结果后再决定损失公式或采集改造。** 扁平化系数、法线置信度、动态目标、独立新采集的价值都可明确讨论，但不把尚未证实的候选原因全部塞进本轮。已有test反复用于开发，后续对外泛化结论需要独立轨迹/采集；同次传感器深度和论文独立表面F-score不可直接相减。

机器可读原始数值、来源哈希和结论分类见 [training_findings.json](training_findings.json)。
