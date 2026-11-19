# capture-s01 完整结果复核与全程测试决定

复核日期：2026-11-19。证据固定在提交 [`ee3e87bf8a63486ed96df1c0d37b6a72d30fca88`](https://github.com/kaiwen123-yang/DN-Splatter/commit/ee3e87bf8a63486ed96df1c0d37b6a72d30fca88)，所有衍生统计见 [findings.json](findings.json)。本报告只读取已交付结果；没有启动新的训练、评估或服务器任务。

**结论：capture-s01 已完成自有 RGB-D 数据的完整接入基线，无需为进入下一阶段重跑 s01。下一步直接进行约 369 秒全程、固定 5 Hz 取帧、累计 30,000 次更新、完整留出评估和原始 Poisson 的第一次全程测试。** 保留全部正式指标和困难帧。后续处理依据这次全程结果决定，不再增加 s01 小验证或调参轮次。

## 1. 已完成的范围和结论边界

本轮为约 48 秒、241 对 RGB-D，其中 193 训练 / 24 验证 / 24 测试。正式输入使用 1280×720 去畸变 RGB、0.3–6 m 米制光轴 Z、冻结位姿和 436,361 个训练初始化点。单段训练实际完成 30,000 次更新，最终零基 step 29,999，得到 440,882 个高斯。Val、test 和原始 Poisson 均绑定最终 checkpoint；其 SHA256 为 `fb6636d164d1edd2ca7a0bdd797078baf5cfb06be440953b4d42b6da750b3ab9`。训练记录中 3,000 个 Train Loss 采样均有限，首个正式训练批次的检查未发现非有限梯度；48 张留出图有效参考像素内的无效深度预测和零累积像素均为 0。这些证据支持流程和数值执行已完成，不能代替几何精度认证。[训练记录][training]、[读取记录][reader]、[原始结果][original]

当前图像能辨认桌椅、箱柱等场景结构，但末段视角和细杆几何存在明显失败。对自己的数据，**“已能完成重建流程”已得到验证，“全采集覆盖和可用几何质量”仍待下一次全程结果回答**。同一轨迹的传感深度不是独立真值，本轮不能据此声称达到了论文原数据的精度。[原始结果][original]、[网格记录][mesh]

## 2. 正式指标保持不变

RGB 为逐帧等权；下表深度 MAE / RMSE 同样是逐帧等权均值。

| 划分 | 帧数 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | 深度 MAE ↓ | 深度 RMSE ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Val | 24 | 23.2433 dB | 0.77935 | 0.29775 | 0.13339 m | 0.27624 m |
| Test | 24 | 23.3502 dB | 0.77934 | 0.30406 | 0.19838 m | 0.32643 m |

全参考像素合并时，Val MAE / RMSE 为 0.12215 / 0.35837 m，Test 为 0.16460 / 0.50281 m。两种聚合的权重不同，不能互换。每帧 MAE 的中位数为 Val 0.09317 m、Test 0.09468 m，说明均值被少数重尾视角显著拉高。[Val 原始 JSON][val]、[Test 原始 JSON][test]

沿用的历史 LPIPS 为 AlexNet、`normalize=False`、RGB 浮点输入 `[0,1]`；报告已记录这一约定。与采用其他归一化方式的外部 LPIPS 数值比较时需要统一口径。本次复核不改变既有指标实现，也不据此重跑 s01。[评估实现][eval]

## 3. 末段 238 / 239 是明显局部失败，不能只看平均数

| 帧 | 划分 | PSNR | 深度 MAE | 深度 RMSE | 深度绝对误差 P95 |
|---|---|---:|---:|---:|---:|
| 238 | Val | 17.21 dB | 0.83238 m | 1.55406 m | 3.90334 m |
| 239 | Test | 13.57 dB | 2.34382 m | 2.61522 m | 4.78319 m |

第 239 帧只有 **99,370 个参考像素，占 Test 的 2.786%**，却贡献 **39.68% 的绝对误差总和和 75.38% 的平方误差总和**。Val 第 238 帧也明显失效，贡献该组 54.93% 的平方误差。其次较差的视角在 Val / Test 中也成对出现：128 / 129 的 MAE 为 0.23792 / 0.25091 m，228 / 229 为 0.18313 / 0.21020 m。因此证据指向特定视角区域的失败，不能把 239 当作唯一偶发异常，更不能直接认定测试划分读取出错。[Val][val]、[Test][test]

下面只做敏感性分解，展示末段帧对均值的影响。**这不是删除困难样本后的正式成绩，也不是下一轮排除这些区域的依据。**

| 统计口径 | Val 等权帧 MAE | Test 等权帧 MAE | Val 像素合并 MAE | Test 像素合并 MAE |
|---|---:|---:|---:|---:|
| 正式全 24 帧 | 0.13339 m | 0.19838 m | 0.12215 m | 0.16460 m |
| 仅诊断：分别暂不计 238 / 239，其余 23 帧 | 0.10300 m | 0.10510 m | 0.10078 m | 0.10213 m |

预览中，238 / 239 的远墙和顶棚都明显模糊，239 的近处有效传感参考区域同样有大面积误差，故“远处超过 6 m”不足以解释全部失败。[238 预览][preview238]、[239 预览][preview239]

已有定位记录显示，238 有 854 个 PnP 内点、内点重投影中位误差 0.819 px；239 有 762 个内点、中位误差 1.006 px。没有据此判为定位彻底失败的证据；反过来，较小的二维残差也不能保证整条米制轨迹没有几何误差。可能相关的因素包括视角覆盖、远景监督不足、同采集深度与位姿的跨视角不一致以及模型表示/优化限制，目前不能唯一归因。[位姿记录][poses]

正式 1280×720、0.3–6 m 训练深度的正值覆盖中位数只有 **16.28%**，范围 9.06%–22.69%。这与此前 640 网格约 95% 的 D2C 观测率不是同一分母、同一深度范围；不能把后者解释为全图都有正式有效深度监督。已有宽基线残差也带方向性长尾，这些都是解释下一次全程结果时需要保留的上下文，而不是新增一轮局部修复的理由。[输入覆盖][inputs]、[跨视角记录][cross]

## 4. 网格有占主导的主体，细节质量仍有限

原始 Poisson 有 457,112 个顶点、911,783 个面和 975 个连通分量。其中最大分量占 **873,616 / 911,783 = 95.81% 的面**，其余所有分量合计 38,167 个面。不能仅凭 975 个分量就说整个场景都碎裂，也不能凭主体连续就认定网格正确。[网格记录][mesh]

历史导出器从 193 个训练视角采样 5,000,051 点，其中 **570,536 点，即 11.41%**，对应模型深度超出 0.3–6 m 传感监督范围；零累积不透明度为 43 点，占 0.000860%。采样法线无非有限值、无零向量，几何数组有限。范围外采样比例提示对远景和补面要谨慎解读，不等于这些点已被逐点证明错误。[网格记录][mesh]

原报告局部图中桌沿增厚和起伏、柱顶变圆、椅腿与地面或邻近表面粘连，说明当前模型更适合检查场景大结构，尚不支持细杆间隙和精确厚度的可靠测量。本次保留原始 Poisson 作为后续对照，不提前增加导出器改造或新网格实验。[原始局部图][original]

## 5. 下一次直接执行全程 5 Hz、30k 基线

“全程”指覆盖原包约 369 秒的全部时间和经过的区域；固定 5 Hz 取帧是本轮已接受的时间采样规则，并不要求把约 30 Hz 的 11,041 张 RGB 全部塞进训练缓存。新场景应有独立输入清单、冻结划分、数据和运行身份，不复用 s01 的完成标志或 checkpoint。已有 s01 配置里的固定 193 / 24 / 24、436,361 点和对应场景范围不能直接当成全程数据的实际值；全程执行器应根据自己的冻结清单一次性处理，并交付实际注册数及缺失覆盖。这个工作属于全程任务本身，不另设一个小规模批准关卡。

训练预算仍为累计 30,000 次更新。按计划训练图数由 193 增至 1,478，在每步使用一个视角的前提下，平均图像被选中次数由 **30,000 / 193 = 155.44 次**降为 **30,000 / 1,478 = 20.30 次**，约为 s01 的 **13.06%**。这是预算密度的算术事实，实际次数随最终帧数和遍历顺序变化；它不证明全程 30k 必然不收敛，也不构成自动延长训练的授权。首先完成既定 30k、完整评估和 Poisson，再根据结果决定后续是否调整。[历史逐步取帧实现][datamanager]

本复核不建议重新跑 s01，不新增超参数搜索、消融或循环小验证，也不提前承诺全场景可由单一一致地图覆盖。全程结果应保留困难末段、动态遮挡和几何失败的实际记录；真正不能注册的帧要报告计数和时间范围，不能无声消失后仍称全量成功。

## 附节 A：历史增密调度的真实规模约束

冻结历史源码的 `refinement_after` 使用以下必要条件：[固定源码][model]

```python
reset_interval = self.config.reset_alpha_every * self.config.refine_every
do_densification = (
    self.step < self.config.stop_split_at
    and self.step % reset_interval
    > self.num_train_data + self.config.refine_every
)
```

本次正式配置为 `reset_alpha_every=30`、`refine_every=100`、`stop_split_at=15000`，因此周期为 3,000 步。s01 的 193 张训练图允许周期余数 300、400、…、2,900 的回调满足这条条件；计划 1,478 张训练图时，满足条件的余数变为 1,600、1,700、…、2,900，仍有增密窗口。这里只计算必要条件，实际增密还取决于可见性、梯度阈值和其他条件，不能从窗口数直接推算新增高斯数。[正式配置][config]、[固定源码][model]

若把所有 11,041 张 RGB 按约 80% 训练，训练图数约 8,833，`step % 3000 > 8833 + 100` 永远为假。原调度下增密将没有窗口；单纯把总训练步数增加也不能改变这个条件。全程 5 Hz 方案避开了这个直接规模问题。报告这一源码约束不等于在本轮擅自改变损失、增密设置或 30k 预算。

## 附节 B：Nerfstudio 1.1.3 缓存行为和内存量级

固定 v1.1.3 对应提交 `9b3cbc79bf239eb3c69e7c288632aab02c4f0bb1`。`FullImageDatamanager` 在训练集超过 500 张且配置为 GPU 图像缓存时，自动将其改为 CPU 缓存；该版本图像缓存线程池实际写为 `ThreadPoolExecutor(max_workers=2)`。因此配置文件中写着 `cache_images: gpu` 或机器有 96 GB 显存，不代表全程运行一定把所有缓存放到显卡；这不是训练失败。[固定缓存源码][nscache]

按 1280×720、RGB 3 通道 + Z 1 通道 + 法线 3 通道、全部 float32 估算，基础张量为 `1280 × 720 × 7 × 4 = 25,804,800` 字节/帧。1,478 张训练图约 38.14 GB；若再计 184 张验证图，共 1,662 张约 **42.89 GB（39.94 GiB）**。这只是张量量级估算，未计初始化、模型、Python 对象、锁页和复制瞬时开销、SfM 及其他活动，不能当作实际峰值保证。11,041 张同规格全缓存仅这些基础数组就约 284.91 GB，超过现有 120 GB 内存。最终内存以全程实际帧数和实现记录为准。[固定缓存源码][nscache]

上述两项约束说明应采用一次性明确的全程采样协议和独立执行器；它们不要求重新进行环境安装、Blackwell 适配或 s01 试训。

## 统计口径与复算

设第 i 帧有效参考像素数为 nᵢ、帧 MAE 为 mᵢ、帧 RMSE 为 rᵢ，则像素合并 MAE 为 `Σ(nᵢ mᵢ) / Σnᵢ`，像素合并 RMSE 为 `sqrt(Σ(nᵢ rᵢ²) / Σnᵢ)`。第 239 帧绝对误差贡献为 `n₂₃₉ m₂₃₉ / Σ(nᵢ mᵢ)`，平方误差贡献为 `n₂₃₉ r₂₃₉² / Σ(nᵢ rᵢ²)`。所有本报告衍生值从冻结逐帧 JSON 计算；全样本复算与原始聚合值在浮点精度内一致。删帧敏感性只重算可由逐帧统计恢复的量，没有伪造无法恢复的合并像素 P50 / P95。输入文件 SHA256、原始及诊断统计、公式和来源均保存在 [findings.json](findings.json)。

[original]: https://github.com/kaiwen123-yang/DN-Splatter/blob/ee3e87bf8a63486ed96df1c0d37b6a72d30fca88/reports/capture-s01-dn-30k/20261119-v1/overview.md
[training]: https://github.com/kaiwen123-yang/DN-Splatter/blob/ee3e87bf8a63486ed96df1c0d37b6a72d30fca88/reports/capture-s01-dn-30k/20261119-v1/training.json
[reader]: https://github.com/kaiwen123-yang/DN-Splatter/blob/ee3e87bf8a63486ed96df1c0d37b6a72d30fca88/reports/capture-s01-dn-30k/20261119-v1/reader_check.json
[val]: https://github.com/kaiwen123-yang/DN-Splatter/blob/ee3e87bf8a63486ed96df1c0d37b6a72d30fca88/reports/capture-s01-dn-30k/20261119-v1/metrics_val.json
[test]: https://github.com/kaiwen123-yang/DN-Splatter/blob/ee3e87bf8a63486ed96df1c0d37b6a72d30fca88/reports/capture-s01-dn-30k/20261119-v1/metrics_test.json
[mesh]: https://github.com/kaiwen123-yang/DN-Splatter/blob/ee3e87bf8a63486ed96df1c0d37b6a72d30fca88/reports/capture-s01-dn-30k/20261119-v1/mesh_validation.json
[inputs]: https://github.com/kaiwen123-yang/DN-Splatter/blob/ee3e87bf8a63486ed96df1c0d37b6a72d30fca88/reports/capture-s01-dn-30k/20261119-v1/inputs.json
[cross]: https://github.com/kaiwen123-yang/DN-Splatter/blob/ee3e87bf8a63486ed96df1c0d37b6a72d30fca88/reports/capture-s01-dn-30k/20261119-v1/cross_view_diagnostic.json
[config]: https://github.com/kaiwen123-yang/DN-Splatter/blob/ee3e87bf8a63486ed96df1c0d37b6a72d30fca88/reports/capture-s01-dn-30k/20261119-v1/config.yml
[eval]: https://github.com/kaiwen123-yang/DN-Splatter/blob/ee3e87bf8a63486ed96df1c0d37b6a72d30fca88/scripts/capture_s01_eval.py
[poses]: https://github.com/kaiwen123-yang/DN-Splatter/blob/d72f56836028d7a7885f21f02522e4f7cc310429/reports/capture-geometry/20261119-s01-5hz-3982cb26/pose_quality.json
[preview238]: https://github.com/kaiwen123-yang/DN-Splatter/blob/ee3e87bf8a63486ed96df1c0d37b6a72d30fca88/reports/capture-s01-dn-30k/20261119-v1/previews/val_frame_00238.jpg
[preview239]: https://github.com/kaiwen123-yang/DN-Splatter/blob/ee3e87bf8a63486ed96df1c0d37b6a72d30fca88/reports/capture-s01-dn-30k/20261119-v1/previews/test_frame_00239.jpg
[model]: https://github.com/maturk/dn-splatter/blob/811f5cdf342d92d12a9d9a4be185d374c430c184/dn_splatter/dn_model.py#L269-L278
[datamanager]: https://github.com/maturk/dn-splatter/blob/811f5cdf342d92d12a9d9a4be185d374c430c184/dn_splatter/dn_datamanager.py#L89-L96
[nscache]: https://github.com/nerfstudio-project/nerfstudio/blob/9b3cbc79bf239eb3c69e7c288632aab02c4f0bb1/nerfstudio/data/datamanagers/full_images_datamanager.py
