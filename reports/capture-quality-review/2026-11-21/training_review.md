# 全程质量复核：训练过程与下一次完整训练

日期：2026-11-21，北京时间。固定证据提交 `3728dae350a240d317702d57e560554d529679c6`。本次读取 B0/B1 训练曲线、B1 全部 30,000 条采样记录、300 条细化事件、配置与记录代码，并核对 DN `811f5c` / Nerfstudio `9b3cbc` 源码。没有访问服务器、恢复或启动训练。复算数值和来源哈希见 [training_findings.json](training_findings.json)。

## 1. 结论与优先级

**下一轮应优先修正完整场景的初始化覆盖、过早 opacity 剔除、视角遍历和增长/细化预算，而不是再单独替换一种法线或在最终 171k 模型后继续加步数。**

Metric3D 对照已实际完成：Test PSNR 从 15.826 到 16.885 dB，传感器深度 MAE 从约0.492到0.336m，说明法线先验有用。但旧 s01 同源 Test 24 图只从18.355到18.407dB，仍明显低于 s01 的23.350dB；B0/B1容量轨迹几乎相同，最终172,425/171,417。更换先验没有解决这条共同训练轨迹的问题。[新对照总览][overview]、[质量分析][quality]

应区分三类证据：

| 证据等级 | 已看到什么 | 能得出的结论 |
| --- | --- | --- |
| 直接事件记录 | step1600净减80.11%，opacity占主要掩码，world/screen均为0 | 首次大幅减少确由低opacity与少量split父替换组成 |
| 直接训练日志 | 15k后不再增长，继续删掉14900时数量的53.13%；相同训练视角的后段PSNR也下降 | 后半程有明确的容量和训练拟合退化，需要优先处理 |
| 源码加采样记录 | 初始opacity=.1、删除阈值=.1；首删前每图仅1或2次；之后按时间顺序循环 | 过早全局筛除与局部时段统计存在明确机制风险 |
| 待完整实验验证 | 更低阈值、更充分恢复、全局随机遍历和更长增长阶段 | 有依据的修复方向，但尚不能保证质量或单独分配因果贡献 |

“点数更多”本身不是目标。目标是完整留出 RGB、深度、覆盖与表面一致性改善，计数用于解释机制。

## 2. 最大损失发生在第一次剔除

B1 step1600的完整记账为：

| 量 | 个数 |
| --- | ---: |
| 细化前 | 2,001,906 |
| split子点新增 | 40,210 |
| duplicate新增 | 257 |
| 低opacity掩码 | 1,627,349 |
| 世界尺度掩码 | 0 |
| 屏幕尺度掩码 | 0 |
| split父点替换掩码 | 20,105 |
| 删除并集 | 1,644,258 |
| 细化后 | 398,115 |

满足 `2,001,906 + 40,210 + 257 - 1,644,258 = 398,115`，净减少 **1,603,791，80.113%**。opacity与父替换重叠3,196；删除并集中排除所有父替换，仍有 **1,624,153** 个点。这比仅用最终净差更准确，不能把split父替换当作错误删除。[事件][events]、[记录实现][instrument]

全部300条事件均通过相同数量恒等式。B0首条日志的删除并集同样达到1,647,261，所以不是Metric3D独有的现象。

实际采样到step1600结束时，1,355张训练图只出现一次，123张出现两次。源码初始化opacity=.1，剔除阈值也是.1。若一个点的opacity在这点观测下稍低于初始化，它就进入全局剔除掩码。当前没有逐点可见性历史，**不能说这些点全部从未被看过、全部几何正确，或全部都应保留**；能确认的是给模型适应整场景的机会很少就发生了大规模删点。[采样][samples]、[初始化与细化][dnmodel]、[cull实现][nsmodel]

## 3. opacity重置之后反复大删，最后半程持续变小

| Step | 细化前 | 低opacity掩码 | 删除并集 | 细化后 |
| --- | ---: | ---: | ---: | ---: |
| 1,600 | 2,001,906 | 1,627,349 | 1,644,258 | 398,115 |
| 4,600 | 369,021 | 170,037 | 188,117 | 215,111 |
| 7,600 | 354,435 | 156,806 | 170,799 | 210,518 |
| 10,600 | 358,385 | 159,853 | 169,210 | 207,027 |
| 13,600 | 363,743 | 148,873 | 161,325 | 226,371 |

后四次位于各次opacity重置后第一次允许细化的位置。当前每3,000步为一个重置周期，重置发生在周期内第100步，第一次剔除发生在周期内第1,600步；仅隔1,500次训练更新，约一遍训练图。

15k以后，`stop_split_at=15000` 与 `continue_cull_post_densification=True` 使算法只删不补。step14900后有365,752点，最终171,417；后半程删除194,335，约 **53.133%**。原因掩码累计opacity169,025、world-scale25,338，二者可重叠；真正删除量以并集194,335为准。整个B1中屏幕尺寸剔除计数为0，不能把问题归给screen cull。[事件][events]

保留的5k/10k/15k/20k/25k/final checkpoints确认B0/B1非常相似：

| Step | B0 | B1 |
| --- | ---: | ---: |
| 5,000 | 228,417 | 233,461 |
| 10,000 | 356,577 | 358,385 |
| 15,000 | 347,829 | 349,480 |
| 20,000 | 217,279 | 217,832 |
| 25,000 | 189,294 | 188,848 |
| 29,999 | 172,425 | 171,417 |

这些是读取checkpoint得到的实际计数，不是文件字节数推测。[B0曲线][b0curve]、[B1曲线][b1curve]

## 4. 不仅点数下降，相同训练视角的拟合也退化

本次读回两份完整TensorBoard CSV，并把B1标量记录按实际采样帧号关联。比较 `10k≤step<15k` 与 `25k≤step<30k`：先在各区间内对每个视角取其已记录PSNR均值，再对两个区间都出现的478个视角等权平均。

- **B1：19.9687 → 15.2431 dB，变化−4.7256 dB；41/478改善，437/478下降。**
- B0按未改变的同seed采样顺序作相同索引重构：19.9490→15.0911dB。B0没有逐步实际采样CSV，因此这项映射证据弱于B1。

这项比较减少了“后面恰好抽到更难画面”的影响，但仍是**不同训练时刻的在线样本日志**，不是在两个冻结checkpoint上对完整训练集作同一评价。退化与纯剔除阶段同现，构成强定位证据；它尚未把所有因果份额分配给cull，学习率、顺序遍历和优化演化也同时变化。[B1 TensorBoard][b1tb]、[B1采样][samples]

实际B1逐步采样记录还有一项确定缺陷：第一遍1,478图是随机次序，第二遍开始按索引0到1,477顺序循环。完整30k期间，1,038图出现20次、440图出现21次；15k之前仅10或11次。与193张train的s01相比，同样30k从约155.4遍缩为20.3遍，同样15k增长窗口从77.7遍缩为10.1遍。细化统计每100次更新收集的视角至多相当于全量的6.77%，而s01是51.81%。数据划分变化不只改变总训练量，也改变每次全局细化决定依赖的视角覆盖；需要修复每轮shuffle并扩大统计窗口，不能仅在最后追加优化步数。[采样][samples]、[历史管理器][dnmanager]

已有59条 `Eval Images Metrics/rgb_psnr` 来自不断抽取的留出图。不能拿其中某个最高标量称为“最佳checkpoint”。下一完整任务直接在现有B1的5k、10k、15k、20k、25k和final六个完整checkpoint上各评价一次全部184 val，并在新的完整训练中每累计30k更新固定评价全部val；读取已有检查点后直接继续主任务，不等待审阅，也不新增短训练。[NS评价采样][nsmanager]

## 5. 稀疏配准深度没有使当前depth loss整体失效

1280×720 RGB上采用原生深度的forward投影，存在规则孔洞。Test有效0.3–6m参考约占17.16%，这个数字混合了范围、视野、无效测量与投影孔洞；不能把其余82.84%全叫“远景”。四邻域法线诊断没有有效样本，也不能推出DN深度损失没有在工作。[质量统计][quality]

固定 `EdgeAwareLogL1` 实际计算：

1. 每个像素计算 `log(1+abs(predZ-refZ))`。
2. 邻接运算只计算**RGB**梯度权重 `exp(-mean(abs(∇RGB)))`。
3. x/y项分别用当前像素的深度mask选取，再对有效元素取mean；不要求邻居深度有效，也不把无效零深度当作监督目标。

因此没有四邻域并不阻止单像素监督。损失按有效元素平均，也不是再乘0.1716把整体权重缩小。静态边界上，x仅排除最后列、y仅排除最后行；仅以已知Test像素数计算极端上界，两个方向至少还能保留99.54%/99.19%的参考项，这不是train实测计数。[损失源码][loss]

当前配置同时明确 `use_sparse_loss=false`、`use_sdf_loss=false`、`use_depth_smooth_loss=false`、`use_scale_regularization=false`，所以不能把没有开启的稀疏惩罚、SDF定期损失或尺度正则当作本次大删的直接原因；它们的源码默认值不等于实际运行开关。法线监督及法线TV开启。[实际配置][config]

其真实局限是：孔洞处没有直接传感约束；log-L1对大误差的导数按`1/(1+|误差|)`减弱；模型自身不保证在空区域长出新表面。另外，alpha=0时历史模型将depth设为本帧 `depth.detach().max()`，该像素的深度分支无法通过这个常数回传。缺失初始化或剔空的表面，不能指望单靠更多迭代必然恢复。当前未量化全Test的alpha0面积，不把这项机制夸大为全图失败比例。[模型深度分支][dnmodel]

## 6. 建议的一次完整主实验：覆盖与训练策略联合适配

当前首要目标是提高完整场景质量，已有B0/B1足以作为对照。建议一次明确标记为“自采完整场景适配”的新正式训练，保留B1法线，联合以下改动。它是预先定义的工程假设，**不是已验证最优参数，也不是论文原样30k**；最终执行协议可在发布前统一定稿。

### 初始化与完整视角

保留可靠的传感器近景初始化，同时合并**训练地图**中可靠、当前传感初始化未覆盖的SfM远景点。使用已经求得的米制全局尺度，不通过val/test补种子、不盲目把原传感深度扩到10m。可靠性、去重和尺度规则由几何方案明确冻结。当前已有近景点本身再多，也不能替代未初始化的远背景。

每个epoch重新生成固定seed下的新随机排列，无放回遍历全部1,478 train；将sampler剩余列表、RNG和epoch保存到checkpoint以便长任务续接。模型与所有留出、传感器监督定义保持一致。RGB分辨率维持1280×720。

### 增长、剔除与优化预算

| 项目 | 当前B1 | 建议主实验 | 理由 |
| --- | ---: | ---: | --- |
| 总更新 | 30,000 | **240,000** | 每图约162.4次，接近s01的155.4次；120k作为途中完整检查点 |
| opacity剔除阈值 | .1 | **.005** | 给低贡献但可能有效的点更多优化机会；有原始3DGS实现先例 |
| 重置opacity上限 | .2 | **.01** | 历史代码等于阈值×2，必须与阈值变动一起明确记录 |
| warmup | 500 | **4,000** | 初次全局清理前提供多轮曝光 |
| 细化统计窗口 | 100 | **800** | 从局部短片段扩到分散的全局视角样本 |
| 增密梯度阈值 | .0008 | **保留.0008** | 梯度累计已按可见计数归一，不能因窗口扩大乘8 |
| 重置周期 | 3,000 | **24,000** | 约16.2遍图，而不是2遍图重置一次 |
| 重置后恢复 | 约1遍图 | **至少3完整遍图** | 显式延迟新的增密/剔除，不能只拉长重置周期 |
| 停止增密 | 15,000 | **120,000** | 给几何在约81.2遍图内持续生长的机会 |
| 停止增密后剔除 | 开启 | **关闭** | 避免后半程继续无补偿缩减容量 |
| means学习率衰减终点 | 30,000 | **240,000** | 保留初末值，匹配完整优化预算 |
| SH升级间隔 | 1,000 | **保留1,000** | 没有证据要求把全部调度机械乘8 |
| screen-size停止步 | 4,000 | **保留4,000** | 避免盲目改为32k后突然启用原来没执行的screen cull |

低opacity阈值.005与重置上限.01在原始3DGS官方实现中有明确先例，但不同方法、不同数据的最优性不能直接继承。[3DGS删除阈值][gstrain]、[3DGS重置][gsmodel]

恢复条件显式定义为 `step < stop_split_at and step % reset_interval > 3*N + refine_every`，N=1478；同时保留`step > warmup_length`。必须替换旧的`N + refine_every`条件，不能新增一个配置字段却让旧分支继续执行。配合上述调度，首次候选step5600，120k前共115次候选；重置步为24800/48800/72800/96800，相应首次恢复细化29600/53600/77600/101600。原公式重置上限随新阈值给出.01；显式记录实际值，正常流程不同时保留另一套旧重置。stop_split_at及以后两条split/duplicate和post-cull分支均关闭，后120k只优化已有参数。仅修改refine800/reset24k而保留旧条件，重置24800后的首删仍会在26400，依旧只隔约1.08遍图，不能宣称已解决恢复不足。

增密统计的具体公式为 `xys_grad_norm / vis_counts * 0.5 * max(H,W)`。继承的 `after_train` 对radii>0的高斯累计投影梯度范数，并增加可见计数；计数初始化为1，所以分母精确说是`n_visible+1`，并非未经归一的800步梯度和。将统计窗口100改为800后，保留`densify_grad_thresh=.0008`；把它机械乘8会明显抑制增长。也保留`densify_size_thresh=.01`、`n_split_samples=2`。越过warmup后的细化回调末尾统计被清空，需确保实际补丁仍执行同样归一和重置，不把旧统计跨窗口叠加。[继承统计与删除源码][nsmodel]、[DN细化源码][dnmodel]

**world-scale门也必须原样保留：`step > refine_every * reset_alpha_every`。** 新配置下门限为24,000；结合三遍图恢复条件，第一次实际检查world-scale是在step29,600，而非首次opacity剔除的5,600。旧的事件记录器也遵守此门。不要在重写日志/剔除逻辑时意外提前启用world-scale。这不改变0.5m阈值，仅正确保留历史调度。[cull实现][nsmodel]、[现记录器][instrument]

世界尺度.5m阈值目前不是首轮大删主因。对新增远景种子，应记录其初始尺度和以后按种子来源、距离分组的删点数量，再决定是否需要预先明确的尺度策略；不能因“scene_scale=21”就把这个阈值误当作自动乘21，也不能无证据同时放开所有几何约束。`collider near=2/far=6`不直接裁剪此相机光栅化路径，不能依据YAML把它当作远景被截断的原因。

### 完整交付与判断

同一次训练在累计30k、60k、90k、120k、150k、180k、210k、240k更新时保留完整checkpoint并评价全部184 val，实际零基step为更新数减1；不能只用`step % 30000 == 0`而忽略终点。配置确定后仅对最终选定模型报告完整test，不用随机单图标量选模型。当前test已经多轮查看，是固定开发对照，不能重新包装为独立盲测。每个检查点看RGB、深度、alpha覆盖及高斯数；最终跑同一网格协议并报告真实覆盖。三维种子来源、低opacity删除、world-scale删除及合并原因逐次保留。

保留原始代码与B0/B1，保存独立补丁和配置差异，不从171k终点继续训练冒充新完整初始化。不设置自动结束deadline或质量不佳就中途停下来等审阅。现有资源允许尽量保持分辨率和容量，依实际GPU/RAM调整数值等价的缓存与分块；不得用强制删到某个目标点数来制造“容量稳定”。

这套联合适配若改善，说明完整场景方案有效；不能把全部收益分别记到每一个组件。30k/120k/240k同一条新轨迹的完整评价能回答预算增加是否继续受益，B1旧结果回答整套适配是否值得保留。若仍无明显改善，再依据全量可见性、跨视角残差和表面误差决定是否进入几何优化或分块方法，避免继续小改动循环。

[overview]: https://github.com/kaiwen123-yang/DN-Splatter/blob/3728dae350a240d317702d57e560554d529679c6/reports/capture-full-metric3d-30k/20261120-v1/overview.md
[quality]: https://github.com/kaiwen123-yang/DN-Splatter/blob/3728dae350a240d317702d57e560554d529679c6/reports/capture-full-quality/2026-11-21/overview.md
[events]: https://github.com/kaiwen123-yang/DN-Splatter/blob/3728dae350a240d317702d57e560554d529679c6/reports/capture-full-metric3d-30k/20261120-v1/B1_refinement.json
[instrument]: https://github.com/kaiwen123-yang/DN-Splatter/blob/3728dae350a240d317702d57e560554d529679c6/scripts/capture_metric3d_train.py
[samples]: https://github.com/kaiwen123-yang/DN-Splatter/blob/3728dae350a240d317702d57e560554d529679c6/reports/capture-full-metric3d-30k/20261120-v1/B1_samples.csv.gz
[b0curve]: https://github.com/kaiwen123-yang/DN-Splatter/blob/3728dae350a240d317702d57e560554d529679c6/reports/capture-full-metric3d-30k/20261120-v1/B0_training_curve.json
[b1curve]: https://github.com/kaiwen123-yang/DN-Splatter/blob/3728dae350a240d317702d57e560554d529679c6/reports/capture-full-metric3d-30k/20261120-v1/B1_training_curve.json
[b1tb]: https://github.com/kaiwen123-yang/DN-Splatter/blob/3728dae350a240d317702d57e560554d529679c6/reports/capture-full-metric3d-30k/20261120-v1/B1_tensorboard.csv.gz
[dnmodel]: https://github.com/maturk/dn-splatter/blob/811f5cdf342d92d12a9d9a4be185d374c430c184/dn_splatter/dn_model.py
[loss]: https://github.com/maturk/dn-splatter/blob/811f5cdf342d92d12a9d9a4be185d374c430c184/dn_splatter/losses.py
[nsmodel]: https://github.com/nerfstudio-project/nerfstudio/blob/9b3cbc79bf239eb3c69e7c288632aab02c4f0bb1/nerfstudio/models/splatfacto.py
[nsmanager]: https://github.com/nerfstudio-project/nerfstudio/blob/9b3cbc79bf239eb3c69e7c288632aab02c4f0bb1/nerfstudio/data/datamanagers/full_images_datamanager.py
[gstrain]: https://github.com/graphdeco-inria/gaussian-splatting/blob/54c035f7834b564019656c3e3fcc3646292f727d/train.py
[gsmodel]: https://github.com/graphdeco-inria/gaussian-splatting/blob/54c035f7834b564019656c3e3fcc3646292f727d/scene/gaussian_model.py

[dnmanager]: https://github.com/maturk/dn-splatter/blob/811f5cdf342d92d12a9d9a4be185d374c430c184/dn_splatter/dn_datamanager.py
[config]: https://github.com/kaiwen123-yang/DN-Splatter/blob/3728dae350a240d317702d57e560554d529679c6/reports/capture-full-metric3d-30k/20261120-v1/config.yml
