# capture-full 30k：训练与同帧效果复核

复核日期：2026-11-20（北京时间）。依据固定结果提交 `e26ae72a75997dcfd3a907aa26b6c5c0677597a5`，原始指标和大产物保持不变。本文只读取已提交报告及固定版本源码，没有访问AutoDL运行进程、没有启动新的训练。衍生数值和输入哈希见 [training_findings.json](training_findings.json)。

## 1. 结论

**全程执行完整，但重建效果相较s01在同一批画面上普遍退化。训练规模和历史采样/增密调度存在明确不匹配信号，不能把当前问题只归给Poisson，也不能仅靠在最终checkpoint后再加步数期待恢复模型容量。**

全程1478张主训练图已经全部注册，模型从2,001,906点初始化，单次连续训练累计30,000次更新，最后step29,999，最终172,425个高斯。[训练记录][training]、[运行清单][run]

最终数量是初始的 **8.613%**，净减少 **91.387%**；初始点数是s01的4.59倍，而最终高斯数只有s01最终值的39.11%。这是一项需解释的容量变化，**不是已经查明“91.387%的原始点被错误删除”**：分裂、复制、分裂父点移除及其它剔除都会改变总数，没有逐次事件记录不能把净差当作单一删除机制。[训练记录][training]、[s01训练][oldtraining]

## 2. 同源48帧：47帧PSNR下降，旧239反而略有改善

| 同源留出集合 | s01 PSNR | full PSNR | 平均变化 | PSNR下降帧数 | s01深度MAE帧均值 | full深度MAE帧均值 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Val 24帧 | 23.2433 | 18.5485 | −4.6947 dB | 24/24 | 0.13339 m | 0.33783 m |
| Test 24帧 | 23.3502 | 18.3554 | −4.9948 dB | 23/24 | 0.19838 m | 0.39941 m |

唯一PSNR改善的是原来的 `frame_00239`：13.568→14.275 dB，增加约0.707 dB。这意味着本轮主要退化已经不是旧239单点问题。Test同源帧PSNR变化中位数为−5.228 dB，最差下降9.131 dB；Val最差下降11.008 dB。数据均由冻结同源比较逐帧复算。[同源对照][same]

RGB源图一致且仍为留出；两轮位姿、初始化和优化结果不同。因此这些差值是端到端退化证据，尚不能单独区分全局几何变化与训练配置的贡献。正式全程均值也不能与s01均值直接做单变量算法对照。

## 3. 历史采样器实际只随机第一遍

实际配置写有 `train_cameras_sampling_strategy: random`、seed42；但固定历史 `DNSplatterDataManager.next_train()` 在初始列表用完后执行：

```python
self.train_unseen_cameras = [i for i in range(len(self.train_dataset))]
```

父类只在初始化时提供随机列表；DN子类后续不再调用父类的 `sample_train_cameras()`。所以第一遍随机，之后反复按文件时间顺序遍历。完整运行中没有改变此行为的源码补丁，空深度条件适配也没有触发。按1478与seed42重放第一项为1268，正好对应正式首批记录的image_idx=1268。[DN采样器][dnmanager]、[NS父类][nsmanager]、[正式配置][config]、[条件适配][conditional]

全程30k相当于20.30遍。依据单次无恢复训练与固定源码可重构：440张图应被选21次、其余1038张20次；这是静态重构，不能冒充已经逐步记录的采样日志。s01的30k则相当于155.44遍。

当前每100步清空一次增密统计。全程一个统计窗口只包含约6.77%的训练帧，且首遍以后集中在一个约25秒的局部时段；s01同样100步已相当于其训练帧的51.81%。每3000步opacity周期在全程只约2.03遍，在s01约15.54遍。**时间顺序采样与局部统计窗口、opacity重置周期的耦合是可验证机制，也是下一轮应明确处理的设计因素；当前证据不足以断言它是全部劣化的唯一根因。**

## 4. 增密机会减少，15k之后只删点不补点

DN继承并沿用的必要条件为：

```python
reset_interval = reset_alpha_every * refine_every
step < stop_split_at and step % reset_interval > num_train_data + refine_every
```

当前warmup=500、refine_every=100、reset_alpha_every=30、stop_split_at=15000，最终协议没有修改这些值。[DN模型][dnmodel]、[正式配置][config]

| 调度量 | s01：193图 | 全程：1478图 |
| --- | ---: | ---: |
| 30k对应遍数 | 155.44 | 20.30 |
| 15k停止增密时遍数 | 77.72 | 10.15 |
| 首个允许增密/剔除的step | 600 | 1600 |
| 首个候选前的遍数 | 3.11 | 1.08 |
| 总增密候选回调数 | 132 | 70 |
| 3000步opacity周期对应遍数 | 15.54 | 2.03 |

全程候选实际列在 `conditional_adaptations.json`，为1600–2900、4600–5900、7600–8900、10600–11900、13600–14900，各100步一次，共70次。它们是满足调度的机会，不等于这些步必然分裂了多少点；梯度和尺度条件还决定实际数量。

在step≥15000且 `continue_cull_post_densification=True` 时，源码只执行cull，不再split/duplicate；当前后半程每100步有一个这样的回调。因此，原30k终点后直接追加更多步、仍保持stop_split_at=15000和原30k学习率调度，**不会重新生成此前丢失的高斯**。后续若采用更长完整训练，应连同总预算、学习率与增密阶段一起明确定义，而不是简单再加一个30k。

### 可以排除一个过于笼统的归因

NS的屏幕尺寸剔除只在 `step > reset_interval` 且 `step < stop_screen_size_at` 时启用，即当前3000<step<4000。全程增密/剔除候选在2900后直接到4600，该区间没有调用cull，所以**本轮全程没有屏幕尺寸剔除执行窗口**。[NS剔除实现][nsmodel]

当前实际可能改变总数的主要机制是低opacity剔除（阈值0.1）、4600以后允许的世界尺度过大剔除（阈值0.5m）、split父点替换，以及split/duplicate新增。初始opacity也是0.1。仅凭最终点数无法量化各类贡献，不能直接把下降归因于screen size或任何一种阈值。

## 5. 已有公开记录还不能恢复完整训练曲线

公开 `training.json` 对3000个训练标量只保留count、非有限计数和首末值。Train Loss的首末为0.40456/0.26755，训练RGB PSNR首末为13.760/13.685；它们来自不同训练视角，不能用两个端点证明模型收敛或没有学习。所有公开标量有限，只说明没有被记录为NaN/Inf。[训练记录][training]

产物清单证明原始训练日志和TensorBoard事件仍在服务器：训练日志4,048,720字节，事件文件196,710,164字节，均有SHA。下列中间checkpoint也存在；当前只核对了清单，没有读取大文件内容。[产物清单][artifacts]

| Checkpoint | 文件字节数 |
| --- | ---: |
| step-000005000.ckpt | 184,256,557 |
| step-000010000.ckpt | 276,531,757 |
| step-000015000.ckpt | 270,233,453 |
| step-000020000.ckpt | 176,236,845 |
| step-000025000.ckpt | 156,087,597 |
| step-000029999.ckpt | 143,941,997 |

这些文件尺寸支持直接读取已有checkpoint元数据与原始日志，定位哪段发生容量下降；不能从文件字节数伪造精确高斯数或各类cull原因。下一轮可以在同一完整任务开始时一次抽取已有数据，生成高斯数/损失曲线和每次split/duplicate/cull的统计，再直接接续正式实验，不需要增加一个短训练或用户审阅回合。

## 6. 退化具有时间分布，不能只看最低帧

下表是全程Test，深度MAE按参考像素合并，时间相对首张深度。全程每段的留出都完成定位和评价。[分时段结果][time]

| 时段 | Test帧数 | PSNR dB | 深度MAE m |
| --- | ---: | ---: | ---: |
| 0–30 s | 14 | 19.92 | 0.389 |
| 30–60 s | 15 | 15.38 | 0.388 |
| 60–90 s | 15 | 13.79 | 0.400 |
| 90–120 s | 15 | 15.00 | 0.288 |
| 120–150 s | 15 | 12.67 | 0.557 |
| 150–180 s | 15 | 15.30 | 0.538 |
| 180–210 s | 15 | 13.88 | 0.947 |
| 210–240 s | 15 | 17.36 | 0.322 |
| 240–270 s | 15 | 15.60 | 0.666 |
| 270–300 s | 15 | 14.21 | 1.062 |
| 300–330 s | 15 | 17.78 | 0.136 |
| 330–360 s | 15 | 17.18 | 0.214 |
| 360–369.34 s | 5 | 22.07 | 0.073 |

120–150秒的RGB较差，而180–210及270–300秒的像素合并深度MAE接近或超过1m；300秒后的若干段深度明显更好。因此结果并不是“越往后越坏”的简单单调漂移，也不支持把所有失真归因于末端人员或倾斜。具体几何原因需结合轨迹与同视图图像判断。

## 7. 资源：显存有余量，主机缓存已经接近上限

正式训练2022.78秒。训练进程峰RSS约 **79.32 GiB**；采样到的cgroup峰约 **117.98 GiB**；设备显存峰6883MiB，约6.72GiB。[运行清单][run]

cgroup含文件缓存和同容器其它活动，设备显存也不是训练独占计量，不能把这些峰值直接线性外推。但当前每个训练进程的CPU全图缓存已很大，**不能仅因为96GB显存空闲较多，就用同一缓存实现同时开启两套完整训练**。要么正式训练顺序执行、利用CPU并行做日志/网格工作，要么先以保持相同帧和数值的缓存实现解决实际容量；不得通过悄悄缩短采集或改分辨率凑并发。

## 8. 下一轮的约束，而非尚未执行的结果

本报告不预先锁定先验替换或训练参数组合。完整结果已经足以支持继续做一次有明确改动和完整交付的实验。后续方案应：

1. 冻结全程输入、源身份与留出集合，保留本次30k作为完整基线。
2. 将视角遍历、每图曝光量、增密窗口和最终容量一起纳入配置，不能只看最终step数。
3. 如延长训练，明确学习率调度和允许增长的阶段；不要在已结束增密的172k模型上机械加步数。
4. 保存逐步高斯数与各类增删计数、按时间段的指标及同源48帧比较，让下一轮能检验具体假设。
5. 将模型渲染质量与点云/Poisson误差分别评价，更换先验或重建表面都不能自动解释另一环节的劣化。

所有改动方案都属于待执行实验；本报告没有声称已修复全程质量，也没有重新启动s01、小训练或参数搜索。

[training]: https://github.com/kaiwen123-yang/DN-Splatter/blob/e26ae72a75997dcfd3a907aa26b6c5c0677597a5/reports/capture-full-dn-30k/20261119-v1/training.json
[oldtraining]: https://github.com/kaiwen123-yang/DN-Splatter/blob/ee3e87bf8a63486ed96df1c0d37b6a72d30fca88/reports/capture-s01-dn-30k/20261119-v1/training.json
[run]: https://github.com/kaiwen123-yang/DN-Splatter/blob/e26ae72a75997dcfd3a907aa26b6c5c0677597a5/reports/capture-full-dn-30k/20261119-v1/run_manifest.json
[same]: https://github.com/kaiwen123-yang/DN-Splatter/blob/e26ae72a75997dcfd3a907aa26b6c5c0677597a5/reports/capture-full-dn-30k/20261119-v1/comparison_s01_same_sources.json
[config]: https://github.com/kaiwen123-yang/DN-Splatter/blob/e26ae72a75997dcfd3a907aa26b6c5c0677597a5/reports/capture-full-dn-30k/20261119-v1/config.yml
[conditional]: https://github.com/kaiwen123-yang/DN-Splatter/blob/e26ae72a75997dcfd3a907aa26b6c5c0677597a5/reports/capture-full-dn-30k/20261119-v1/conditional_adaptations.json
[artifacts]: https://github.com/kaiwen123-yang/DN-Splatter/blob/e26ae72a75997dcfd3a907aa26b6c5c0677597a5/reports/capture-full-dn-30k/20261119-v1/artifact_manifest.json
[time]: https://github.com/kaiwen123-yang/DN-Splatter/blob/e26ae72a75997dcfd3a907aa26b6c5c0677597a5/reports/capture-full-dn-30k/20261119-v1/metrics_by_time.json
[dnmanager]: https://github.com/maturk/dn-splatter/blob/811f5cdf342d92d12a9d9a4be185d374c430c184/dn_splatter/dn_datamanager.py
[dnmodel]: https://github.com/maturk/dn-splatter/blob/811f5cdf342d92d12a9d9a4be185d374c430c184/dn_splatter/dn_model.py
[nsmanager]: https://github.com/nerfstudio-project/nerfstudio/blob/9b3cbc79bf239eb3c69e7c288632aab02c4f0bb1/nerfstudio/data/datamanagers/full_images_datamanager.py
[nsmodel]: https://github.com/nerfstudio-project/nerfstudio/blob/9b3cbc79bf239eb3c69e7c288632aab02c4f0bb1/nerfstudio/models/splatfacto.py
