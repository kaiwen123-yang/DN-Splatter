# 2026-11-23 表面质量与 B3 实现审查

审查对象为 GitHub `main` 冻结提交 `d6405240df30c49e71365f87fdce4ec7ce7c7462`。本轮读取代码、结果 JSON 和实际预览，没有访问 AutoDL 实时进程，没有启动训练或修改远端。本报告中的“已完成”只用于仓库已有结果；仓库没有 B3 最终结果不等于服务器一定没有完成。

## 结论

**B2 的渲染已经大幅进步，最终表面仍明显不合格。当前应完成已经实施的 B3 全量表面候选及比较，收敛本轮范围，不再不断扩张新的融合模块。**

B3 的局部融合有具体证据依据：同一套原生深度和相机，单帧及短邻域融合保留的表面，在全 1,478 帧融合后大幅消失；已有四个失败体素块显示零交叉被正自由空间值覆盖。把融合限制在短时子图后再作独立观察支持的裁接，比继续全局平均有明确针对性。但它依旧依赖现有位姿和传感器证据，且边缘保护与旧面回退很保守，不能提前承诺细杆恢复或论文级网格。

最新 `d640524` 修的是裁切多边形合法性、嵌套环与三角化稳定性，**没有改变训练结果，也没有证明 B3 已提升重建质量**。

## 1. B2：不能用 PSNR 代替网格质量

B2 已完成 240,000 次更新，按验证集选择 210,000 次更新（零基 step 209,999）。选中模型 test PSNR 为 24.681 dB、SSIM 0.8044、LPIPS 0.2010，传感器参考深度 MAE 为 0.1236 m。这些是已有结果，不应再把 B2 当待执行方案。[B2 报告][b2]

同一 test 集共有 184 张图、29,100,252 个有效传感器参考像素。下表的命中率、10 cm 正确率和前景冲突率都以**全部参考像素**作分母；MAE 只在网格射线命中处计算。[完整数字][surface]

| 表面 | 命中率 | 10 cm 正确覆盖 | 前景冲突率 | 命中 MAE | 命中误差 P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| B2 原生传感器全局 TSDF | 75.53% | 5.79% | 17.57% | 1.8215 m | 6.2821 m |
| B2 模型全局 TSDF | 98.03% | 56.10% | 14.21% | 0.4778 m | 2.9800 m |
| B2 supported Poisson | 96.27% | 61.95% | 22.26% | 0.2073 m | 0.8130 m |
| B1 supported Poisson | 92.10% | 59.02% | 22.29% | 0.2051 m | 0.8461 m |

“前景冲突”具体指网格第一交点比传感器 Z 提前超过 `max(0.05 m, 0.03 × Z)`，不是主观视觉质量评分。命中率高也可能命中错误壳层；只降低命中 MAE，也可能只是删掉难处。现有指标组合有必要保留。[评估实现][evalbase]

模型深度自身的 test 10 cm 正确覆盖为 72.72%，导出的 supported Poisson 为 61.95%，差约 10.77 个百分点。此差距说明“模型到表面”链条存在重要损失，也包含模型不同视角之间的不一致；不能全部归咎于 Poisson。B2 相比 B1，网格正确覆盖仅增加约 2.92 个百分点，远小于 RGB 的改善。[11-22 复核][review22]

在四种表面共同命中的像素上，B2 supported Poisson MAE 0.2000 m、B1 0.2069 m，说明表面确有小幅改善；它远不足以支持“网格已经好了”。全局原生 sensor TSDF 已失败，不能再称它为本数据的几何质量上限。

### 已实际查看的固定预览

本轮直接打开了 `surface_table.jpg`、`surface_pillar.jpg`、`surface_chair_legs.jpg`，不是仅从数字推断。预览的网格列采用统一法线灰度着色，不是纹理图，比较重点是轮廓、连续性、薄结构和表面起伏。

- **桌面/桌沿：** 各导出仍有明显颗粒与起伏。模型 TSDF 中桌面主体较容易辨认，supported Poisson 并未得到平整、清楚的薄边。
- **柱面：** 模型 TSDF 的主体面相对连续；supported Poisson 有明显碎裂、条带、缺口，拼接齿边远未恢复。
- **椅腿：** 模型 TSDF 里仍可辨认一些交叉细杆；supported Poisson 与 B1 的细杆被颗粒表面、粘连及缺损淹没。

因此“全局指标较好的 supported Poisson”不等于“每个目标物体看起来最好”。应保留同一视角的固定物体展示，而不是只选全局表格获胜者的漂亮截图。[桌沿][table] · [柱面][pillar] · [椅腿][chair]

## 2. TSDF 消失的证据已足够具体，不必重复猜默认参数

### 已排除的简单解释

`capture_quality_tsdf.py` 明确设置了：`depth_sampling_stride=1`、体素 0.01 m、截断带 0.04 m、`depth_scale=1.0`，深度先过滤到 0.3–6 m，RGBD 的 `depth_trunc=6.000001`。原生 sensor 分支使用 848×480 深度网格及深度到 RGB 外参；model 分支用模型 Z，alpha 至少 0.5。[TSDF 实现][tsdf]

所以不能再把“Open3D 默认 stride=4 漏块”或“默认 3 m 截断”当作此次已经发现的根因。四个失败块也不是完全未分配：全局融合后其 4,096 个体素都有观测，而且全为正值。当前证据更支持**相互矛盾的观察把表面零交叉冲掉**，而不是简单的未分配体素。

### 单帧、邻域、全局之间发生了什么

11-22 原生闭环检查在同一训练相机上比较；它不依赖留出帧 PnP，因此不能用“test 定位不好”解释这些例子。[闭环结果][closure]

| 训练帧 | 单帧 TSDF 10 cm 正确覆盖 | 邻近 5/9 帧 | 全 1,478 帧 |
| --- | ---: | ---: | ---: |
| 0 | 75.49% | 73.64% | 13.61% |
| 330 | 76.38% | 58.45% | 2.45% |
| 1420 | 83.30% | 80.11% | 5.80% |

四个定向选取的缺表面体素块，在 ±2 秒局部融合中分别有 343、932、549、1,093 条正负符号交叉；全局融合后四块全部没有内部正负交叉。体素块检查还模拟了截断带触达/分配，不能脱离实现再断言它完全漏算了分配条件。[块级结果][block] · [块级实现][blockcode]

这些是强故障证据，但四块是针对失败部位选出的，不是对全场景随机抽样。它们不能单独判定最终原因究竟是位姿、动态物体、深度偏差还是遮挡关系。单帧几何自洽也只验证这条投影实现的内部闭合，不能独立证明真实标定完全正确。

全训练轨迹相邻两方向 2,954 对、8,045,328 次有效比较中，79.65% 在 10 cm 内；1303→1304 与 1305→1306 两条边明显失常，旋转约 172°，1304 与 1305 彼此却能相互支持。其余长回访样本也不是全面崩坏。因此应保留“局部错误轨迹和融合冲突”的具体判断，不宜把整条轨迹或每张深度一概判为不可用。[相邻/回访检查][neighbor]

## 3. B3 当前方案是自动几何裁接，原理成立但上限明确

已读实际实现，而非只读计划。

1. **局部融合。** 将 1,478 张训练图分为约 2 秒的 187 个时间子图，遇相邻旋转超过 90°再断开。每个子图独立生成 sensor 和 model TSDF；仍复用上面的 1 cm / 4 cm / stride 1 配置。较短融合范围有助于避免远时段冲突把局部表面冲掉。[子图实现][submaps]
2. **独立证据。** 使用原生传感器 Z，排除源子图；同子图最多一票，支持/冲突的不同票要求光心距离至少 0.1 m、视线差至少 3°，遮挡与未知不当作反对票。新增面每个顶点、边中点和重心共 7 点都须至少 2 票支持且支持数大于冲突数。[CPU 规则][votes] · [CUDA 实现][cuda]
3. **保守替换。** 模型候选以 B2 supported Poisson 作为旧面回退。新面有证据还不够，遮挡它的旧面也必须被至少 2 个独立观察判为自由空间冲突且冲突数大于支持数，才切去相应旧面。混合边界有限次细分，无法确认就保留旧面。[裁接实现][assemble]
4. **分层。** 按有向法线、面距离及子图持有者去重，保留相反朝向的薄结构两侧；新几何的位置仍来自模型 TSDF，sensor 候选另行标记，不应混称为模型质量提升。[分层实现][layers]

这不是人工画框删除难例，也不是通过改评估掩码提升数字。它是额外的传感器约束表面后处理，应在论文复现比较里标为自定义导出方法，不能把其收益全部记到原版 DN-Splatter 算法。

### 需要提前讲清的限制

- **相机误差仍共同存在。** 局部子图与独立投票仍使用已有相机；如果多个“独立”观察有同方向系统偏差，票多也不等于几何真值。1304/1305 的错误分支不能靠它们彼此支持就通过。
- **细杆可能被证据规则挡住。** `edge_distance > 2` 排除了深度断边附近；一根在原生深度图里只有几像素宽的椅腿，可能没有任何合格内部像素。原生 `fx=414.23 px`，3 m 处 1 像素约 7.2 mm，因此每侧 2 像素的内缩尺度约 1.45 cm。这只是投影尺度推导，未测量实际杆宽。再要求每面 7 点全部支持，规则对薄结构非常保守。不能提前承诺 B3 会解决椅腿和齿边。[边缘实现][repeat] · [原生标定][cal]
- **保留旧面有取舍。** 它保护覆盖率，却可能让许多已有错误面继续留下；完整的新面面积、替换面积和配对像素增损比“脚本成功退出”更能说明实际改了多少。
- **构建容差较评估宽。** 构建使用随训练重复性估计、最高 20 cm 的容差；评价仍固定原来的阈值。因此“通过构建证据”本身不保证落在 10 cm 正确范围内，这是需用全量验证判断的质量问题，不是再加一个小检查就能证明的事情。

## 4. 最新裁切补丁实际做了什么

`d640524` 只改 `scripts/capture_surface_assemble.py`。它将无效的多边形经过 `make_valid` 修复，只保留具有面积的 polygon；继续保留内孔，增加域面积守恒判断；earcut 面积检查失败时，用 Delaunay 单元与真实多边形相交后重做三角化。它还允许从一个精确列出的旧 assembly 代码哈希恢复未完成的装配，保存旧身份记录，并增加装配进度 JSON。[提交差异][commit]

旧实现已经传入外环和内环；本次并非从“完全不支持孔洞”改成“支持孔洞”，而是处理浮点裁切留下的无效嵌套环与失败回退。

补丁不改变支持投票、训练 checkpoint 或新增面的几何证据规则；面积守恒也不是表面位置正确的证明。它的价值是让冻结候选能完整生成并接受评估。

## 5. B3 的运行状态：已启动，最终结果尚无 GitHub 证据

仓库 `launch_verified.json` 的观测时间为 **2026-11-22 05:58:46 UTC（北京时间 13:58:46）**，明确记录 A1 已运行、34/187 个子图、272 张训练帧，训练决策及收益字段为空。因此不能说 B3 尚未开始。[启动收据][launch]

截至本次冻结提交，B3 报告目录仍只有该启动收据，没有完成报告、最终网格指标、选点/冻结收据或最终产物清单。今天的裁切修复表明实现继续推进，但不能据此声称现在服务器仍运行、已经完成或已经获得更好网格。

**实际主 shell 链也还只连到 A4：** `launch_capture_surface_quality.sh` 接管/恢复 A3，分别对 model 与 sensor 候选执行 184 张全 val，写 `state/frozen_pair_val.json` 后结束。它没有在此脚本里自动接上 B 诊断、条件 C 训练、最终 test 与交付。协调程序把这些接续工作列为后续实现任务。[主链][chain] · [协调程序][coordinator]

因此下一次 Codex 回传应直接说明真实阶段、已有产物和剩余步骤。不能只用“已启动后台”推定整轮会自动完成，也不能在未查现有 tmux/锁/收据的情况下重复启动新作业。

## 6. 下一步：把这一轮做成可判断的完整结果

**保持现有 B3 方案，优先交付全量 A3/A4，再完成已经约定的条件分支和最终回传；暂不新增 B4、不再另造融合方法。** 只有与这次实际阻塞直接相关的拓扑/资源/恢复修复才继续做。

本轮完成应给出下列成套结果；这些大部分已在冻结协议中，不是新增微任务：

- 同一 B2 checkpoint、同一相机下，旧 supported Poisson、model TSDF、新 model surface 的全 184 张 val 对照。sensor 结果独立呈现，不能当模型候选晋级。
- 全参考像素分母不变，同时报告命中、5/10/20 cm 正确覆盖、前景冲突、命中 MAE/P95，以及正确→错误与错误→正确的配对变化。不能通过删面、删时段、缩参考范围或换相机未披露来制造提升。
- 按冻结的 val 规则选择；所有选择、参数、训练决策冻结后一次 test。若后续确有相机修正，端到端比较与相机未改的配对子集要分开。
- 桌沿、柱面、椅腿的同相机同裁片，完整轨迹网格视频，以及新增/替换/保留面积和完整产物收据。局部看起来更好而全局变坏，或全局略升但细杆不改善，都应如实说明。

现有晋级规则要求命中率至少 90%、10 cm 正确覆盖不低于 B2、前景冲突不高于 B2；达到 70% 正确覆盖与 15% 冲突是更高目标。规则已有，不应再边看 test 边移动。[选择实现][select] · [冻结协议][protocol]

若 B3 明显改善大面和错误壳层，就把这一轮作为“冻结模型、改进导出”的有效结果。若其改善很小，尤其椅腿仍受边缘证据和表示限制，就应停止扩张裁接器，依据完整结果转向已证实的位姿/深度一致性问题或另一个表面表示做公平完整对照。**继续增加局部规则不等于重建质量必然提高。**

[b2]: https://github.com/kaiwen123-yang/DN-Splatter/blob/d6405240df30c49e71365f87fdce4ec7ce7c7462/reports/capture-full-quality-v2/20261121-v1/overview.md
[surface]: https://github.com/kaiwen123-yang/DN-Splatter/blob/d6405240df30c49e71365f87fdce4ec7ce7c7462/reports/capture-full-quality-v2/20261121-v1/surfaces_test.json
[review22]: https://github.com/kaiwen123-yang/DN-Splatter/blob/d6405240df30c49e71365f87fdce4ec7ce7c7462/reports/capture-quality-review/2026-11-22/overview.md
[table]: https://github.com/kaiwen123-yang/DN-Splatter/blob/d6405240df30c49e71365f87fdce4ec7ce7c7462/reports/capture-full-quality-v2/20261121-v1/previews/surface_table.jpg
[pillar]: https://github.com/kaiwen123-yang/DN-Splatter/blob/d6405240df30c49e71365f87fdce4ec7ce7c7462/reports/capture-full-quality-v2/20261121-v1/previews/surface_pillar.jpg
[chair]: https://github.com/kaiwen123-yang/DN-Splatter/blob/d6405240df30c49e71365f87fdce4ec7ce7c7462/reports/capture-full-quality-v2/20261121-v1/previews/surface_chair_legs.jpg
[evalbase]: https://github.com/kaiwen123-yang/DN-Splatter/blob/d6405240df30c49e71365f87fdce4ec7ce7c7462/scripts/capture_metric3d_surface_eval.py
[tsdf]: https://github.com/kaiwen123-yang/DN-Splatter/blob/d6405240df30c49e71365f87fdce4ec7ce7c7462/scripts/capture_quality_tsdf.py
[closure]: https://github.com/kaiwen123-yang/DN-Splatter/blob/d6405240df30c49e71365f87fdce4ec7ce7c7462/reports/capture-quality-review/2026-11-22/tsdf_closure_check.json
[block]: https://github.com/kaiwen123-yang/DN-Splatter/blob/d6405240df30c49e71365f87fdce4ec7ce7c7462/reports/capture-quality-review/2026-11-22/tsdf_block_check.json
[blockcode]: https://github.com/kaiwen123-yang/DN-Splatter/blob/d6405240df30c49e71365f87fdce4ec7ce7c7462/scripts/capture_tsdf_block_check.py
[neighbor]: https://github.com/kaiwen123-yang/DN-Splatter/blob/d6405240df30c49e71365f87fdce4ec7ce7c7462/reports/capture-quality-review/2026-11-22/native_neighbor_check.json
[submaps]: https://github.com/kaiwen123-yang/DN-Splatter/blob/d6405240df30c49e71365f87fdce4ec7ce7c7462/scripts/capture_surface_submaps.py
[votes]: https://github.com/kaiwen123-yang/DN-Splatter/blob/d6405240df30c49e71365f87fdce4ec7ce7c7462/scripts/capture_surface_votes.py
[cuda]: https://github.com/kaiwen123-yang/DN-Splatter/blob/d6405240df30c49e71365f87fdce4ec7ce7c7462/scripts/capture_surface_evidence.cu
[assemble]: https://github.com/kaiwen123-yang/DN-Splatter/blob/d6405240df30c49e71365f87fdce4ec7ce7c7462/scripts/capture_surface_assemble.py
[layers]: https://github.com/kaiwen123-yang/DN-Splatter/blob/d6405240df30c49e71365f87fdce4ec7ce7c7462/scripts/capture_surface_layers.py
[repeat]: https://github.com/kaiwen123-yang/DN-Splatter/blob/d6405240df30c49e71365f87fdce4ec7ce7c7462/scripts/capture_surface_repeatability.py
[cal]: https://github.com/kaiwen123-yang/DN-Splatter/blob/d6405240df30c49e71365f87fdce4ec7ce7c7462/reports/capture-bag-audit/20261119T091841Z-3982cb26/calibration.json
[commit]: https://github.com/kaiwen123-yang/DN-Splatter/commit/d6405240df30c49e71365f87fdce4ec7ce7c7462
[launch]: https://github.com/kaiwen123-yang/DN-Splatter/blob/d6405240df30c49e71365f87fdce4ec7ce7c7462/reports/capture-full-quality-v3/20261122-v1/launch_verified.json
[chain]: https://github.com/kaiwen123-yang/DN-Splatter/blob/d6405240df30c49e71365f87fdce4ec7ce7c7462/scripts/launch_capture_surface_quality.sh
[coordinator]: https://github.com/kaiwen123-yang/DN-Splatter/blob/d6405240df30c49e71365f87fdce4ec7ce7c7462/scripts/coordinate_capture_surface.py
[select]: https://github.com/kaiwen123-yang/DN-Splatter/blob/d6405240df30c49e71365f87fdce4ec7ce7c7462/scripts/capture_surface_select.py
[protocol]: https://github.com/kaiwen123-yang/DN-Splatter/blob/d6405240df30c49e71365f87fdce4ec7ce7c7462/configs/capture_surface_quality_protocol.json
