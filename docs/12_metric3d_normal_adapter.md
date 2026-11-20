# Metric3D v2 法线接入约定

状态：源码核验与执行约定，尚未推理或训练。用于 capture-full 的完整 30k 法线替换实验；基线为 `e26ae72a75997dcfd3a907aa26b6c5c0677597a5`。几何、1478/184/184 划分、传感器 Z（0.3–6 m）、初始化、采样顺序和训练调度沿用基线。

## 1. 固定模型与输入

仅用 **Metric3D v2 ViT-Large**：源码 `eb5b6fac0dc155e4e52f576e304fbf11655ff339`，入口 `metric3d_vit_large`，配置 `vit.raft5.large.py`，官方权重 `metric_depth_vit_large_800k.pth`。[1]、[2] 下载后记录大小、SHA256、加载键差异；权重 URL 的 `main` 可变。

输入使用基线的 1846 张去畸变 RGB 及 K。内部预处理不同于 Omnidata 的 384×384，属于法线方法变化；落盘网格、原始内参和读取规则保持一致。

按官方 hub 的 RGB 预处理：[1]

- 输入尺寸按 **H×W = 616×1064**；`s=min(616/H,1064/W)`，OpenCV `INTER_LINEAR` 缩放到 `(int(W*s),int(H*s))`，正数 `int` 截断，不四舍五入。
- 当前 720×1280 输入得到 `s=0.83125`、598×1064；按上下左右顺序 padding 为 `[9,9,0,0]`。不得直接把整图拉伸到 616×1064。
- RGB 常数填充/均值为 `[123.675,116.28,103.53]`，标准差 `[58.395,57.12,57.375]`；按 0–255 浮点 RGB 做标准化，`eval()`、无梯度推理。
- 记录原 K、s、整数尺寸、padding 和 `K_resize`。依官方令 `fx'=s*fx, fy'=s*fy, cx'=s*cx, cy'=s*cy`；padded 网格的主点加左/上 padding。保留高度截断记录。派生 K 仅供推理审计；训练仍读基线原 K。

## 2. 法线后处理与唯一一次坐标变换

`prediction_normal[:, :3]` 是有符号 XYZ；第 4 通道是 concentration 参数 κ。[1]、[3] 去 padding，再以 bilinear、`align_corners=False`、`antialias=False` 恢复到 720×1280，逐像素 L2 单位化。记录插值及非有限/零长度数量，不填补异常。网格、双线性恢复和 PNG 量化沿用基线约定；单位化是本适配的后处理。

官方反投影采用 `K^-1[u,v,1]`，法线朝相机、`dot(n,xyz)<=0`；一致性损失直接比较该法线与预测 XYZ。[4]、[5] 原始 `n_cv` 的轴是 x 右、y 下、z 前；正对墙约为 `[0,0,-1]`。

历史 DN `811f5cdf342d92d12a9d9a4be185d374c430c184` 的模型把朝相机的世界法线乘 OpenGL c2w 的旋转，监督坐标应为：[6]

```text
D = diag(1, -1, -1)
n_dn = D n_cv
target_dn = (n_dn + 1) / 2
```

当前 `normal_format=opengl`、相机坐标法线读取分支**已经执行 D**。[7] 所以保存原始 CV 法线，不提前翻 Y/Z：

```text
encoded = clip((n_cv + 1) / 2, 0, 1)
png_rgb = (encoded * 255).astype(uint8)   # 截断，与基线一致
```

用 PIL RGB 保存；如用 OpenCV，写入前明确 RGB→BGR。官方 hub 的 `cv2.imwrite` 是可视化示例，直接照搬会让 PIL 训练读取时交换 X/Z。不要预翻坐标后又让 loader 翻一次，也不要根据其注释中的 OpenGL/OpenCV 名称猜方向。

一次性用实际 DN loader 读回全部 PNG，比较输出与 `((2*png_rgb/255-1) @ D+1)/2`。读取器有“整图最小值<0 才重新编码”的历史条件；读回发现不一致须记录并修正适配错误，无需小训练。

原始 XYZ float32 NPY、κ 与元数据留档；**训练只读 PNG**，历史 NPY 分支无条件除以 255，不能直接喂有符号浮点数组。[7] κ 不是已校准概率，本轮不加权、不阈值筛选，避免引入第二个训练变量。

## 3. 米制深度仅作诊断

官方 hub 的返回深度先去 padding、双线性恢复原尺寸，再**仅一次**乘 `resized_fx/1000 = s*fx/1000`。[1] 使用基线去畸变 RGB 网格的 fx；不再乘 s，不乘 SfM 全局尺度，也不重新应用模型内部深度归一化。该尺度不用于法线。

分别保存未裁剪的规范深度、米制诊断深度和尺度元数据。若另存官方 0–300 m 可视化截断版，明确区分。本轮传感器监督文件与 0.3–6 m 参考掩码保持基线哈希；Metric3D 深度不替换监督、不补洞、不参与位姿或初始化。

## 4. 环境与实现来源

基线环境记录为 PyTorch `2.7.1+cu128`。复用已通过完整训练的 Blackwell 栈；Metric3D 在项目内独立克隆环境中补足实际依赖，保留版本清单。不得照装其旧 `requirements_v2.txt` 中的 torch 2.0.1、torchvision 0.15.2、xformers 0.0.21，或改变 DN 环境。[8] 优先 AutoDL 官方学术加速获取 GitHub，固定提交与文件哈希。推理结束释放资源再训练。

ICCV 2025 官方 EmbodiedSplat README 指向 `gchhablani/dn-splatter` 的 `embodied_splat` 分支。核验固定提交 `69c2a39010a95a8e257382ee1cd23633a1b52e1b`：预训练法线脚本有 `metric3d_v2` 路由，但导入的 `dn_splatter/scripts/metric3d_normal.py` 不在完整仓库树，直接读取返回 404。[9] 因此只作为路线依据，不能声称有完整可直接执行的官方 wrapper；本适配以以上源码数值链为依据。

[1]: https://github.com/YvanYin/Metric3D/blob/eb5b6fac0dc155e4e52f576e304fbf11655ff339/hubconf.py
[2]: https://github.com/YvanYin/Metric3D/blob/eb5b6fac0dc155e4e52f576e304fbf11655ff339/mono/configs/HourglassDecoder/vit.raft5.large.py
[3]: https://github.com/YvanYin/Metric3D/blob/eb5b6fac0dc155e4e52f576e304fbf11655ff339/mono/model/decode_heads/RAFTDepthNormalDPTDecoder5.py
[4]: https://github.com/YvanYin/Metric3D/blob/eb5b6fac0dc155e4e52f576e304fbf11655ff339/training/mono/model/losses/depth_to_normal.py
[5]: https://github.com/YvanYin/Metric3D/blob/eb5b6fac0dc155e4e52f576e304fbf11655ff339/training/mono/model/losses/NormalBranchLoss.py
[6]: https://github.com/maturk/dn-splatter/blob/811f5cdf342d92d12a9d9a4be185d374c430c184/dn_splatter/dn_model.py
[7]: https://github.com/maturk/dn-splatter/blob/811f5cdf342d92d12a9d9a4be185d374c430c184/dn_splatter/data/dn_dataset.py
[8]: https://github.com/YvanYin/Metric3D/blob/eb5b6fac0dc155e4e52f576e304fbf11655ff339/requirements_v2.txt
[9]: https://github.com/gchhablani/dn-splatter/blob/69c2a39010a95a8e257382ee1cd23633a1b52e1b/dn_splatter/scripts/normals_from_pretrain.py
