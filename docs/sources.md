# 官方来源与审阅范围

核查日期：2026-11-18（Asia/Shanghai）。本仓库计划依据论文与源码静态审阅；目标服务器的兼容性和训练尚待实测。

## 论文与数据

- [DN-Splatter 作者项目页](https://maturk.github.io/dn-splatter/)
- [WACV 2025 论文页面](https://openaccess.thecvf.com/content/WACV2025/html/Turkulainen_DN-Splatter_Depth_and_Normal_Priors_for_Gaussian_Splatting_and_Meshing_WACV_2025_paper.html)
- [WACV 2025 补充材料](https://openaccess.thecvf.com/content/WACV2025/supplemental/Turkulainen_DN-Splatter_Depth_and_WACV_2025_supplemental.pdf)
- [MuSHRoom 官方仓库](https://github.com/TUTvision/MuSHRoom)
- [Nerfstudio 数据约定](https://docs.nerf.studio/quickstart/data_conventions.html)

主文第4.4/4.5节、实验Implementation details、Table4；补充A.1/A.2和B.2为首轮关键协议来源。参考论文值只用于明确比较目标，不能填入实测结果。

## 固定源码

- [官方仓库](https://github.com/maturk/dn-splatter)
- [历史候选811f5cd](https://github.com/maturk/dn-splatter/tree/811f5cdf342d92d12a9d9a4be185d374c430c184)
- [历史模型与损失路径](https://github.com/maturk/dn-splatter/blob/811f5cdf342d92d12a9d9a4be185d374c430c184/dn_splatter/dn_model.py)
- [历史MuSHRoom解析器](https://github.com/maturk/dn-splatter/blob/811f5cdf342d92d12a9d9a4be185d374c430c184/dn_splatter/data/mushroom_dataparser.py)
- [历史Poisson导出](https://github.com/maturk/dn-splatter/blob/811f5cdf342d92d12a9d9a4be185d374c430c184/dn_splatter/export_mesh.py)
- [历史Faro评估](https://github.com/maturk/dn-splatter/blob/811f5cdf342d92d12a9d9a4be185d374c430c184/dn_splatter/data/mushroom_utils/eval_faro.py)
- [当前main的审计快照97588b4](https://github.com/maturk/dn-splatter/tree/97588b4290128ce7ba6fdbfaac3020b42b17de4c)
- [当前main的正则策略](https://github.com/maturk/dn-splatter/blob/97588b4290128ce7ba6fdbfaac3020b42b17de4c/dn_splatter/regularization_strategy.py)
- [Nerfstudio1.1.3训练和续训](https://github.com/nerfstudio-project/nerfstudio/blob/v1.1.3/nerfstudio/engine/trainer.py)
- [Nerfstudio1.1.3评估入口](https://github.com/nerfstudio-project/nerfstudio/blob/v1.1.3/nerfstudio/scripts/eval.py)

## Blackwell和工具链

- [NVIDIA计算能力表：RTX PRO6000 Blackwell为12.0](https://developer.nvidia.com/cuda/gpus)
- [CUDA12.8发布说明](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/)
- [PyTorch2.7发布说明：Blackwell和cu128](https://pytorch.org/blog/pytorch-2-7/)
- [PyTorch历史安装组合](https://pytorch.org/get-started/previous-versions/)
- [PyTorch2.7.1扩展编译](https://github.com/pytorch/pytorch/blob/v2.7.1/torch/utils/cpp_extension.py)
- [gsplat1.0.0构建正常与legacy两套后端](https://github.com/nerfstudio-project/gsplat/blob/v1.0.0/setup.py)
- [DN-Splatter作者确认旧gsplat接口限制，Issue106](https://github.com/maturk/dn-splatter/issues/106)

## AutoDL

- [官方学术资源加速：source /etc/network_turbo](https://www.autodl.com/docs/network_turbo/)
- [官方Git说明](https://www.autodl.com/docs/git/)

AutoDL文档列出GitHub及HuggingFace相关支持域，也说明服务不保证稳定且可能影响其他网络。按用户要求优先使用内置服务，在本项目子shell限定作用域；不据此臆造其他域名的代理支持。
