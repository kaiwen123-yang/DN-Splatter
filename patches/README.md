# Blackwell VR Room patches

| Patch | Observed behavior and change |
|---|---|
| `gsplat-1.0.0-build-jobs.patch` | gsplat v1.0.0 disabled Ninja, so its build ignored the project job budget. Both CUDA extensions now use Ninja with an absolute path to the same pinned GLM headers. The observed build selected `MAX_JOBS=21` under the 25-CPU, 120-GiB allocation. |
| `dn-811f5cd-omnidata-checkpoint.patch` | PyTorch 2.7.1 rejected the callback key `pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint` when loading the official Omnidata v2 archive. The low-resolution normal-prior loader allows that class inside a `safe_globals` context and reads the checkpoint with `weights_only=True`. Zenodo MD5 is `f2e0962ea6e248a1648f20d74b1b034d` and observed SHA-256 is `38db3f7d952f48f900f0b32d6cce6749e41641045ebc5713cf29618f54aff3bb`. |
| `omnidata-tools-0.0.23-backbone.patch` | The library downloaded an ImageNet bootstrap during model construction. An optional constructor argument lets the DN loader omit that download when it immediately installs the complete verified Omnidata state with strict key matching. Every other caller retains the upstream default. |
| `dn-811f5cd-poisson-threads.patch` | The selected DN exporter omitted the Open3D thread argument. It now passes `n_threads=DN_CPU_BUDGET` and reports the number of samples before the existing outlier filter. The saved PLY supplies the post-filter count. |

The execution checkout uses DN-Splatter `811f5cd`, gsplat `c7b0a38` and GLM `45008b2`. The installer checks each patch against its source tree. Runtime checks and measured outcomes are recorded in the run report.
