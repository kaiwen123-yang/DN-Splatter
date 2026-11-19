"""Write the reviewable Chinese result page from verified run evidence."""
from capture_rectify import read
from capture_s01_inputs import REPORT
from vr_room_run import RUN


def write_overview(metrics, mesh, trained, stages):
    rows = []
    depths = []
    for split, m in metrics.items():
        rgb = m["rgb_frame_means"]
        rows.append(
            f"| {split} | {m['frame_count']} | {rgb['psnr']:.4f} | {rgb['ssim']:.5f} | {rgb['lpips']:.5f} | {m['reference_pixels']:,} | {m['invalid_prediction_pixels']:,} | {m['reference_alpha_zero']:,} |"
        )
        for scope, key in (("等权帧均值", "depth_frame_means"), ("全像素聚合", "depth_pixel_pooled")):
            d = m[key]
            values = " | ".join(
                f"{d[k]:.5f}" if d else "未定义"
                for k in ("MAE_m", "RMSE_m", "AbsRel", "absolute_error_P50_m", "absolute_error_P95_m")
            )
            depths.append(f"| {split} | {scope} | {values} |")
    report = [
        "# Gemini 335 首段 DN-Splatter 基线",
        "",
        f"约 48 秒的 capture-s01 完成 {trained['cumulative_updates']:,} 次更新，最终零基 step {trained['step']}，含 {trained['final_gaussians']:,} 个高斯。val、test 和原始 Poisson 使用同一最终模型。",
        "",
        f"最终 checkpoint SHA256 为 `{trained['checkpoint_sha256']}`。",
        "",
        "## 留出渲染与传感深度一致性",
        "",
        "| 划分 | 帧数 | PSNR (dB) | SSIM | LPIPS | 参考像素 | 无效预测 | 参考内零累积 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        *rows,
        "",
        "| 划分 | 聚合方式 | MAE (m) | RMSE (m) | AbsRel | 绝对差 P50 (m) | 绝对差 P95 (m) |",
        "|---|---|---:|---:|---:|---:|---:|",
        *depths,
        "",
        "RGB 指标逐帧等权。深度参照是同一采集在 0.3–6 m 内的有效传感器 Z；按有效参考像素计分，保留累积不透明度为零的像素。历史渲染器在这些位置填入本帧最大 Z。存在无效预测时，完整参考指标记为未定义，逐帧 JSON 另列条件结果与分母。训练深度正值覆盖的中位数为 16.28%。",
        "",
        "[逐帧 val 结果](metrics_val.json) · [逐帧 test 结果](metrics_test.json)",
        "",
        "各图上排依次为原始 RGB、渲染 RGB 和法线，下排为传感器 Z、预测 Z 和绝对残差。深度色标为 0.3–6 m，绝对残差色标为 0–0.5 m，空洞用黑色表示。",
        "",
    ]
    for split in ("val", "test"):
        for p in sorted((REPORT / "previews").glob(split + "_*.jpg")):
            report.extend([f"![{p.stem}](previews/{p.name})", ""])
    report.extend(
        [
            "## 原始 Poisson 网格",
            "",
            f"193 个训练视角共采样 {mesh['samples']:,} 点，其中 {mesh['sampled_alpha_zero']:,} 点累积不透明度为零，{mesh['sampled_outside_sensor_range']:,} 点的模型深度超出传感器监督范围。原始网格有 {mesh['vertices']:,} 个顶点、{mesh['faces']:,} 个面和 {mesh['connected_components']:,} 个连通分量。采样法线中，非有限值有 {mesh['sampled_nonfinite_normals']:,} 个，零向量有 {mesh['sampled_zero_normals']:,} 个。",
            "",
            "使用历史 depth 9、最低 1% 密度顶点裁剪、25 线程。远景和闭合补面缺少独立参考。下列固定训练视角局部图并列展示原始 RGB、网格顶点颜色与面法线明暗，供复核桌沿、柱面厚度、折叠椅腿及其间隙。",
            "",
            "![原始网格全景](previews/mesh_global.jpg)",
            "",
            "![桌沿局部](previews/mesh_table.jpg)",
            "",
            "![柱面局部](previews/mesh_pillar.jpg)",
            "",
            "![椅腿局部](previews/mesh_chair_legs.jpg)",
            "",
            "[网格计数、包围盒与局部相机](mesh_validation.json)",
            "",
            "## 输入与宽基线局限",
            "",
            "正式输入副本保留 1280×720、193/24/24 划分、既有米制位姿以及 436,361 点训练初始化。生成 241 组 Omnidata PNG 与 NPY，正式读取器逐帧检查 RGB、深度、法线、K 与 c2w。原几何成员在发布前再次校验。",
            "",
            "40 个有向比较使用固定 640 网格。164→179 的局部平滑支持有 33,000 像素，绝对 Z 差中位数 3.60 cm、P95 2.25 m；反向有 31,652 像素，中位数 3.34 cm、P95 0.196 m。宽基线长尾具有方向性。val 和 test 位于同一轨迹，曾用于几何诊断。数据没有外部真值，结论限于这段采集。",
            "",
            "[跨视角分母及正负候选](cross_view_diagnostic.json) · [六幅双向对照图](previews/) · [法线权重与输入指纹](normal_prior.json) · [正式读取](reader_check.json)",
            "",
            "## 耗时、资源与大产物",
            "",
            "峰值来自采样，GPU 为整卡，RAM 为整个容器。短阶段缺少采样时显示空白。",
            "",
            "| 阶段 | 秒 | 采样峰值显存 (MiB) | 采样峰值容器内存 (GiB) |",
            "|---|---:|---:|---:|",
        ]
    )
    for s in stages:
        ram = s["peak_sampled_cgroup_memory_bytes"]
        report.append(
            f"| {s['stage']} | {s['duration_seconds']:.1f} | {s['peak_sampled_gpu_used_mib_device'] if s['resource_samples'] else ''} | {format(ram/2**30,'.2f') if ram else ''} |"
        )
    report.extend(
        [
            "",
            f"服务器运行目录为 `{RUN}`。完整法线、48 帧 RGB/Z/法线渲染、所有 checkpoint、采样 PLY、原始网格及日志逐文件登记大小与 SHA256。",
            "",
            "[产物清单](artifact_manifest.json) · [训练与损失](training.json) · [阶段记录](run_manifest.json) · [可重放命令](commands.sh) · [正式配置](config.yml)",
        ]
    )
    (REPORT / "overview.md").write_text("\n".join(report) + "\n", encoding="utf-8")
