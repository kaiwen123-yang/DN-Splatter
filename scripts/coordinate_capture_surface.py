"""Keep the requested B3 SSH coordination alive on Windows after the interactive handoff."""
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from capture_archive_transport import atomic
from capture_quality_receive import lock_receiver
from capture_surface_receive import DEST, ID, ROOT, STATE, document


CONTROL = ROOT / ".local/transfer" / ID
STATUS = CONTROL / "coordinator.json"
BRIEF = """
你现在是用户指定的唯一 B3 后台协调者。主对话在确认实际 tmux 计算和本进程成功接手后结束，
你的终点是完整交付或可核验的真实阻塞。不要再启动任何 AI 协调者、额外 Codex 进程或聊天。
先读 AGENTS.md、prompts/local_capture_surface_quality.md、prompts/autodl_capture_surface_quality.md、
docs/15_capture_surface_quality_plan.md、协议、复核及本 run 实际状态。直接实施缺的代码，不能只出计划。
本机仓库是工作目录，经既有 SSH 在 AutoDL 执行；在本机读取 .local/transfer/full-host 时不打印它。
可将 shell 命令以 UTF-8 传给 python .local/transfer/metric3d_remote.py。只输出项目相关状态。
服务器原协调凭据返回 401；不要再启动它，不复制本机凭据。AutoDL 的 GitHub 操作先在子 shell 用
/etc/network_turbo，检查目标用户仓库。保持一个实验作业，各重阶段顺序运行。
本地未跟踪的旧中文报告属于用户，不暂存。保全两端修改，只发布本项目小文件，不 force push。

初始交接：33a672f 实施 A1；3e70778 增加 A2 容差脚本和 Windows 接收器。实际 A1 正在
dn-capture-b3:frozen 内运行，有 B2 checkpoint、旧相机、187 个组的不可变身份、每子图模型与
原生 TSDF 及哈希。先读 export/B2/identity.json、state/B2_local_surfaces.json、各子图 receipt，
不要重复启动。A1 进行时立即实现余下真正的几何、位姿诊断、权重/纹理检查、条件训练及最终选取交付，
不要仅等待 A1。尽快建立版本化 native 持久主链并尝试它，阶段按计划和三项实测开关运行，保留恢复入口。
已有 scripts/capture_surface_repeatability.py B2|B3 可串行使用。根据实测内存/磁盘使用数值等价的
空间索引、批处理、稀疏块和线程。若其他项目占用内存，不清理、停止或调整它们，不因利用率而排队。
可基于分块/流式缓存解决技术问题。无截止时间、短试训、扫描或人工审批。正确模型表面与原生参照分别成面，
检查和保留失败。A2/A3/A4 和 B/C 不能仅生成空报告或伪造校验。条件训练复用 B2 现有正确实现，显式绑定 B3。
正式第一 batch 和细化事件、八次 full val、最终 240k 或有三项实测依据的 skipped_equivalent 按协议执行。
所有选择冻结前不看本轮 test。中文结果必须分开冻结导出和训练贡献并展示固定物体与困难视角。

Windows 已有独立、单实例、只读回传程序 scripts/capture_surface_receive.py。若它已退出可重启同一脚本。
完成后提交报告 reports/capture-full-quality-v3/20261122-v1/overview.md 和 artifacts.json（artifacts、
references）。先审查 diff 再推送用户 main。发布服务端 state/delivery_ready.json 的 schema_version 为 1，
run_id、40 位 commit、status、training_updates(0 或 240000)、selected_step、artifacts、references 齐备，
清单须与 Git 中 artifacts.json 一致。各项为 path/bytes/sha256。大件在 B3 run 内，已有 B2 引用走
B2 已有本地收据路径，包括 210k checkpoint；仅导出时引用旧 RGB 视频，新做网格视频。完整核对本地
receiver.json status=received 和回读 SHA。需要的话改进接收器同时保留现有行为。所有运行笔记保留在 run，
每次接手先看 git status、日志和阶段。若确实无技术恢复路径，写 .local/transfer/capture-full-quality-v3-20261122-v1/blocked.json，
包含 run_id、实际失败阶段、可核验的原始证据路径、已尝试措施和重新接续命令，再结束。已解决的失败不得留下该标记。
"""


def now():
    return datetime.now(timezone.utc).isoformat()


def tail_bytes(path):
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - 16000))
        return stream.read()


def current_assignment():
    # The experiment spans separate CLI invocations. Give each implementation turn an executable deliverable.
    work = [
        ("capture_surface_votes.py", "实现一个实际可供后续融合与深度权重调用的批量可见性投票模块。读取 A1 identity 的原生训练相机和 A2 tolerances，使用原生 Z 首面、断边距离、独立光心和视线角、源子图排除，分开 S/C、遮挡与未知，分组内按断边距离、残差及稳定 ID 选票。空间分块/相机裁剪，不做近似丢点。在真实已完成子图上做函数级等价性核对。"),
        ("capture_surface_fuse.py", "用已有 votes 和 A1/A2 实现可运行的 A3 模型和传感器融合命令：逐面顶点、三个中点及中心；支持与冲突；两票替换旧自由空间、补洞、其余旧面回退；法线分层与持有子图、裁接去重、统计面积；块级可恢复。包括可发现的命令用法和输出 JSON。"),
        ("capture_surface_evaluate.py", "实现 A4 和全轨迹网格评价，复用已有模块。全184个视角、固定分母、按视角/深度/时间、paired正确像素转移、共同命中、3训练相机与4个局部、固定物体图。只在全部选择冻结后允许 test。"),
        ("capture_surface_pose.py", "实际实现 B3 坏段外训练地图 1304/1305 联合重定位、独立未拟合对应和双向真实深度验收、1514 单独诊断及保留失败标记。输出独立相机、配对残差与实测开关。"),
        ("capture_surface_weights.py", "实际实现所有训练有效深度的两类独立投票、w 像素权重、留组交叉验证与按距离的诊断；使用 B2 真实原生对应冻结 RGB-Z 像素映射，保留空像素和全部帧。"),
        ("capture_surface_texture.py", "对三个物体在训练视角的原尺寸固定局部做真图-真图几何重投影、边缘偏差、高通残差分位数与跨机位清晰度判断；检查真实可见高斯归因，输出可启用的额外父点/明确实测关闭理由。"),
        ("capture_surface_train.py", "实现条件训练门、初始化、每方向带权 EdgeAwareLogL1 以及全1等价检查、可能的局部细分和 B3 正式训练入口。复用 B2 trainer，run/输入/回调/恢复全部指向 B3，使用 B2 210k 精确 val 选点。"),
        ("capture_surface_select.py", "实现最终 checkpoint 推荐 vs 网格 val 选择，验证输入冻结后再统一开发 test、B3 未改相机及旧 mesh 对照，按精确门槛与固定同分规则。"),
        ("capture_surface_publish.py", "实现完整时间轴/原隐私遮挡、固定局部、中文结果报告及 Git 与私有大件交付/引用清单，兼容 capture_surface_receive.py。"),
        ("launch_capture_surface_quality.sh", "串接全部真实阶段、计算开关、一次条件训练和交付到单实例可恢复主链，采用已运行的 A1/A2，不重复启动；部署进项目 tmux，核验后持续运行至收据。"),
    ]
    for name, task in work:
        if not (ROOT / "scripts" / name).is_file():
            return (f"本次首先交付 scripts/{name}。{task} 这是具体编码工作：请直接读必要函数并使用工具写该文件、"
                    "做语法和接口校验、审查、提交、部署，而后再继续下一项。已经核实 A1 正常，不需要再花本轮重复检查 A1。"
                    "若远端已存在等效实现，取回审阅合并并复用。主协调者是调用你的持久 Python 程序，"
                    "它独立持有锁，当前会话拥有实际实施权限。")
    return "请核对整个 native 主链实际运行状态、已实现行为和错误，修复并续接直到完成最终回传。"


def main():
    source_version = Path(__file__).stat().st_mtime_ns
    CONTROL.mkdir(exist_ok=True, parents=True)
    exe = shutil.which("codex.exe")
    if not exe:
        raise RuntimeError("Existing Windows Codex executable unavailable")
    with (CONTROL / "coordinator.lock").open("a+b") as lock:
        lock_receiver(lock)
        attempt = int(document(STATUS).get("attempt", 0)) if STATUS.exists() else 0
        while True:
            if Path(__file__).stat().st_mtime_ns != source_version:
                lock.close()
                os.execv(sys.executable, [sys.executable, __file__])
            receipt = document(STATE) if STATE.exists() else {}
            if receipt.get("run_id") == ID and receipt.get("status") in ("received", "server_blocked"):
                atomic(STATUS, dict(run_id=ID, pid=os.getpid(), utc=now(), status=receipt["status"], attempt=attempt))
                return
            blocked = CONTROL / "blocked.json"
            if blocked.is_file() and document(blocked).get("run_id") == ID:
                atomic(STATUS, dict(run_id=ID, pid=os.getpid(), attempt=attempt, utc=now(),
                                    status="blocked", evidence=blocked.name))
                return
            attempt += 1
            prefix = f"segment-{attempt:03}"
            log = CONTROL / (prefix + ".jsonl")
            last = CONTROL / (prefix + ".md")
            command = [exe, "exec", "--ephemeral", "-C", str(ROOT), "-s", "danger-full-access",
                       "-c", 'approval_policy="never"', "--color", "never", "--json", "-o", str(last), "-"]
            standalone_defaults = any(b"model_not_found" in tail_bytes(p)
                                      for p in CONTROL.glob("segment-*.jsonl"))
            if standalone_defaults:
                command.insert(2, "--ignore-user-config")
            with log.open("xb") as output, (CONTROL / (prefix+".stderr")).open("xb") as errors:
                process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.PIPE, stdout=output, stderr=errors)
                atomic(STATUS, dict(run_id=ID, pid=os.getpid(), worker_pid=process.pid, attempt=attempt,
                      utc=now(), status="coordinating", log=log.name))
                process.communicate((BRIEF + "\n\n" + current_assignment()).encode("utf-8"))
            atomic(STATUS, dict(run_id=ID, pid=os.getpid(), attempt=attempt, utc=now(), exit_code=process.returncode,
                  status="checking_delivery", log=log.name, last=last.name))
            if process.returncode:
                # A revoked credential or exhausted account cannot be repaired by restarting this client.
                tail = tail_bytes(log) + tail_bytes(CONTROL / (prefix+".stderr"))
                if any(term in tail.lower() for term in (b"token_revoked", b"refresh_token_reused",
                       b"insufficient_quota", b"unauthorized", b"invalid_api_key")):
                    atomic(STATUS, dict(run_id=ID, pid=os.getpid(), attempt=attempt, utc=now(),
                                        status="coordinator_access_blocked", log=log.name))
                    return
                if b"model_not_found" in tail and not standalone_defaults:
                    continue
            # One active coordinator at a time. A returned CLI call alone is not an experiment receipt.
            time.sleep(15 if process.returncode == 0 else 180)


if __name__ == "__main__":
    main()
