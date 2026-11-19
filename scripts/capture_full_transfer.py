"""Transfer verified packet members in restartable tar streams over an existing SSH alias."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import tarfile

from capture_rectify import read, sha, verify


def transfer(host, packet, remote):
    verify(packet)
    identity = sha(packet / "manifest.json")
    members = read(packet / "manifest.json")["files"]
    remote = str(PurePosixPath(remote))
    q = shlex.quote(remote)
    marker = shlex.quote(remote + ".parts/" + identity)
    ssh = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=12", host]
    subprocess.run([*ssh, f"mkdir -p {q} {marker}"], check=True)
    for start in range(0, len(members), 128):
        group = members[start:start+128]
        done = f"{marker}/{start}"
        if subprocess.run([*ssh, f"test -f {done}"]).returncode == 0:
            continue
        proc = subprocess.Popen([*ssh, f"tar -xmf - -C {q}"], stdin=subprocess.PIPE)
        with proc.stdin, tarfile.open(fileobj=proc.stdin, mode="w|") as tape:
            for row in group:
                tape.add(packet / row["path"], arcname=row["path"], recursive=False)
        if proc.wait():
            raise RuntimeError(f"Incomplete stream at member {start}")
        checks = "".join(f'{r["sha256"]}  {r["path"]}\n' for r in group)
        subprocess.run([*ssh, f"cd {q} && sha256sum --quiet -c - && touch {done}"],
                       input=checks.encode(), check=True)
        print(f"Verified {min(start+128,len(members))}/{len(members)}", flush=True)
    content=(packet / "manifest.json").read_bytes()
    subprocess.run([*ssh, f"cd {q} && cat > manifest.pending && "
                    f"echo '{identity}  manifest.pending' | sha256sum --quiet -c - && mv manifest.pending manifest.json"],
                   input=content, check=True)
    print("Transferred manifest",identity,flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("host",help="Existing authorized SSH alias")
    args=parser.parse_args()
    paths=read(Path("configs/capture_full_test_protocol.json"))["paths"]
    transfer(args.host, Path(paths["local_transfer"]), paths["server_project_root"]+"/"+paths["geometry_run"]+"/transfer")
