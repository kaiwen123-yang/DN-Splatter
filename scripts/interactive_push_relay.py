"""Stream only pinned public Hugging Face inputs to a server without a local data copy."""

import argparse
import importlib.util
import json
import shlex
import struct
import sys
from pathlib import Path

from interactive_push_assets import SOURCES, fetch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", choices=SOURCES)
    p.add_argument("--session-module", type=Path, required=True,
                   help="Local authorized SSH connector exporting connect()")
    p.add_argument("--remote-root", required=True)
    p.add_argument("--run-id", default="interactive-world-20261128-v1")
    args = p.parse_args()
    spec = importlib.util.spec_from_file_location("dn_local_session", args.session_module)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    kind, repo, revision = SOURCES[args.source]
    source_url = "https://huggingface.co/" + ("datasets/" if kind == "datasets" else "") + repo + "/resolve/" + revision + "/"
    metadata_url = f"https://huggingface.co/api/{kind}/{repo}/revision/{revision}?blobs=true"
    root = shlex.quote(args.remote_root)
    run = Path(".local/runs", args.run_id).as_posix()
    d = shlex.quote(f"{run}/e_interaction/receipts/{args.source}-relay")
    out = shlex.quote(f"{run}/e_interaction/assets/{args.source}")
    cmd = (f"cd {root} && mkdir -p {d} && "
           f"flock -n {d}/lock env DN_PUBLIC_INPUT_PIPE=1 /root/miniconda3/bin/python -u "
           f"scripts/interactive_push_assets.py {args.source} {out}; r=$?; echo $r > {d}/status; exit $r")
    with module.connect() as client:
        chan = client.get_transport().open_session()
        chan.set_combine_stderr(True)
        chan.exec_command(cmd)
        count = total = 0
        with chan.makefile("rb") as reader:
            for line in reader:
                if not line.startswith(b"HF_FETCH "):
                    print(line.decode("utf8", "replace").rstrip(), flush=True)
                    continue
                url, interval = json.loads(line[len("HF_FETCH "):])
                if url != metadata_url and not url.startswith(source_url):
                    raise ValueError("Unexpected request outside frozen public source")
                data = fetch(url, interval)
                chan.sendall(struct.pack("!Q", len(data)))
                chan.sendall(data)
                count += 1
                total += len(data)
                if count == 1 or count % 25 == 0:
                    print(f"{args.source}: {count} requests, {total} B forwarded", flush=True)
        sys.exit(chan.recv_exit_status())


if __name__ == "__main__":
    main()
