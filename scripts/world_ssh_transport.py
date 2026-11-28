"""Windows session-only SSH adapter for verified restore and small-report publication."""
import argparse
import ctypes as ct
import getpass
import json
import os
import shutil
import sys
import threading
from ctypes import wintypes as wt
from pathlib import Path

import paramiko


TOKEN = "Codex/DN-Splatter/interactive-world-20261128-v1"
class Credential(ct.Structure):
    _fields_ = [("flags", wt.DWORD), ("kind", wt.DWORD), ("name", wt.LPWSTR),
                ("comment", wt.LPWSTR), ("last_written", wt.FILETIME), ("size", wt.DWORD),
                ("blob", ct.c_void_p), ("persist", wt.DWORD), ("attributes_count", wt.DWORD),
                ("attributes", ct.c_void_p), ("alias", wt.LPWSTR), ("user", wt.LPWSTR)]


api = ct.WinDLL("Advapi32", use_last_error=True)
def register(args):
    if not args.host or not args.user or not 0 < args.port < 65536:
        raise ValueError("Register the authorized host, port and user")
    fields = dict(host=args.host,port=args.port,user=args.user,password=getpass.getpass("SSH password: "))
    data = json.dumps(fields).encode("utf-8")
    buffer = ct.create_string_buffer(data)
    record = Credential(kind=1, name=TOKEN, size=len(data),
                        blob=ct.cast(buffer, ct.c_void_p), persist=1,
                        user=fields["user"])
    if not api.CredWriteW(ct.byref(record), 0):
        raise ct.WinError(ct.get_last_error())
    print("Session connection ready.")


def connect():
    ptr = ct.POINTER(Credential)()
    if not api.CredReadW(wt.LPCWSTR(TOKEN), 1, 0, ct.byref(ptr)):
        raise ct.WinError(ct.get_last_error())
    try:
        settings = json.loads(ct.string_at(ptr.contents.blob, ptr.contents.size))
    finally:
        api.CredFree(ptr)
    client = paramiko.SSHClient()
    client.load_host_keys(str(Path.home()/".ssh"/"known_hosts"))
    client.connect(hostname=settings["host"], port=settings["port"], username=settings["user"],
                   password=settings["password"], allow_agent=False, look_for_keys=False,
                   timeout=15, banner_timeout=30, auth_timeout=30)
    client.get_transport().set_keepalive(25)
    return client


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("op", choices=["register", "forget", "run", "script", "put", "get", "stream"])
    parser.add_argument("items", nargs="*")
    parser.add_argument("--host")
    parser.add_argument("--user")
    parser.add_argument("--port",type=int,default=22)
    args = parser.parse_args()
    if args.op == "register":
        register(args)
        return
    if args.op == "forget":
        if not api.CredDeleteW(wt.LPCWSTR(TOKEN), 1, 0):
            raise ct.WinError(ct.get_last_error())
        return
    with connect() as client:
        if args.op in ("put", "get"):
            with client.open_sftp() as ftp:
                operation = getattr(ftp, args.op)
                for source,target in zip(args.items[::2],args.items[1::2],strict=True):
                    operation(source,target)
            return
        chan = client.get_transport().open_session(window_size=16777216)
        chan.set_combine_stderr(True)
        command = "bash -s" if args.op == "script" else args.items[0]
        chan.exec_command(command)
        def feed():
            try:
                if args.op == "script":
                    raw = Path(args.items[0]).read_bytes() if args.items else sys.stdin.buffer.read()
                    chan.sendall(raw.removeprefix(b'\xef\xbb\xbf').replace(b'\r\n',b'\n'))
                elif args.op == "stream":
                    with chan.makefile_stdin("wb") as dest:
                        shutil.copyfileobj(sys.stdin.buffer,dest,1024*1024)
            finally:
                chan.shutdown_write()
        sender=threading.Thread(target=feed)
        sender.start()
        with chan.makefile("rb") as stream:
            for chunk in iter(lambda:stream.read(32768),b""):
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
        sender.join()
        sys.exit(chan.recv_exit_status())


if __name__ == "__main__":
    main()
