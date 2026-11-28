"""Serve the four verified replay files on loopback with browser seek support."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re


MEDIA=Path(__file__).resolve().parents[1]/".local/runs/lingbot-depth-map-20261128-v1/media"
FILES={"depth.mp4":MEDIA/"depth/depth_hall.mp4",
       **{key+".mp4":MEDIA/"map"/key/"trajectory_cloud.mp4" for key in
          ("short_train_256","full_train_1478","full_all_1846_exploratory")}}


class Replays(BaseHTTPRequestHandler):
    def do_HEAD(self):
        self.serve(False)

    def do_GET(self):
        self.serve(True)

    def serve(self,body):
        path=FILES.get(self.path.lstrip("/"))
        if path is None or not path.is_file():
            self.send_error(404); return
        size=path.stat().st_size
        start,end=0,size-1
        header=self.headers.get("Range")
        if header:
            match=re.fullmatch(r"bytes=(\d*)-(\d*)",header)
            if not match or not any(match.groups()):
                self.send_error(416);return
            a,b=match.groups()
            start=max(0,size-int(b)) if not a else int(a)
            end=min(end,int(b)) if a and b else end
            if start>end or start>=size:
                self.send_response(416);self.send_header("Content-Range",f"bytes */{size}");self.end_headers();return
        self.send_response(206 if header else 200)
        self.send_header("Content-Type","video/mp4")
        self.send_header("Accept-Ranges","bytes")
        if header:self.send_header("Content-Range",f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length",str(end-start+1));self.end_headers()
        if body:
            with path.open("rb") as f:
                f.seek(start)
                left=end-start+1
                try:
                    while left:
                        data=f.read(min(left,512*1024))
                        if not data:break
                        self.wfile.write(data);left-=len(data)
                except (BrokenPipeError,ConnectionResetError):
                    pass


if __name__=="__main__":
    ThreadingHTTPServer(("127.0.0.1",18789),Replays).serve_forever()
