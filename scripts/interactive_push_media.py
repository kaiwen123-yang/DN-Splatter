"""Synchronized complete real PushT replay videos and one-minute held-out preview."""

import argparse
import json
from pathlib import Path
import os

import av
import cv2
import numpy as np

from interactive_push_assets import file_hash,write_json
from interactive_push_track import Tracker,front_file,images


class Writer:
    def __init__(self,path):
        self.path=path;self.pending=path.with_suffix(".pending.mp4")
        self.file=av.open(str(self.pending),"w")
        self.video=self.file.add_stream("libx264",rate=30)
        self.video.width=960;self.video.height=266;self.video.pix_fmt="yuv420p"
        self.video.options={"crf":"23","preset":"veryfast","threads":"1"}
        self.frames=0

    def put(self,image):
        im=av.VideoFrame.from_ndarray(image,format="bgr24")
        for packet in self.video.encode(im):self.file.mux(packet)
        self.frames+=1

    def close(self):
        for packet in self.video.encode():self.file.mux(packet)
        self.file.close();os.replace(self.pending,self.path)
        with av.open(str(self.path)) as reader:
            if reader.streams.video[0].frames!=self.frames:raise ValueError("Encoded frame count changed")
        return {"path":str(self.path),"frames":self.frames,"bytes":self.path.stat().st_size,"sha256":file_hash(self.path)[0]}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("root",type=Path);a=p.parse_args()
    r=a.root;track=r/"track_v2";out=r/"media";out.mkdir(exist_ok=True)
    cv2.setNumThreads(1)
    tr=Tracker(json.loads((track/"tracker_freeze.json").read_text()))
    roles=json.loads((r/"prepare/split.json").read_text())["roles"]
    preview=Writer(out/"heldout_60s.mp4");manifest=[]
    for e in range(60):
        real=np.load(track/"labels"/f"episode_{e:06d}.npz")
        modes=[np.load(r/"calibration/replay"/m/"C3"/f"episode_{e:06d}.npz") for m in ("open_loop","causal_online")]
        video=Writer(out/f"episode_{e:06d}.mp4")
        for i,im in enumerate(images(front_file(r/"assets/pusht",e))):
            ok=bool(real["object_valid"][i]);ref=real["object_xytheta"][i]
            pred=[q["predicted_pre_observation"][i] for q in modes]
            panels=[im, tr.outline(im,pred[0],True),tr.outline(im,pred[1],True),tr.outline(im,ref,ok)]
            labels=[f"Real front  e{e} {i/30:.2f}s","C3 frozen open loop","C3 online prior", "Shared RGB reference"]
            if ok:
                err=[np.linalg.norm(q[:2]-ref[:2])*100 for q in pred]
                labels[1]+=f" {err[0]:.1f}cm";labels[2]+=f" {err[1]:.1f}cm"
            else: labels[3]+=" unavailable"
            xy=real["action_command"][i]
            cv2.putText(panels[0],f"xy {xy[0]:.3f} {xy[1]:.3f}m",(4,229),0,.39,(245,245,245),1)
            panel=np.zeros((266,960,3),np.uint8);panel[26:]=np.hstack(panels)
            for j,s in enumerate(labels):cv2.putText(panel,s,(j*240+4,18),0,.43,(240,240,240),1)
            video.put(panel)
            if e in roles["test"] and i<150:preview.put(panel)
        manifest.append(video.close())
        real.close()
        for m in modes:m.close()
    write_json(out/"manifest.json",{"full_episodes":manifest,"heldout_preview":preview.close(),
               "layout":"same real front video timestamp, frozen predictions, causal prior before the observation, independent RGB label"})


if __name__=="__main__":
    main()
