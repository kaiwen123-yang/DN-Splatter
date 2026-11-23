"""Bound B3 host memory using exactly reversible representations of decoded training images."""
from pathlib import Path

import numpy as np
import torch


def reclaimable_budget():
    cg=Path('/sys/fs/cgroup')
    hard=int((cg/'memory.max').read_text().strip())
    usage=int((cg/'memory.current').read_text())
    stats={k:int(v) for k,v in (l.split() for l in (cg/'memory.stat').read_text().splitlines())}
    return hard-usage+max(0,stats.get('file',0)-stats.get('shmem',0)-stats.get('unevictable',0))


class CompactViews:
    """Expand the same full-resolution CPU tensors at their original dataset indices."""
    def __init__(self,dataset,dtype,scene):
        self.rows=[]
        self.uncompressed_bytes=0
        self.resident_bytes=0
        self.file_backed_bytes=0
        for i in range(len(dataset)):
            row=dataset.get_data(i,image_type=dtype)
            if row['image'].shape[:2]!=(int(dataset.cameras.height[i]),int(dataset.cameras.width[i])):
                raise ValueError('B3 cache camera/image mismatch')
            packed={}
            for key,t in row.items():
                if not torch.is_tensor(t):
                    packed[key]=('literal',t)
                    continue
                self.uncompressed_bytes+=t.numel()*t.element_size()
                # A uint8 code is kept only if decoding gives the exact original float tensor.
                if key in ('image','normal') and t.dtype==torch.float32:
                    code=(t*255).round().clamp(0,255).to(torch.uint8)
                    if torch.equal(code.float()/255,t):
                        self.resident_bytes+=code.numel()
                        packed[key]=('byte',code)
                        continue
                if key=='sensor_depth':
                    name=dataset.image_filenames[i].stem+'.npy'
                    path=scene/'depth'/name
                    value=np.load(path,mmap_mode='r',allow_pickle=False)
                    if value.size==t.numel() and value.dtype==t.numpy().dtype and np.array_equal(value.reshape(t.shape),t.numpy()):
                        packed[key]=('depth',(path,tuple(t.shape)))
                        self.file_backed_bytes+=value.nbytes
                        continue
                self.resident_bytes+=t.numel()*t.element_size()
                packed[key]=('literal',t)
            self.rows.append(packed)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self,index):
        result={}
        for key,(mode,data) in self.rows[index].items():
            if mode=='byte':
                result[key]=data.float()/255
            elif mode=='depth':
                path,shape=data
                result[key]=torch.from_numpy(np.load(path,mmap_mode='c',allow_pickle=False).reshape(shape))
            else:
                result[key]=data
        return result


def bind(manager,scene,output,write):
    budget=reclaimable_budget()
    # The B2 run measured approximately 79 GiB peak host RSS with the native full cache.
    if budget>=80*1024**3:
        write(output,dict(mode='native',estimated_reclaimable_bytes=budget))
        return None
    observations=dict(mode='lossless_compact',estimated_reclaimable_bytes=budget,splits={})
    write(output,observations)
    from unittest.mock import patch
    def load(dm,split,cache_images_device):
        dataset=dm.train_dataset if split=='train' else dm.eval_dataset
        dist=dataset.cameras.distortion_params
        if cache_images_device!='cpu' or dist is not None and torch.count_nonzero(dist):
            raise ValueError('compact cache requires the rectified CPU dataset')
        dm.train_cameras=dm.train_dataset.cameras
        views=CompactViews(dataset,dm.config.cache_images_type,scene)
        observations['splits'][split]=dict(frames=len(views),raw_tensor_bytes=views.uncompressed_bytes,
            resident_tensor_bytes=views.resident_bytes,file_backed_depth_bytes=views.file_backed_bytes,
            codec='uint8 / 255 only after complete tensor equality; depth mmap only after complete equality')
        write(output,observations)
        return views
    return patch.object(manager,'_load_images',load)
