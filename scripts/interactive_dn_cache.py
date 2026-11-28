"""Lossless, on-demand, process-shared full-image cache for the fixed DN datasets."""
from concurrent.futures import ThreadPoolExecutor
import contextlib
import json
import os
from pathlib import Path

import numpy as np
import torch


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.pending")
    temp.write_text(json.dumps(value, allow_nan=False) + "\n", encoding="utf8")
    temp.replace(path)


def _save_array(path, data):
    temp = path.with_name(path.name + f".{os.getpid()}.pending")
    with temp.open("wb") as out:
        np.save(out, data, allow_pickle=False)
    temp.replace(path)


def _encode_image(value):
    """Use a byte code only after a bit-exact round trip of the complete tensor."""
    if value.ndim != 3 or value.dtype != np.float32 or not np.isfinite(value).all():
        return None
    if value.min() < 0 or value.max() > 1:
        return None
    code = np.rint(value * 255).astype(np.uint8)
    if np.array_equal(code.astype(np.float32) / np.float32(255), value):
        return code, None
    # Historical OpenGL normal conversion can round a code differently per channel.
    tables = np.zeros((code.shape[-1], 256), dtype=np.float32)
    for c, table in enumerate(tables):
        table[code[..., c]] = value[..., c]
        if not np.array_equal(table[code[..., c]], value[..., c]):
            return None
    return code, tables


class MMapViews:
    def __init__(self, dataset, dtype, directory, device=None):
        self.dataset = dataset
        self.dtype = dtype
        self.directory = directory
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dn-next-view")
        self.future = None
        self.next_id = None
        self.hits = self.reads = 0
        self.device = None if device is None else torch.device(device)
        self.copy_stream = torch.cuda.Stream(device=self.device) if self.device is not None else None
        if self.copy_stream is not None:
            with torch.cuda.stream(self.copy_stream):
                # Reuse exactly the float32 values produced by the original CPU decode.
                self.byte_values = torch.arange(256,dtype=torch.float32).div_(255).to(self.device)
        # Ordered evaluation code may access cameras before cached_eval[i]. Install
        # all corrections up front only on sources which actually have distortion.
        dist = dataset.cameras.distortion_params
        if dist is not None:
            for i in torch.nonzero((dist != 0).any(dim=-1)).flatten().tolist():
                self._record(i)

    def __len__(self):
        return len(self.dataset)

    def _produce(self, index, prefix):
        from nerfstudio.data.datamanagers.full_images_datamanager import _undistort_image
        camera = self.dataset.cameras[index].reshape(())
        data = self.dataset.get_data(index, image_type=self.dtype)
        h, w = data["image"].shape[:2]
        if (h, w) != (int(camera.height.item()), int(camera.width.item())):
            raise ValueError("decoded image and camera size differ")
        corrected = None
        if camera.distortion_params is not None and torch.any(camera.distortion_params != 0):
            K, rgb, mask = _undistort_image(camera, camera.distortion_params.numpy(), data,
                                             data["image"].numpy(), camera.get_intrinsics_matrices().numpy())
            data["image"] = torch.from_numpy(rgb)
            if mask is not None:
                data["mask"] = mask
            corrected = [float(K[0,0]), float(K[1,1]), float(K[0,2]), float(K[1,2]), rgb.shape[1], rgb.shape[0]]
        fields = {}
        for key, item in data.items():
            if not torch.is_tensor(item):
                fields[key] = {"literal": item}
                continue
            value = item.detach().cpu().numpy()
            if key == "sensor_depth":
                source = Path(self.dataset.sensor_depth_filenames[index])
                if source.suffix == ".npy":
                    raw = np.load(source, allow_pickle=False, mmap_mode="r")
                    if (raw.dtype == value.dtype and raw.size == value.size
                            and np.array_equal(raw.reshape(value.shape), value)):
                        fields[key] = {"file": str(source.resolve()), "shape": list(value.shape)}
                        continue
            codec = _encode_image(value) if key in ("image", "normal") else None
            path = prefix.with_name(prefix.name + "-" + key + ".npy")
            field = {"file": path.name}
            if codec is not None:
                value, lut = codec
                field["byte"] = True
                if lut is not None:
                    lutpath = path.with_suffix(".lut.npy")
                    _save_array(lutpath, lut)
                    field["lut"] = lutpath.name
            _save_array(path, value)
            fields[key] = field
        return {"fields": fields, "camera": corrected}

    def _record(self, index):
        import fcntl
        prefix = self.directory / f"{index:05d}"
        record = prefix.with_suffix(".json")
        if not record.exists():
            self.directory.mkdir(parents=True, exist_ok=True)
            with prefix.with_suffix(".lock").open("a+b") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                if not record.exists():
                    atomic_json(record, self._produce(index, prefix))
        row = json.loads(record.read_bytes())
        if row["camera"] is not None:
            for key, value in zip(("fx","fy","cx","cy","width","height"), row["camera"]):
                getattr(self.dataset.cameras, key)[index] = value
        return row["fields"]

    def _decode(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        result = {}
        for key, field in self._record(index).items():
            if "literal" in field:
                result[key] = field["literal"]
                continue
            # ACCESS_COPY keeps shared pages immutable when a native caller writes its tensor.
            arr = np.load(self.directory / field["file"], mmap_mode="c", allow_pickle=False)
            if "shape" in field:
                arr = arr.reshape(field["shape"])
            tensor = torch.from_numpy(arr)
            if self.device is not None:
                # This mapping is COW, and this task never writes it. PyTorch stages
                # pageable CPU memory, while the next render proceeds on another stream.
                tensor = tensor.to(self.device,non_blocking=True)
                if "lut" in field:
                    lut = torch.from_numpy(np.load(self.directory/field["lut"],allow_pickle=False))
                    lut = lut.to(self.device,non_blocking=True)
                    tensor = torch.stack([table[tensor[..., c].long()] for c,table in enumerate(lut)],dim=-1)
                elif field.get("byte"):
                    tensor = self.byte_values[tensor.long()]
            elif "lut" in field:
                lut = np.load(self.directory / field["lut"], allow_pickle=False)
                tensor = torch.from_numpy(np.stack([row[arr[..., c]] for c,row in enumerate(lut)],axis=-1))
            elif field.get("byte"):
                tensor = tensor.float().div_(255)
            result[key] = tensor
        return result

    def _read(self,index):
        if self.copy_stream is None:
            return self._decode(index),None
        with torch.cuda.stream(self.copy_stream):
            batch=self._decode(index)
            return batch,self.copy_stream.record_event()

    def __getitem__(self, index):
        self.reads += 1
        if self.future is not None and self.next_id == index:
            task, self.future = self.future, None
            self.hits += 1
            batch,event=task.result()
        else:
            batch,event=self._read(index)
        if event is not None:
            consumer=torch.cuda.current_stream(self.device)
            consumer.wait_event(event)
            for value in batch.values():
                if torch.is_tensor(value):
                    value.record_stream(consumer)
        return batch

    def prefetch(self, index):
        if self.future is None:
            self.next_id = index
            self.future = self.executor.submit(self._read, index)

    def close(self):
        self.executor.shutdown(wait=True)


@contextlib.contextmanager
def shared_views(manager_class, cache_dir, prefetch_to_gpu=False):
    from unittest.mock import patch
    opened = []

    def load(dm, split, cache_images_device):
        dataset = dm.train_dataset if split == "train" else dm.eval_dataset
        # val/test must not share a split index. Include ordered image identities.
        import hashlib
        names = "\n".join(map(str, dataset.image_filenames)).encode("utf8")
        directory = cache_dir / hashlib.sha256(names).hexdigest()
        dm.train_cameras = dm.train_dataset.cameras
        device = dm.device if prefetch_to_gpu and split=="train" and torch.device(dm.device).type=="cuda" else None
        view = MMapViews(dataset, dm.config.cache_images_type, directory,device=device)
        opened.append(view)
        return view

    with patch.object(manager_class, "_load_images", load):
        try:
            yield opened
        finally:
            for view in opened:
                view.close()


@contextlib.contextmanager
def shared_seed_geometry(model_class, directory):
    """Memoize only untrained deterministic native KNN/rotation conversions."""
    import fcntl
    import hashlib
    import inspect
    from unittest.mock import patch
    from dn_splatter import dn_model
    directory.mkdir(parents=True,exist_ok=True)
    nearest=model_class.k_nearest_sklearn
    quaternion=dn_model.matrix_to_quaternion

    def result(op, arg, names, produce):
        raw=arg.detach().cpu().contiguous().numpy()
        h=hashlib.sha256()
        h.update(inspect.getsource(op).encode("utf8"))
        h.update(str((raw.shape,raw.dtype,names)).encode())
        h.update(memoryview(raw).cast("B"))
        folder=directory/h.hexdigest()
        folder.mkdir(exist_ok=True)
        ready=folder/"ready.json"
        with (folder/"build.lock").open("a+b") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not ready.is_file():
                arrays=produce()
                for k,a in zip(names,arrays,strict=True):
                    _save_array(folder/(k+".npy"),a)
                atomic_json(ready,{"native_function":op.__qualname__,"input_shape":list(raw.shape)})
        return [np.load(folder/(k+".npy"),mmap_mode="c",allow_pickle=False) for k in names]

    def knn(model, x, k):
        if k!=3:
            return nearest(model,x,k)
        return tuple(result(nearest,x,("distances_k3","indices_k3"),lambda:nearest(model,x,k)))

    def quat(matrix):
        arrays=result(quaternion,matrix,("quats",),lambda:[quaternion(matrix).detach().cpu().numpy()])
        return torch.from_numpy(arrays[0]).to(matrix.device)

    with patch.object(model_class,"k_nearest_sklearn",knn),patch.object(dn_model,"matrix_to_quaternion",quat):
        yield
