"""Strict JSONL dataset adapter: training never opens complete reference masks."""
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import torch
from torch.utils.data import Dataset
from .geometry import farthest_points, project, cell_indices


def read_manifest(path):
    path = Path(path).resolve()
    rows = []
    for line_number,line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(),1):
        if not line.strip():
            continue
        row = json.loads(line)
        for key in ("id","group","split","rgb","points","calibration"):
            if key not in row:
                raise ValueError(f"{path}:{line_number}: missing {key}")
        if row["split"] not in ("train","val","test","predict"):
            raise ValueError("split must be train/val/test/predict")
        if "scene" in row and (not isinstance(row["scene"],str) or not row["scene"].strip()):
            raise ValueError("scene must be a nonempty prespecified resampling identifier")
        if row["split"] == "train" and any(key in row for key in ("reference", "ground_truth", "evaluation_core")):
            raise ValueError(f"Training record {row['id']} contains a complete-reference field")
        if row["split"] == "train" and "core" not in row:
            raise ValueError("Training requires conservative core masks")
        for key in ("rgb","points","calibration","core","valid","reference","evaluation_core","fps_indices"):
            if key in row:
                row[key] = str((path.parent / row[key]).resolve())
                if not Path(row[key]).is_file():
                    raise FileNotFoundError(f"{row['id']}: {key}: {row[key]}")
        rows.append(row)
    if not rows:
        raise ValueError("Empty manifest")
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate frame ids in manifest")
    groups = {}
    files = {}
    for row in rows:
        group,split = row["group"],row["split"]
        if group in groups and groups[group] != split:
            raise ValueError(f"Acquisition group {group} crosses dataset splits")
        groups[group] = split
        for key in ("rgb","points"):
            source = row[key]
            if source in files and files[source] != split:
                raise ValueError(f"{key} file crosses dataset splits: {source}")
            files[source] = split
    return rows


SEGMINE_PAPER_SPLITS = {
    "train": {"frames": 524, "scenes": 17, "groups": 8},
    "val": {"frames": 75, "scenes": 3, "groups": 2},
    "test": {"frames": 150, "scenes": 6, "groups": 2},
}


def verify_paper_segmine(rows):
    """Audit the reported SegMine working-subset counts; do not alter a manifest."""
    scene_splits = {}
    group_splits = {}
    summary = {}
    for split, expected in SEGMINE_PAPER_SPLITS.items():
        selected = [row for row in rows if row["split"] == split]
        if any(not isinstance(row.get("scene"), str) or not row["scene"].strip()
               for row in selected):
            raise ValueError(f"SegMine {split} records require nonempty scene identifiers")
        for row in selected:
            scene = row["scene"]
            if scene in scene_splits and scene_splits[scene] != split:
                raise ValueError(f"SegMine scene {scene} crosses dataset splits")
            scene_splits[scene] = split
            group = row["group"]
            if group in group_splits and group_splits[group] != split:
                raise ValueError(f"SegMine group {group} crosses dataset splits")
            group_splits[group] = split
        actual = {"frames": len(selected),
                  "scenes": len({row["scene"] for row in selected}),
                  "groups": len({row["group"] for row in selected})}
        if actual != expected:
            raise ValueError(f"SegMine {split} counts {actual} do not match paper protocol {expected}")
        summary[split] = actual
    if len(rows) != sum(item["frames"] for item in SEGMINE_PAPER_SPLITS.values()):
        raise ValueError("SegMine paper manifest must contain only the 749 train/val/test frames")
    return {"dataset": "SegMine working subset", "records": len(rows), "splits": summary}


def read_binary(path,size):
    if path is None:
        return Image.fromarray(np.ones((size[1],size[0]),dtype=np.uint8)*255)
    mask = Image.open(path).convert("L")
    if mask.size != size:
        raise ValueError(f"RGB/mask size mismatch: {path}")
    values = np.unique(np.asarray(mask))
    if not set(values.tolist()).issubset({0,1,255}):
        raise ValueError(f"Expected binary 0/1 or 0/255 mask: {path}")
    return Image.fromarray((np.asarray(mask)>0).astype(np.uint8)*255)


def verify_paper_core_equality(rows):
    """Verify, rather than assume, Ye=Yc on each held-out evaluation frame."""
    checked=[]
    for row in rows:
        if row["split"] not in ("val","test"):
            continue
        missing=[key for key in ("core","evaluation_core","reference") if key not in row]
        if missing:
            raise ValueError(f"{row['id']}: paper core audit requires {', '.join(missing)}")
        with Image.open(row["rgb"]) as rgb:
            size=rgb.size
        core=np.asarray(read_binary(row["core"],size),dtype=np.uint8)
        evaluation=np.asarray(read_binary(row["evaluation_core"],size),dtype=np.uint8)
        read_binary(row["reference"],size)  # Also check the corresponding reference geometry.
        if not np.array_equal(core,evaluation):
            mismatch=int(np.count_nonzero(core != evaluation))
            raise ValueError(f"{row['id']}: core and evaluation_core differ at {mismatch} pixels")
        checked.append({"id":row["id"],"split":row["split"],
                        "width":size[0],"height":size[1],
                        "binary_core_sha256":hashlib.sha256(core.tobytes()).hexdigest()})
    if not checked:
        raise ValueError("Paper core audit requires validation or test frames")
    checked.sort(key=lambda item:(item["split"],item["id"]))
    return {"checked_frames":len(checked),"identical_core_masks":True,"frames":checked}


def verify_natural_degradation_subset(rows, frame_ids, expected_frames=40):
    """Require every natural-degradation frame to belong to the held-out test split."""
    if not isinstance(frame_ids,list) or len(frame_ids) != expected_frames or any(
            not isinstance(frame_id,str) or not frame_id for frame_id in frame_ids):
        raise ValueError(f"Natural-degradation list must contain {expected_frames} frame IDs")
    if len(set(frame_ids)) != len(frame_ids):
        raise ValueError("Natural-degradation frame IDs must be unique")
    by_id={row["id"]:row for row in rows}
    for frame_id in frame_ids:
        if frame_id not in by_id:
            raise ValueError(f"Natural-degradation frame absent from manifest: {frame_id}")
        if by_id[frame_id]["split"] != "test":
            raise ValueError(f"Natural-degradation frame is not held-out test: {frame_id}")
    selected=[by_id[frame_id] for frame_id in frame_ids]
    return {"frames":len(selected),"split":"test",
            "scenes":sorted({row.get("scene","") for row in selected}),
            "groups":sorted({row["group"] for row in selected}),
            "frame_ids_sha256":hashlib.sha256("\n".join(sorted(frame_ids)).encode()).hexdigest()}


def tensor_image(image):
    return torch.from_numpy(np.array(image,dtype=np.float32).transpose(2,0,1).copy())/255


def tensor_mask(mask):
    return torch.from_numpy((np.array(mask)>0).copy())[None]


def aligned_geometry(image,masks,K,size,rng,augment,mode="pad",return_transform=False):
    """Aspect-preserving resize, then shared crop/padding/flip and updated K."""
    if mode not in {"pad", "crop"}:
        raise ValueError("resize_mode must be pad or crop")
    ow,oh = image.size
    scale = rng.uniform(.9,1.1) if augment else 1.
    ratio = (min(size/ow,size/oh) if mode == "pad" else max(size/ow,size/oh)) * scale
    nw,nh = max(1,int(round(ow*ratio))),max(1,int(round(oh*ratio)))
    sx,sy = nw/ow,nh/oh
    H = np.array([[sx,0,(sx-1)/2],[0,sy,(sy-1)/2],[0,0,1]],dtype=np.float32)
    image = image.resize((nw,nh),Image.Resampling.BILINEAR)
    masks = {k:v.resize((nw,nh),Image.Resampling.NEAREST) for k,v in masks.items()}
    def offset(length):
        gap = abs(length-size)
        shift = int(rng.integers(0,gap+1)) if augment else gap//2
        return shift if length >= size else -shift
    left,top = offset(nw),offset(nh)
    box = (left,top,left+size,top+size)
    image = image.crop(box)
    masks = {k:v.crop(box) for k,v in masks.items()}
    H = np.array([[1,0,-left],[0,1,-top],[0,0,1]],dtype=np.float32) @ H
    if augment and rng.random() < .5:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        masks = {k:v.transpose(Image.Transpose.FLIP_LEFT_RIGHT) for k,v in masks.items()}
        H = np.array([[-1,0,size-1],[0,1,0],[0,0,1]],dtype=np.float32) @ H
    result = (image,masks,torch.from_numpy(H @ K))
    return result + (H,) if return_transform else result


def original_area_weights(original_valid, H, size):
    """Count original valid pixel centres per output cell, in bounded chunks.

    This implements nearest-neighbour sampling of a working mask back at
    original centres. Cropped-out centres count in the denominator, not weights.
    """
    valid = np.asarray(original_valid) > 0
    total = int(valid.sum())
    if total == 0:
        raise ValueError("Original image has no valid pixels")
    weights = np.zeros(size*size, dtype=np.int64)
    for start in range(0, valid.shape[0], 256):
        y,x = np.nonzero(valid[start:start+256])
        y = y + start
        u = np.floor(H[0,0]*x + H[0,1]*y + H[0,2] + .5).astype(np.int64)
        v = np.floor(H[1,0]*x + H[1,1]*y + H[1,2] + .5).astype(np.int64)
        inside = (u >= 0) & (u < size) & (v >= 0) & (v < size)
        weights += np.bincount(v[inside]*size + u[inside], minlength=size*size)
    return torch.from_numpy(weights.reshape(1,size,size).astype(np.float32)), torch.tensor(total)


def weak_view(image,points,rng,noise_scale):
    image = ImageEnhance.Brightness(image).enhance(rng.uniform(.8,1.2))
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(.8,1.2))
    image = ImageEnhance.Color(image).enhance(rng.uniform(.8,1.2))
    hsv = np.array(image.convert("HSV"))
    hsv[:,:,0] = (hsv[:,:,0].astype(np.int16)+int(rng.uniform(-.05,.05)*255)) % 256
    image = tensor_image(Image.fromarray(hsv,"HSV").convert("RGB"))
    image = (image+torch.from_numpy(rng.normal(0,rng.uniform(0,.02),image.shape).astype(np.float32))).clamp(0,1)
    keep = rng.random(len(points)) >= rng.uniform(0,.05)
    cloud = points[torch.from_numpy(keep)].clone()
    noise = rng.normal(0,rng.uniform(0,.002)*noise_scale,(len(cloud),3)).astype(np.float32)
    cloud[:,:3] += torch.from_numpy(noise)
    return {"image":image,"points":cloud}


def degrade(image,points,K,T,omega,rng,noise_scale,tolerance):
    size = (image.height,image.width)
    grid = (size[0]//4,size[1]//4)
    ids,pixels = project(points,K,T,size,tolerance)
    cells = cell_indices(pixels,size,grid)
    counts = torch.bincount(cells,minlength=grid[0]*grid[1]).reshape(1,*grid).float()
    state = rng.choice(4,p=[.4,.25,.25,.1])
    occlusion = np.zeros(size,dtype=np.float32)
    blur_sigma = 0.
    if state in (1,3):
        kind = rng.choice(3,p=[.5,.3,.2])
        if kind in (0,2):
            blur_sigma = rng.uniform(.5,3.)
            image = image.filter(ImageFilter.GaussianBlur(blur_sigma))
        if kind in (1,2):
            fraction = rng.uniform(.05,.15)
            aspect = rng.uniform(.5,2.)
            height = min(size[0],max(1,int(np.sqrt(fraction*size[0]*size[1]/aspect))))
            width = min(size[1],max(1,int(height*aspect)))
            top,left = int(rng.integers(size[0]-height+1)),int(rng.integers(size[1]-width+1))
            image.paste((0,0,0),(left,top,left+width,top+height))
            occlusion[top:top+height,left:left+width] = 1.
    keep = torch.ones(len(points),dtype=torch.bool)
    sigma = 0.
    if state in (2,3):
        kind = rng.choice(3,p=[.5,.3,.2])
        if kind in (0,2):
            keep = torch.from_numpy(rng.random(len(points)) >= rng.uniform(.1,.4))
        if kind in (1,2):
            sigma = rng.uniform(.005,.025)
    kept_counts = torch.bincount(cells[keep[ids]],minlength=grid[0]*grid[1]).reshape(1,*grid).float()
    fraction_dropped = (1-kept_counts/counts.clamp_min(1)).clamp(0,1)
    cloud = points[keep].clone()
    cloud[:,:3] += torch.from_numpy(rng.normal(0,sigma*noise_scale,(len(cloud),3)).astype(np.float32))
    occ_grid = torch.nn.functional.adaptive_max_pool2d(torch.from_numpy(occlusion)[None],grid)
    qv = 1-torch.maximum(occ_grid,torch.full_like(occ_grid,min(1,blur_sigma**2/25)))
    qa = 1-torch.maximum(fraction_dropped,torch.full_like(fraction_dropped,min(1,sigma**2/.05**2)))
    valid_v = torch.nn.functional.interpolate(omega[None].float(),size=grid,mode="nearest")[0].bool()
    return {"image":tensor_image(image),"points":cloud}, {
        "q_v":qv,"q_a":qa,"valid_v":valid_v,"valid_a":(counts>0)&valid_v}


class RoadDataset(Dataset):
    def __init__(self,manifest,split,config):
        self.rows = [r for r in read_manifest(manifest) if r["split"] == split]
        if not self.rows:
            raise ValueError(f"No {split} records")
        self.split,self.config,self.epoch = split,config,0

    def __len__(self):
        return len(self.rows)

    def __getitem__(self,index):
        row,c = self.rows[index],self.config
        seed_key = f"{c['seed']}:{self.epoch}:{row['id']}".encode()
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256(seed_key).digest()[:8],"little"))
        image = Image.open(row["rgb"]).convert("RGB")
        masks = {"omega":read_binary(row.get("valid"),image.size)}
        if self.split == "train":
            masks["core"] = read_binary(row["core"],image.size)
        elif "reference" in row:
            masks["reference"] = read_binary(row["reference"],image.size)
            if "evaluation_core" in row:
                masks["evaluation_core"] = read_binary(row["evaluation_core"],image.size)
        points = np.load(row["points"],allow_pickle=False)
        if points.ndim != 2 or points.shape[1] != 4 or not np.isfinite(points).all():
            raise ValueError(f"{row['id']}: points must be finite N x 4 [x,y,z,intensity]")
        points = torch.from_numpy(points.astype(np.float32))
        if ((points[:,3] < 0) | (points[:,3] > 1)).any():
            raise ValueError("Normalize LiDAR intensity to [0,1] in dataset preparation")
        if "fps_indices" in row:
            if "points_sha256" in row and hashlib.sha256(Path(row["points"]).read_bytes()).hexdigest() != row["points_sha256"]:
                raise ValueError(f"{row['id']}: cached FPS refers to a different point file")
            ids = torch.from_numpy(np.load(row["fps_indices"],allow_pickle=False)).long()
            expected = min(len(points),c["num_points"])
            if ids.ndim != 1 or len(ids) != expected or len(ids.unique()) != len(ids) or (len(ids) and (ids.min()<0 or ids.max()>=len(points))):
                raise ValueError("Invalid cached FPS indices")
        else:
            ids = farthest_points(points[:,:3],c["num_points"])
        points = points[ids]
        calib = json.loads(Path(row["calibration"]).read_text())
        K,T = np.asarray(calib["K"],np.float32),torch.tensor(calib["T_lidar_to_camera"],dtype=torch.float32)
        if K.shape != (3,3) or T.shape != (4,4) or not np.isfinite(K).all() or not torch.isfinite(T).all():
            raise ValueError("Calibration must contain finite 3x3 K and 4x4 T_lidar_to_camera")
        original_valid = masks["omega"]
        image,masks,K,H = aligned_geometry(image,masks,K,c["image_size"],rng,self.split=="train",
                                           c.get("resize_mode", "pad"),return_transform=True)
        result = {"id":row["id"],"scene":row.get("scene", ""),"group":row["group"],
                  "K":K,"T":T,**{k:tensor_mask(v) for k,v in masks.items()}}
        if not result["omega"].any():
            raise ValueError(f"{row['id']}: transformed image has no valid pixels")
        if self.split == "train":
            result["area_weights"], result["original_valid_area"] = original_area_weights(original_valid,H,c["image_size"])
            result["core"] &= result["omega"]
            result["weak1"] = weak_view(image.copy(),points,rng,c["coordinate_noise_scale_m"])
            result["weak2"] = weak_view(image.copy(),points,rng,c["coordinate_noise_scale_m"])
            result["student"],result["quality"] = degrade(image.copy(),points,K,T,result["omega"],rng,
                  c["coordinate_noise_scale_m"],c["depth_tolerance_m"])
        else:
            result["student"] = {"image":tensor_image(image),"points":points}
        return result


def collate(samples):
    if isinstance(samples[0],dict):
        if any(s.keys() != samples[0].keys() for s in samples):
            raise ValueError("All records in a batch must supply the same fields, including reference/evaluation_core")
        return {k:([s[k] for s in samples] if k=="points" else collate([s[k] for s in samples])) for k in samples[0]}
    if isinstance(samples[0],torch.Tensor):
        return torch.stack(samples)
    return samples


def to_device(data,device):
    if isinstance(data,torch.Tensor):
        return data.to(device,non_blocking=True)
    if isinstance(data,dict):
        return {k:to_device(v,device) for k,v in data.items()}
    if isinstance(data,list):
        return [to_device(v,device) for v in data]
    return data
