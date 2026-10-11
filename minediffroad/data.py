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
    for line_number,line in enumerate(path.read_text().splitlines(),1):
        if not line.strip():
            continue
        row = json.loads(line)
        for key in ("id","group","split","rgb","points","calibration"):
            if key not in row:
                raise ValueError(f"{path}:{line_number}: missing {key}")
        if row["split"] not in ("train","val","test","predict"):
            raise ValueError("split must be train/val/test/predict")
        if row["split"] == "train" and ("reference" in row or "ground_truth" in row):
            raise ValueError(f"Training record {row['id']} contains a complete-reference field")
        if row["split"] == "train" and "core" not in row:
            raise ValueError("Training requires conservative core masks")
        for key in ("rgb","points","calibration","core","valid","reference","fps_indices"):
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


def tensor_image(image):
    return torch.from_numpy(np.array(image,dtype=np.float32).transpose(2,0,1).copy())/255


def tensor_mask(mask):
    return torch.from_numpy((np.array(mask)>0).copy())[None]


def aligned_geometry(image,masks,K,size,rng,augment):
    """One homography for RGB, every mask and projected LiDAR coordinates."""
    ow,oh = image.size
    scale = rng.uniform(.9,1.1) if augment else 1.
    side = max(32,int(round(size*scale)))
    sx,sy = side/ow,side/oh
    H = np.array([[sx,0,(sx-1)/2],[0,sy,(sy-1)/2],[0,0,1]],dtype=np.float32)
    image = image.resize((side,side),Image.Resampling.BILINEAR)
    masks = {k:v.resize((side,side),Image.Resampling.NEAREST) for k,v in masks.items()}
    if side >= size:
        left = int(rng.integers(0,side-size+1)) if augment else (side-size)//2
        top = int(rng.integers(0,side-size+1)) if augment else (side-size)//2
    else:
        left = -int(rng.integers(0,size-side+1)) if augment else -(size-side)//2
        top = -int(rng.integers(0,size-side+1)) if augment else -(size-side)//2
    box = (left,top,left+size,top+size)
    image = image.crop(box)
    masks = {k:v.crop(box) for k,v in masks.items()}
    H = np.array([[1,0,-left],[0,1,-top],[0,0,1]],dtype=np.float32) @ H
    if augment and rng.random() < .5:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        masks = {k:v.transpose(Image.Transpose.FLIP_LEFT_RIGHT) for k,v in masks.items()}
        H = np.array([[-1,0,size-1],[0,1,0],[0,0,1]],dtype=np.float32) @ H
    return image,masks,torch.from_numpy(H @ K)


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
        image,masks,K = aligned_geometry(image,masks,K,c["image_size"],rng,self.split=="train")
        result = {"id":row["id"],"K":K,"T":T,**{k:tensor_mask(v) for k,v in masks.items()}}
        if self.split == "train":
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
