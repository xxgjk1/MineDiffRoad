"""MiT-B0 stride-four features and the official LitePT-S topology.

MiT modules are implemented here with official checkpoint-compatible key names.
The supplied supplement explicitly chooses the stride-four stage. Later stages
are therefore not instantiated; see docs/IMPLEMENTATION_NOTES.md.
"""
import hashlib
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F


class PatchEmbed(nn.Module):
    def __init__(self, inc, outc, kernel, stride):
        super().__init__()
        self.proj = nn.Conv2d(inc,outc,kernel,stride,kernel//2)
        self.norm = nn.LayerNorm(outc,eps=1e-6)

    def forward(self,x):
        x = self.proj(x)
        h,w = x.shape[-2:]
        return self.norm(x.flatten(2).transpose(1,2)),h,w


class DWConv(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.dwconv = nn.Conv2d(channels,channels,3,1,1,groups=channels)

    def forward(self,x,h,w):
        return self.dwconv(x.transpose(1,2).reshape(len(x),-1,h,w)).flatten(2).transpose(1,2)


class MixFFN(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.fc1 = nn.Linear(channels,channels*4)
        self.dwconv = DWConv(channels*4)
        self.fc2 = nn.Linear(channels*4,channels)

    def forward(self,x,h,w):
        return self.fc2(F.gelu(self.dwconv(self.fc1(x),h,w)))


class EfficientAttention(nn.Module):
    def __init__(self,channels,heads,sr):
        super().__init__()
        self.heads, self.sr_ratio = heads,sr
        self.q = nn.Linear(channels,channels)
        self.kv = nn.Linear(channels,channels*2)
        self.proj = nn.Linear(channels,channels)
        if sr > 1:
            self.sr = nn.Conv2d(channels,channels,sr,stride=sr)
            self.norm = nn.LayerNorm(channels,eps=1e-6)

    def forward(self,x,h,w):
        b,n,c = x.shape
        q = self.q(x).reshape(b,n,self.heads,c//self.heads).transpose(1,2)
        context = x
        if self.sr_ratio > 1:
            context = self.sr(x.transpose(1,2).reshape(b,c,h,w)).flatten(2).transpose(1,2)
            context = self.norm(context)
        kv = self.kv(context).reshape(b,-1,2,self.heads,c//self.heads).permute(2,0,3,1,4)
        attended = F.scaled_dot_product_attention(q,kv[0],kv[1])
        return self.proj(attended.transpose(1,2).reshape(b,n,c))


class MixBlock(nn.Module):
    def __init__(self,channels,heads,sr,drop_path):
        super().__init__()
        from ._vendor.litept.compat import DropPath
        self.norm1 = nn.LayerNorm(channels,eps=1e-6)
        self.attn = EfficientAttention(channels,heads,sr)
        self.norm2 = nn.LayerNorm(channels,eps=1e-6)
        self.mlp = MixFFN(channels)
        self.drop_path = DropPath(drop_path)

    def forward(self,x,h,w):
        x = x + self.drop_path(self.attn(self.norm1(x),h,w))
        return x + self.drop_path(self.mlp(self.norm2(x),h,w))


class MiTB0Stride4(nn.Module):
    out_channels = 32

    def __init__(self):
        super().__init__()
        self.patch_embed1 = PatchEmbed(3,32,7,4)
        self.block1 = nn.ModuleList([MixBlock(32,1,8,p) for p in (0.,.1/7)])
        self.norm1 = nn.LayerNorm(32,eps=1e-6)
        self.register_buffer("mean",torch.tensor([.485,.456,.406]).reshape(1,3,1,1),persistent=False)
        self.register_buffer("std",torch.tensor([.229,.224,.225]).reshape(1,3,1,1),persistent=False)

    def forward(self,image):
        x,h,w = self.patch_embed1((image-self.mean)/self.std)
        for block in self.block1:
            x = block(x,h,w)
        return self.norm1(x).transpose(1,2).reshape(len(image),32,h,w)

    def load_imagenet(self, path, expected_sha256=None):
        path = Path(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if expected_sha256 and expected_sha256 != digest:
            raise ValueError("MiT checkpoint SHA-256 mismatch")
        checkpoint = torch.load(path,map_location="cpu",weights_only=True)
        state = checkpoint.get("state_dict",checkpoint.get("model",checkpoint))
        cleaned = {}
        for key,value in state.items():
            key = key.removeprefix("module.").removeprefix("backbone.")
            if key in self.state_dict():
                cleaned[key] = value
        self.load_state_dict(cleaned,strict=True)
        return {"path":str(path.resolve()),"sha256":digest,"loaded_keys":len(cleaned)}


class LitePTEncoder(nn.Module):
    out_channels = 72

    def __init__(self,voxel_size=.05):
        super().__init__()
        try:
            from ._vendor.litept.model import LitePT
        except ImportError as error:
            raise ImportError("LitePT-S requires a compatible CUDA spconv build and addict; see README") from error
        self.net = LitePT()
        self.voxel_size = voxel_size

    def forward(self,clouds):
        if not next(self.parameters()).is_cuda:
            raise RuntimeError("The real LitePT-S sparse backbone requires CUDA; CPU tests use explicit fixtures")
        unique_clouds, grids, inverse_maps, batch_ids = [],[],[],[]
        active = []
        offset = 0
        for original_id,cloud in enumerate(clouds):
            if len(cloud) == 0:
                continue
            grid = torch.floor((cloud[:,:3]-cloud[:,:3].min(0).values)/self.voxel_size).long()
            unique,inverse = torch.unique(grid,dim=0,return_inverse=True)
            first = torch.full((len(unique),),len(cloud),device=cloud.device,dtype=torch.long)
            first.scatter_reduce_(0,inverse,torch.arange(len(cloud),device=cloud.device),reduce="amin")
            unique_clouds.append(cloud[first])
            grids.append(unique)
            inverse_maps.append(inverse+offset)
            batch_ids.append(torch.full((len(unique),),len(active),device=cloud.device,dtype=torch.long))
            offset += len(unique)
            active.append(original_id)
        output = [c.new_zeros(len(c),72) for c in clouds]
        if not active:
            return output
        points = torch.cat(unique_clouds)
        # Upstream pooling shuffles serialization even in eval. Disable every
        # such flag in eval so repeated predictions use identical neighborhoods.
        for module in self.net.modules():
            if hasattr(module,"shuffle_orders"):
                module.shuffle_orders = self.training
        encoded = self.net({"coord":points[:,:3].float(),"feat":points.float(),
                            "grid_coord":torch.cat(grids),"batch":torch.cat(batch_ids)})
        for idx,inverse in zip(active,inverse_maps):
            output[idx] = encoded.feat[inverse]
        return output
