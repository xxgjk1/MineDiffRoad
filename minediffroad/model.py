import copy
import math
import torch
from torch import nn
import torch.nn.functional as F
from .backbones import MiTB0Stride4, LitePTEncoder
from .geometry import aggregate, resize


def conv(inc,outc,kernel=1,groups=1,bias=False):
    return nn.Conv2d(inc,outc,kernel,padding=kernel//2,groups=groups,bias=bias)


def quality_head():
    return nn.Sequential(conv(257,64),nn.GroupNorm(8,64),nn.SiLU(),conv(64,64,3,64),
                         conv(64,32),nn.SiLU(),conv(32,1,bias=True))


class SegmentationPath(nn.Module):
    def __init__(self,config,image_encoder=None,point_encoder=None):
        super().__init__()
        self.image_encoder = image_encoder if image_encoder is not None else MiTB0Stride4()
        self.point_encoder = point_encoder if point_encoder is not None else LitePTEncoder(config["voxel_size"])
        self.image_projection = conv(self.image_encoder.out_channels,128)
        self.point_projection = conv(self.point_encoder.out_channels*2,128)
        self.quality_v, self.quality_a = quality_head(),quality_head()
        self.fusion = nn.Sequential(conv(257,128),nn.GroupNorm(16,128),nn.SiLU(),
                                    conv(128,128,3,128),conv(128,128),nn.GroupNorm(16,128),nn.SiLU())
        self.base_features = nn.Sequential(conv(128,64,3),nn.GroupNorm(8,64),nn.SiLU())
        self.base_logits = conv(64,2,bias=True)
        self.residual = nn.Sequential(conv(197,64,3),nn.GroupNorm(8,64),nn.SiLU(),
                                      conv(64,32,3),nn.GroupNorm(8,32),nn.SiLU(),conv(32,1,bias=True))
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)
        self.alpha = config["residual_bound"]
        self.tolerance = config["depth_tolerance_m"]

    def forward(self,image,clouds,K,T):
        size = image.shape[-2:]
        zv = self.image_projection(self.image_encoder(image))
        grid = zv.shape[-2:]
        # Sparse convolutions use float32; RGB/task layers can use autocast.
        with torch.autocast(device_type=image.device.type,enabled=False):
            point_features = self.point_encoder([c.float() for c in clouds])
            aligned = [aggregate(f,p,k,t,size,grid,self.tolerance)
                       for f,p,k,t in zip(point_features,clouds,K,T)]
        za = self.point_projection(torch.cat([a[0] for a in aligned]))
        va = torch.cat([a[1] for a in aligned]).to(zv.dtype)
        joint = torch.cat([zv,za,va],1)
        lv,la = self.quality_v(joint),self.quality_a(joint)
        rv,ra = lv.sigmoid(),va*la.sigmoid()
        weights = (rv+1e-6)/(rv+ra+2e-6)
        fused = self.fusion(torch.cat([weights*zv,(1-weights)*za,va],1))
        base_feat = self.base_features(fused)
        base_logits = self.base_logits(base_feat)
        logits_full = resize(base_logits,size)
        sb = logits_full[:,1:2]-logits_full[:,0:1]
        pb = sb.sigmoid()
        context = torch.cat([fused,base_feat,base_logits,rv,ra,va],1)
        correction = self.alpha*(1-(2*pb-1).abs())*resize(self.residual(context),size).tanh()
        sf = sb+correction
        va_full = resize(va,size,True)
        return {"logit":sf,"prob":sf.sigmoid(),"base_prob":pb,"delta":correction,
                "rv":resize(rv,size),"ra":resize(la.sigmoid(),size)*va_full,
                "quality_logit_v":lv,"quality_logit_a":la,"point_support":va,
                "point_features":za,"base_features":base_feat,"base_logits":base_logits}


class ResidualBlock(nn.Module):
    def __init__(self,time_condition=False):
        super().__init__()
        self.norm1,self.norm2 = nn.GroupNorm(16,128),nn.GroupNorm(16,128)
        self.c1,self.c2 = conv(128,128,3),conv(128,128,3)
        self.film = nn.Linear(128,256) if time_condition else None

    def forward(self,x,time=None):
        h = self.norm1(x)
        if self.film is not None:
            scale,shift = self.film(time).chunk(2,dim=1)
            h = h*(1+scale[:,:,None,None])+shift[:,:,None,None]
        h = self.c1(F.silu(h))
        return x+self.c2(F.silu(self.norm2(h)))


class ReconstructionBranch(nn.Module):
    def __init__(self,steps=1000,offset=.008,min_alpha=1e-6):
        super().__init__()
        self.steps = steps
        self.condition_input = conv(199,128)
        self.condition_blocks = nn.ModuleList([ResidualBlock() for _ in range(3)])
        self.noise_input = conv(129,128,3)
        self.noise_blocks = nn.ModuleList([ResidualBlock(True) for _ in range(3)])
        self.noise_output = conv(128,1,bias=True)
        self.time_mlp = nn.Sequential(nn.Linear(128,256),nn.SiLU(),nn.Linear(256,128))
        times = torch.arange(steps+1,dtype=torch.float64)/steps
        f = torch.cos((times+offset)/(1+offset)*math.pi/2).square()
        self.register_buffer("alpha_bar",(f/f[0]).clamp_min(min_alpha).float())

    def time_code(self,t):
        freq = torch.exp(-math.log(10000)*torch.arange(64,device=t.device)/63)
        angle = t.float()[:,None]/self.steps*freq[None]
        return self.time_mlp(torch.cat([angle.sin(),angle.cos()],1))

    def forward(self,image,output,states,t=None,noise=None):
        x0 = states.target.detach().float()
        if t is None:
            t = torch.randint(1,self.steps+1,(1,),device=x0.device).expand(len(x0))
        if t.shape != (len(x0),) or ((t < 1) | (t > self.steps)).any():
            raise ValueError("t must have shape [batch] and lie in [1,T]")
        noise = torch.randn_like(x0) if noise is None else noise
        a = self.alpha_bar[t][:,None,None,None]
        xt = a.sqrt()*x0+(1-a).sqrt()*noise
        grid = output["point_features"].shape[-2:]
        condition = torch.cat([resize(image,grid),output["point_features"],
                               output["base_features"],output["base_logits"],
                               output["point_support"],resize(states.valid,grid,True)],1)
        condition = self.condition_input(condition)
        for block in self.condition_blocks:
            condition = block(condition)
        h = self.noise_input(torch.cat([resize(xt,grid),condition],1))
        time = self.time_code(t)
        for block in self.noise_blocks:
            h = block(h,time)
        noise_pred = resize(self.noise_output(h),x0.shape[-2:]).float()
        recovered = ((xt-(1-a).sqrt()*noise_pred)/a.sqrt()).clamp(0,1)
        return {"x0":recovered,"noise_pred":noise_pred,"noise":noise,"t":t}


class MineDiffRoad(nn.Module):
    def __init__(self,config,image_encoder=None,point_encoder=None):
        super().__init__()
        self.segmentation = SegmentationPath(config,image_encoder,point_encoder)
        self.reconstruction = ReconstructionBranch(config["diffusion_steps"],
                                                    config["cosine_offset"],config["min_alpha"])

    def forward(self,*args,**kwargs):
        return self.segmentation(*args,**kwargs)

    def deployment_model(self):
        return copy.deepcopy(self.segmentation).eval()


class EMATeacher:
    def __init__(self,student,decay=.99):
        self.model = copy.deepcopy(student.segmentation).eval().requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self,student):
        for dst,src in zip(self.model.parameters(),student.segmentation.parameters()):
            dst.lerp_(src.detach(),1-self.decay)
        for dst,src in zip(self.model.buffers(),student.segmentation.buffers()):
            dst.copy_(src)
        self.model.eval()
