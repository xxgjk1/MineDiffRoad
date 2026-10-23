"""Software checks on synthetic tensors."""
import json
import os
import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch import nn
from minediffroad.geometry import project,aggregate,farthest_points,resize
from minediffroad.supervision import build_states,fixed_quota_selection
from minediffroad.losses import balanced_segmentation,structural_losses,total_loss
from minediffroad.model import MineDiffRoad,EMATeacher
from minediffroad.backbones import MiTB0
from minediffroad.data import read_manifest,RoadDataset,collate,aligned_geometry
from minediffroad.engine import load_config,train_step,rng_state,restore_rng,seed_everything
from minediffroad.metrics import from_confusion,postprocess
from minediffroad._vendor.litept.compat import segment_csr,flash_attn_varlen_qkvpacked_func

ROOT=Path(__file__).resolve().parents[1]
torch.set_num_threads(2)


class FixtureImage(nn.Module):
    out_channels=8
    def __init__(self):
        super().__init__(); self.conv=nn.Conv2d(3,8,4,4)
    def forward(self,x):
        return self.conv(x)


class FixturePoints(nn.Module):
    out_channels=8
    def __init__(self):
        super().__init__(); self.linear=nn.Linear(4,8)
    def forward(self,clouds):
        return [self.linear(c) for c in clouds]


def config():
    c=load_config(ROOT/"configs/example.yaml")
    c.update(image_size=64,num_points=256,batch_size=2,workers=0,device="cpu",skeleton_iterations=2)
    return c


def inputs(device="cpu",batch=2):
    image=torch.rand(batch,3,64,64,device=device)
    clouds=[torch.cat([torch.rand(512,2,device=device)*10-5,
                       torch.rand(512,1,device=device)*6+8,
                       torch.rand(512,1,device=device)],1) for _ in range(batch)]
    K=torch.tensor([[50.,0,32],[0,50,32],[0,0,1]],device=device).repeat(batch,1,1)
    T=torch.eye(4,device=device).repeat(batch,1,1)
    return image,clouds,K,T


class CoreTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)

    def test_ring_widths_and_states(self):
        core=torch.zeros(1,1,65,65,dtype=torch.bool); core[:,:,32,32]=1
        ones=torch.ones_like(core,dtype=torch.float)
        s=build_states(core,ones,ones,ones,warmup=True)
        self.assertEqual(int(s.ring.sum()),25**2-1)
        self.assertFalse((s.negative & s.positive).any())
        self.assertTrue(torch.equal(s.unknown,~(s.positive|s.negative)))
        self.assertTrue(s.negative[:,:,:10].all())
        self.assertFalse(s.unknown.any())

    def test_negative_and_positive_reliability(self):
        core=torch.zeros(1,1,64,64,dtype=torch.bool); core[:,:,28:36,28:36]=1
        one=torch.ones_like(core,dtype=torch.float); zero=one*0
        s=build_states(core,one,one,zero,zero,zero)
        self.assertFalse(s.admitted.any())
        self.assertTrue(s.negative[s.exterior].all())
        s=build_states(core,one,one,zero,one*.96,one*.94)
        self.assertTrue(s.candidates.any())
        self.assertFalse((s.positive&s.candidates).any())
        self.assertTrue(torch.allclose(s.target[s.candidates],torch.full_like(s.target[s.candidates],.95)))

    def test_detachment_and_strict_thresholds(self):
        core=torch.zeros(1,1,40,40,dtype=torch.bool); core[:,:,20,20]=1
        one=torch.ones_like(core,dtype=torch.float,requires_grad=True)
        s=build_states(core,one,one*.5,one,one*.01,one*.01)
        self.assertFalse(s.admitted.any()); self.assertFalse(s.target.requires_grad)

    def test_empty_partial_loss_and_gradients(self):
        z=torch.randn(1,1,8,8,requires_grad=True); y=torch.ones_like(z)
        valid=torch.zeros_like(z,dtype=torch.bool)
        self.assertEqual(float(balanced_segmentation(z,y,valid).detach()),0.)
        valid[:,:,3:5,3:5]=1
        loss=balanced_segmentation(z,y,valid); loss.backward()
        self.assertEqual(float(z.grad[~valid].abs().sum()),0.)
        self.assertGreater(float(z.grad[valid].abs().sum()),0.)

    def test_unknown_fill_cannot_change_structure(self):
        valid=torch.zeros(1,1,48,48,dtype=torch.bool); valid[:,:,8:40,8:40]=1
        y=torch.rand(1,1,48,48); p=torch.rand_like(y).requires_grad_()
        changed=y.clone(); changed[~valid]=100
        a=structural_losses(p,y,valid,2)
        b=structural_losses(p,changed,valid,2)
        for x,z in zip(a,b):
            self.assertTrue(torch.allclose(x,z))
        sum(a).backward()
        self.assertEqual(float(p.grad[~valid].abs().sum()),0.)

    def test_projection_and_depth_buffer(self):
        p=torch.tensor([[0.,0,2,1],[0,0,2.4,1],[0,0,3.,1],[0,0,-2,1],[100,0,2,1]])
        K=torch.tensor([[1.,0,2],[0,1,2],[0,0,1]])
        ids,pixels=project(p,K,torch.eye(4),(5,5),.5)
        self.assertEqual(ids.tolist(),[0,1]); self.assertEqual(pixels.tolist(),[12,12])
        f=torch.tensor([[1.],[3.],[20.],[20.],[20.]],requires_grad=True)
        features,support,_=aggregate(f,p,K,torch.eye(4),(5,5),(1,1),.5)
        self.assertEqual(features.flatten().tolist(),[2.,3.]); self.assertTrue(support.all())
        features.sum().backward(); self.assertEqual(float(f.grad[2:].abs().sum()),0.)

    def test_fps_no_duplicates_or_padding(self):
        p=torch.rand(30,3)
        a=farthest_points(p,12); b=farthest_points(p,12)
        self.assertTrue(torch.equal(a,b)); self.assertEqual(len(a.unique()),12)
        self.assertEqual(len(farthest_points(p,40)),30)

    def test_fixed_quota_ties(self):
        domain=torch.ones(2,1,3,3,dtype=torch.bool)
        chosen=fixed_quota_selection(domain,torch.ones_like(domain,dtype=torch.float),.6)
        self.assertEqual(chosen[0].flatten().tolist(),[True]*5+[False]*4)

    def test_operator_adapters(self):
        x=torch.randn(7,4,requires_grad=True); ptr=torch.tensor([0,2,7])
        y=segment_csr(x,ptr,"mean")
        self.assertTrue(torch.allclose(y,torch.stack([x[:2].mean(0),x[2:].mean(0)])))
        qkv=torch.randn(7,3,2,6,requires_grad=True)
        out=flash_attn_varlen_qkvpacked_func(qkv,ptr.int(),5,softmax_scale=6**-.5)
        q,k,v=qkv[:2].unbind(1)
        manual=((q.transpose(0,1) @ k.transpose(0,1).transpose(-1,-2))*6**-.5).softmax(-1) @ v.transpose(0,1)
        self.assertTrue(torch.allclose(out[:2],manual.transpose(0,1),atol=1e-6))
        (out.square().mean()+y.square().mean()).backward()
        self.assertTrue(torch.isfinite(qkv.grad).all())

    def test_mit_stage_shape_and_weight_loading(self):
        model=MiTB0()
        x=model(torch.rand(2,3,64,64))
        self.assertEqual(x.shape,(2,512,16,16))
        x.mean().backward()
        for i in range(1,5):
            self.assertGreater(float(getattr(model,f"patch_embed{i}").proj.weight.grad.abs().sum()),0.)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"weights.pt"; torch.save(model.state_dict(),path)
            metadata=MiTB0().load_imagenet(path)
            self.assertEqual(len(metadata["sha256"]),64)
            torch.save({k:v for k,v in model.state_dict().items() if k.startswith(("patch_embed1.","block1.","norm1."))},path)
            with self.assertRaisesRegex(ValueError,"Full four-stage"):
                MiTB0().load_imagenet(path)

    def test_residual_bound_and_missing_point_support(self):
        model=MineDiffRoad(config(),FixtureImage(),FixturePoints()).eval()
        image,clouds,K,T=inputs()
        clouds=[c[:0] for c in clouds]
        output=model(image,clouds,K,T)
        self.assertFalse(output["ra"].any())
        self.assertFalse(output["delta"].any())
        with torch.no_grad():
            model.segmentation.residual[-1].bias.fill_(100)
        output=model(image,clouds,K,T)
        self.assertLessEqual(float(output["delta"].detach().abs().max()),.100001)
        self.assertLessEqual(float((output["prob"]-output["base_prob"]).detach().abs().max()),.025001)

    def test_reconstruction_gradient_and_endpoint(self):
        model=MineDiffRoad(config(),FixtureImage(),FixturePoints())
        image,clouds,K,T=inputs()
        output=model(image,clouds,K,T)
        core=torch.zeros(2,1,64,64,dtype=torch.bool); core[:,:,20:35,20:35]=1
        one=torch.ones_like(core,dtype=torch.float)
        states=build_states(core,one,one,one,one*.98,one*.96)
        reconstructed=model.reconstruction(image,output,states,t=torch.tensor([1000,1000]))
        self.assertTrue(torch.isfinite(reconstructed["x0"]).all())
        # Transfer supervision does not update the reconstruction head.
        transfer=balanced_segmentation(output["logit"],reconstructed["x0"],states.valid)
        transfer.backward(retain_graph=True)
        self.assertIsNone(model.reconstruction.noise_output.weight.grad)
        model.zero_grad(set_to_none=True)
        (reconstructed["noise_pred"]-reconstructed["noise"]).square().mean().backward()
        self.assertGreater(float(model.reconstruction.noise_output.weight.grad.abs().sum()),0.)
        self.assertGreater(float(model.segmentation.base_features[0].weight.grad.abs().sum()),0.)

    def test_ema_excludes_reconstruction_and_updates(self):
        model=MineDiffRoad(config(),FixtureImage(),FixturePoints())
        ema=EMATeacher(model,.5)
        self.assertFalse(any("reconstruction" in k for k in ema.model.state_dict()))
        before=next(ema.model.parameters()).clone()
        with torch.no_grad():
            next(model.segmentation.parameters()).add_(2)
        ema.update(model)
        self.assertTrue(torch.allclose(next(ema.model.parameters()),before+1))

    def test_metric_empty_cases(self):
        self.assertEqual(from_confusion([0,0,0,20])["road_iou"],1.)
        self.assertEqual(from_confusion([0,0,5,20])["precision"],0.)
        p=np.zeros((10,10)); p[1:3,1:3]=1
        self.assertFalse(postprocess(p,.5,5).any())

    def test_rng_roundtrip_safe_serialization(self):
        state=rng_state()
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"rng.pt"; torch.save(state,path)
            state=torch.load(path,weights_only=True)
        a=torch.rand(10); restore_rng(state); b=torch.rand(10)
        self.assertTrue(torch.equal(a,b))

    def test_dataset_and_training_reference_rejection(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            Image.fromarray(np.zeros((64,64,3),np.uint8)).save(root/"rgb.png")
            core=np.zeros((64,64),np.uint8); core[20:40,20:40]=255
            Image.fromarray(core).save(root/"core.png")
            np.save(root/"points.npy",np.array([[0,0,10,.5],[1,1,12,.2]],np.float32))
            (root/"calib.json").write_text(json.dumps({"K":[[50,0,32],[0,50,32],[0,0,1]],"T_lidar_to_camera":np.eye(4).tolist()}))
            row={"id":"frame","group":"scene","split":"train","rgb":"rgb.png","points":"points.npy","calibration":"calib.json","core":"core.png"}
            manifest=root/"data.jsonl"; manifest.write_text(json.dumps(row)+"\n")
            dataset=RoadDataset(manifest,"train",config()); dataset.epoch=1
            a,b=dataset[0],dataset[0]
            self.assertTrue(torch.equal(a["student"]["image"],b["student"]["image"]))
            batch=collate([a,b])
            model=MineDiffRoad(config(),FixtureImage(),FixturePoints())
            loss,terms,counts=train_step(model,batch,1,config())
            loss.backward(); self.assertTrue(torch.isfinite(loss))
            self.assertEqual(float(terms["rec"].detach()),0.)
            row["reference"]="core.png"; manifest.write_text(json.dumps(row))
            with self.assertRaises(ValueError):
                read_manifest(manifest)

    def test_group_split_leakage_rejection(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); (root/"dummy").touch()
            rows=[{"id":str(i),"group":"same_scene","split":s,"rgb":"dummy","points":"dummy","calibration":"dummy","core":"dummy"} for i,s in enumerate(["train","test"])]
            manifest=root/"data.jsonl"; manifest.write_text("\n".join(json.dumps(r) for r in rows))
            with self.assertRaisesRegex(ValueError,"crosses dataset splits"):
                read_manifest(manifest)

    def test_geometric_augmentation_preserves_projection(self):
        image=Image.fromarray(np.zeros((96,128,3),np.uint8))
        K=np.array([[80,0,64],[0,80,48],[0,0,1]],np.float32)
        mask=np.zeros((96,128),np.uint8); mask[44:53,60:69]=255
        for seed in range(8):
            _,m,k=aligned_geometry(image,{"core":Image.fromarray(mask)},K,64,np.random.default_rng(seed),True)
            _,pixel=project(torch.tensor([[0.,0.,10.,1.]]),k,torch.eye(4),(64,64))
            self.assertEqual(len(pixel),1)
            self.assertTrue(np.array(m["core"]).reshape(-1)[pixel.item()] > 0)


@unittest.skipUnless(os.environ.get("MINEDIFFROAD_TEST_CUDA")=="1" and torch.cuda.is_available(),"opt-in real CUDA backbone test")
class RealBackboneTest(unittest.TestCase):
    def test_real_litept_forward_backward_and_deterministic_eval(self):
        seed_everything(17)
        c=config(); c["device"]="cuda"
        model=MineDiffRoad(c).cuda()
        args=inputs("cuda")
        output=model(*args)
        core=torch.zeros(2,1,64,64,dtype=torch.bool,device="cuda"); core[:,:,20:40,20:40]=True
        one=torch.ones_like(core,dtype=torch.float)
        states=build_states(core,one,one,one,one*.98,one*.96)
        rec=model.reconstruction(args[0],output,states,t=torch.tensor([300,300],device="cuda"))
        quality={"q_v":torch.ones_like(output["quality_logit_v"]),"q_a":torch.ones_like(output["quality_logit_a"]),
                 "valid_v":torch.ones_like(output["quality_logit_v"],dtype=torch.bool),"valid_a":output["point_support"].bool()}
        loss,_,=total_loss(output,states,quality,rec,c["loss_weights"],2)
        loss.backward()
        grads=[p.grad for p in model.segmentation.point_encoder.parameters() if p.grad is not None]
        self.assertTrue(grads); self.assertTrue(all(torch.isfinite(g).all() for g in grads))
        model.eval()
        with torch.no_grad():
            a=model(*args)["prob"]; b=model(*args)["prob"]
        self.assertTrue(torch.allclose(a,b,atol=1e-6,rtol=1e-5),f"max absolute difference: {(a-b).abs().max().item()}")

    def test_real_backbone_amp_optimizer_step(self):
        seed_everything(17)
        c=config(); c["device"]="cuda"
        model=MineDiffRoad(c).cuda().train()
        optimizer=torch.optim.AdamW(model.parameters(),lr=c['learning_rate'],weight_decay=c['weight_decay'])
        args=inputs("cuda")
        core=torch.zeros(2,1,64,64,dtype=torch.bool,device="cuda"); core[:,:,20:40,20:40]=1
        one=torch.ones_like(core,dtype=torch.float)
        with torch.autocast("cuda",dtype=torch.float16):
            out=model(*args)
            states=build_states(core,one,one,one,one*.98,one*.96)
            rec=model.reconstruction(args[0],out,states,t=torch.tensor([400,400],device="cuda"))
            quality={"q_v":torch.ones_like(out["quality_logit_v"],dtype=torch.float),
                     "q_a":torch.ones_like(out["quality_logit_a"],dtype=torch.float),
                     "valid_v":torch.ones_like(out["quality_logit_v"],dtype=torch.bool),"valid_a":out["point_support"].bool()}
            loss,_=total_loss(out,states,quality,rec,c["loss_weights"],2)
        scaler=torch.amp.GradScaler("cuda",init_scale=128.)
        scaler.scale(loss).backward(); scaler.unscale_(optimizer)
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        old=model.segmentation.base_logits.weight.detach().clone()
        scaler.step(optimizer); scaler.update()
        self.assertFalse(torch.equal(old,model.segmentation.base_logits.weight))


if __name__ == "__main__":
    unittest.main()
