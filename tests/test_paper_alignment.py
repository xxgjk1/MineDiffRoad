"""Deterministic model, supervision, geometry, and statistics checks."""
import copy
import tempfile
import unittest
import numpy as np
import torch
from PIL import Image
from minediffroad.backbones import MiTB0
from minediffroad.data import (aligned_geometry,original_area_weights,
                               verify_paper_core_equality,verify_natural_degradation_subset)
from minediffroad.metrics import recovery_metrics, selection_diagnostics, postprocess
from minediffroad.statistics import paired_iou_intervals
from minediffroad.supervision import build_states, cap_unknown, dilate
from minediffroad.engine import load_config, train
from minediffroad.engine import train_step, evaluate_loader
from minediffroad.model import EMATeacher
from tests.test_core import ROOT, config, FixtureImage, FixturePoints, inputs
from minediffroad.model import MineDiffRoad


class PaperAlignmentTests(unittest.TestCase):
    def test_paper_resolution_and_full_encoder(self):
        c=load_config(ROOT/'configs/paper.yaml')
        self.assertEqual(c['image_size'],1024)
        self.assertEqual(c['batch_size'],2)
        self.assertEqual(c['learning_rate'],1e-3)
        self.assertEqual(c['weight_decay'],1e-4)
        self.assertEqual(c['learning_rate_schedule'],'constant')
        self.assertEqual(load_config(ROOT/'configs/example.yaml')['learning_rate_schedule'],'constant')
        invalid_schedule=copy.deepcopy(c)
        invalid_schedule['learning_rate_schedule']='cosine'
        with self.assertRaisesRegex(ValueError,'constant learning rate'):
            train(invalid_schedule,'does-not-exist.jsonl')
        for key in ('learning_rate','weight_decay'):
            invalid=copy.deepcopy(c)
            invalid[key]=None
            with self.assertRaisesRegex(ValueError,'Set learning_rate'):
                train(invalid,'does-not-exist.jsonl')
        with torch.inference_mode():
            features=MiTB0().eval().forward_stages(torch.zeros(1,3,1024,1024))
        self.assertEqual([tuple(f.shape[1:]) for f in features],
                         [(32,256,256),(64,128,128),(160,64,64),(256,32,32)])

    def test_area_cap_minimal_and_per_image(self):
        omega=torch.ones(2,1,21,21,dtype=torch.bool)
        core=torch.zeros_like(omega); core[:,:,10,10]=True
        unknown=omega & ~core
        unknown[1]=False; unknown[1,:,8:12,8:12]=True; unknown[1] &= ~core[1]
        retained,steps=cap_unknown(unknown,omega,core,outer=21)
        self.assertEqual(steps.tolist(),[5,0])
        self.assertLessEqual(float(retained[0].sum()),.3*441)
        self.assertEqual(int(retained[0].sum()),120)
        self.assertTrue(retained[0,0,10,11])  # Core-adjacent unknown stays unknown.
        self.assertFalse(retained[0,0,0,0])
        self.assertTrue(torch.equal(retained[1],unknown[1]))

    def test_area_cap_rejects_infeasible_inner_band(self):
        omega=torch.ones(1,1,9,9,dtype=torch.bool)
        core=torch.zeros_like(omega); core[:,:,4,4]=True
        ring=dilate(core,9).bool() & ~core
        with self.assertRaisesRegex(ValueError,"preserving the core-adjacent band"):
            cap_unknown(ring,omega,core,outer=9,fraction=0)

    def test_original_area_weights_and_cap(self):
        image=Image.new('RGB',(80,40)); valid=Image.new('L',image.size,255)
        K=np.array([[40,0,40],[0,40,20],[0,0,1]],np.float32)
        for mode in ('pad','crop'):
            _,m,k,H=aligned_geometry(image,{'omega':valid},K,40,np.random.default_rng(17),False,mode,True)
            weights,total=original_area_weights(valid,H,40)
            self.assertEqual(int(total),3200)
            self.assertTrue(np.allclose(k.numpy(),H@K))
            self.assertAlmostEqual(abs(H[0,0]),abs(H[1,1]))
            omega=torch.tensor(np.array(m['omega'])>0)[None,None]
            core=torch.zeros_like(omega); core[:,:,20,20]=True
            unknown=omega & ~core
            retained,steps=cap_unknown(unknown,omega,core,outer=79,
                                       area_weights=weights[None],original_valid_area=total[None])
            self.assertLessEqual(float((retained*weights).sum()),.3*3200)
            previous_radius=39-(int(steps)-1)
            before=unknown & dilate(core,2*previous_radius+1).bool()
            if int(steps)>0:
                self.assertGreater(float((before*weights).sum()),.3*3200)
            expected=3200 if mode=='pad' else 1600
            self.assertEqual(int(weights.sum()),expected)
            self.assertFalse((retained & ~omega).any())

    def test_both_native_sizes_preserve_aspect_and_intrinsics(self):
        for w,h in ((2048,1536),(1920,1200)):
            image=Image.new('RGB',(w,h)); mask=Image.new('L',(w,h),255)
            K=np.array([[800,0,w/2],[0,800,h/2],[0,0,1]],np.float32)
            _,m,k,H=aligned_geometry(image,{'omega':mask},K,1024,np.random.default_rng(0),False,'pad',True)
            self.assertAlmostEqual(H[0,0],H[1,1])
            self.assertTrue(np.allclose(k.numpy(),H@K))
            weights,total=original_area_weights(mask,H,1024)
            self.assertEqual(int(total),w*h)
            self.assertEqual(int(weights.sum()),w*h)
            self.assertLess(np.count_nonzero(m['omega']),1024**2)

    def test_partition_and_candidates_follow_cap(self):
        core=torch.zeros(1,1,32,32,dtype=torch.bool); core[:,:,16,16]=True
        one=torch.ones_like(core,dtype=torch.float)
        omega=core.clone(); omega[:,:,2:30,2:30]=True
        s=build_states(core,omega,one,one,one*.97,one*.96)
        self.assertTrue(s.restored.any())
        self.assertFalse(s.admitted.any())
        self.assertTrue(torch.equal(s.negative,s.exterior|s.admitted|s.restored))
        self.assertTrue(torch.equal(s.positive|s.negative|s.unknown,omega))
        self.assertFalse((s.negative&s.unknown).any())
        self.assertTrue(torch.equal(s.candidates,s.unknown))
        self.assertFalse((s.candidates&s.restored).any())
        self.assertLessEqual(float(s.unknown.sum()),.3*float(omega.sum()))

    def test_confidence_control_retains_exterior(self):
        core=torch.zeros(1,1,40,40,dtype=torch.bool); core[:,:,20,20]=True
        one=torch.ones_like(core,dtype=torch.float)
        s=build_states(core,one,one,one,one*.01,one*.01,
                       negative_rule='confidence',point_support=one*0)
        self.assertFalse(s.admitted.any())
        self.assertTrue(s.negative[s.exterior].all())
        s=build_states(core,one,one,one,one*.01,one*.01,
                       negative_rule='confidence',point_support=one)
        self.assertTrue(torch.equal(s.admitted,s.ring))
        self.assertFalse(s.unknown.any())

    def test_matched_conf_uses_identical_teacher_cutoff(self):
        core=torch.zeros(1,1,40,40,dtype=torch.bool); core[:,:,20,20]=True
        one=torch.ones_like(core,dtype=torch.float)
        teacher=one*.08
        strict=build_states(core,one,one,one,teacher,teacher,
                            negative_rule='confidence',point_support=one)
        matched=build_states(core,one,one,one,teacher,teacher,
                             negative_rule='confidence_matched',point_support=one)
        self.assertFalse(strict.admitted.any())
        self.assertTrue(torch.equal(matched.admitted,matched.stable_background))

    def test_paper_core_audit_checks_pixel_identity(self):
        from pathlib import Path
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            rgb=root/'rgb.png'; Image.new('RGB',(4,3)).save(rgb)
            mask=root/'mask.png'; Image.fromarray(np.array([[0,255,0,0]]*3,dtype=np.uint8)).save(mask)
            duplicate=root/'duplicate.png'; Image.fromarray(np.array([[0,1,0,0]]*3,dtype=np.uint8)).save(duplicate)
            row={'id':'frame1','split':'test','rgb':str(rgb),'core':str(mask),
                 'evaluation_core':str(duplicate),'reference':str(mask)}
            audit=verify_paper_core_equality([row])
            self.assertEqual(audit['checked_frames'],1)
            self.assertEqual(len(audit['frames'][0]['binary_core_sha256']),64)
            Image.new('L',(4,3),0).save(duplicate)
            with self.assertRaisesRegex(ValueError,'differ at 3 pixels'):
                verify_paper_core_equality([row])

    def test_natural_degradation_ids_are_held_out(self):
        rows=[{'id':f'frame{i}','split':'test','scene':'heldout_scene','group':'heldout_group'}
              for i in range(40)]
        ids=[row['id'] for row in rows]
        self.assertEqual(verify_natural_degradation_subset(rows,ids)['frames'],40)
        rows[0]['split']='train'
        with self.assertRaisesRegex(ValueError,'not held-out test'):
            verify_natural_degradation_subset(rows,ids)

    def test_concat_control_is_runnable(self):
        c=config(); c['feature_fusion']='concat'
        model=MineDiffRoad(c,FixtureImage(),FixturePoints()).eval()
        out=model(*inputs())
        self.assertEqual(out['prob'].shape,(2,1,64,64))
        self.assertTrue(torch.isfinite(out['prob']).all())

    def test_post_warmup_training_backward(self):
        c=config(); model=MineDiffRoad(c,FixtureImage(),FixturePoints())
        image,clouds,K,T=inputs()
        core=torch.zeros(2,1,64,64,dtype=torch.bool); core[:,:,25:35,25:35]=True
        omega=torch.ones_like(core)
        view={'image':image,'points':clouds}
        with torch.no_grad():
            out=model(image,clouds,K,T)
        batch={'core':core,'omega':omega,'area_weights':omega.float(),
               'original_valid_area':torch.tensor([4096,4096]),'K':K,'T':T,
               'student':view,'weak1':view,'weak2':view,
               'quality':{'q_v':torch.ones_like(out['quality_logit_v']),
                          'q_a':torch.ones_like(out['quality_logit_a']),
                          'valid_v':torch.ones_like(out['quality_logit_v'],dtype=torch.bool),
                          'valid_a':out['point_support'].bool()}}
        loss,terms,counts=train_step(model,batch,11,c,EMATeacher(model,.99))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(counts['exterior_background_pixels'],0)
        self.assertLessEqual(counts['unknown_pixels'],.3*8192)

    def test_evaluator_keeps_scene_seed_and_recovery_fields(self):
        c=config(); model=MineDiffRoad(c,FixtureImage(),FixturePoints()).eval()
        image,clouds,K,T=inputs(); omega=torch.ones(2,1,64,64,dtype=torch.bool)
        reference=omega.clone(); reference[:,:,:20]=False
        core=reference.clone(); core[:,:,:,40:]=False
        batch={'id':['one','two'],'scene':['scene_a','scene_b'],'group':['heldout','heldout'],
               'student':{'image':image,'points':clouds},'K':K,'T':T,'omega':omega,
               'reference':reference,'evaluation_core':core}
        with tempfile.TemporaryDirectory() as folder:
            from pathlib import Path
            import csv
            output=Path(folder)/'eval'
            result=evaluate_loader(model,[batch],c,torch.device('cpu'),output)
            with (output/'per_image.csv').open(encoding='utf-8',newline='') as f:
                rows=list(csv.DictReader(f))
            self.assertEqual([r['scene'] for r in rows],['scene_a','scene_b'])
            self.assertEqual([int(r['seed']) for r in rows],[0,0])
            self.assertEqual(result['recovery']['mr']['valid_frames'],2)
            self.assertIn('tp',rows[0])

    def test_metrics_and_undefined_denominators(self):
        valid=np.ones(6,dtype=bool); y=np.array([1,1,1,0,0,0],bool)
        core=np.array([1,0,0,0,0,0],bool); p=np.array([1,1,0,1,0,0],bool)
        self.assertEqual(recovery_metrics(p,y,core,valid),{'mr':.5,'ep':.5})
        a=~core; s=np.array([0,1,0,1,1,0],bool)
        scores=selection_diagnostics(a,s,y,core,valid)
        self.assertEqual(scores['fns'],1/3); self.assertEqual(scores['br'],2/3)
        self.assertIsNone(selection_diagnostics(a,s&False,y,core,valid)['fns'])
        self.assertIsNone(recovery_metrics(core,y,y,valid)['mr'])
        with self.assertRaises(ValueError):
            selection_diagnostics(a,valid,y,core,valid)
        valid2=np.ones((5,5),bool); valid2[:,2]=False
        pred=postprocess(np.ones((5,5)),.5,11,valid2)
        self.assertFalse(pred.any())  # Padding cannot join two small components.


class StatisticsTests(unittest.TestCase):
    def records(self):
        rows=[]
        for seed in range(5):
            for scene,n in (('small',10),('large',100)):
                rows.append(dict(seed=seed,scene=scene,id=scene,tp=n,fp=0,fn=n,tn=2*n))
        return rows

    def test_paired_crossed_bootstrap(self):
        reference=self.records(); candidate=copy.deepcopy(reference)
        for row in candidate:
            row['tp']+=row['fn']//2; row['fn']//=2
        a=paired_iou_intervals(reference,candidate)
        b=paired_iou_intervals(reference,candidate)
        self.assertEqual(a,b)
        self.assertAlmostEqual(a['gain_pp'],25)
        np.testing.assert_allclose(a['paired_t_95_pp'],[25,25])
        np.testing.assert_allclose(a['crossed_bootstrap_95_pp'],[25,25])

    def test_statistics_reject_unpaired_or_unscoped_records(self):
        a=self.records(); b=copy.deepcopy(a)
        with self.assertRaisesRegex(ValueError,'identical paired'):
            paired_iou_intervals(a,b[:-1])
        b[0]['scene']=''
        with self.assertRaisesRegex(ValueError,'nonempty'):
            paired_iou_intervals(a,b)
        b=copy.deepcopy(a); b[0]['tp']+=1
        with self.assertRaisesRegex(ValueError,'identical valid'):
            paired_iou_intervals(a,b)

    def test_pixel_pooling_not_mean_frame_iou(self):
        a=self.records(); b=copy.deepcopy(a)
        for row in b:
            if row['scene']=='small':
                row['tp']*=2; row['fn']=0
        result=paired_iou_intervals(a,b,100)
        self.assertAlmostEqual(result['candidate_mean_iou'],120/220)
        self.assertNotAlmostEqual(result['candidate_mean_iou'],.75)
