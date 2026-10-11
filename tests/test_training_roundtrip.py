"""Opt-in integration test; all inputs/outputs live in a temporary directory.

This validates file IO, EMA initialization and checkpoint resume. It is NOT a
dataset experiment and generated numerical outputs are never distributed.
"""
import json
import os
from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image
import torch
from minediffroad.engine import train,load_deployment,predict
from tests.test_core import config


@unittest.skipUnless(os.environ.get("MINEDIFFROAD_TEST_ROUNDTRIP")=="1" and torch.cuda.is_available(),
                     "opt-in checkpoint/CLI integration check")
class TrainingRoundtrip(unittest.TestCase):
    def test_training_resume_deployment(self):
        with tempfile.TemporaryDirectory(prefix="minediffroad_software_test_") as folder:
            root=Path(folder)
            rng=np.random.default_rng(17)
            rows=[]
            for i,split in enumerate(["train","train","val"]):
                stem=str(i)
                Image.fromarray(rng.integers(0,255,(64,64,3),dtype=np.uint8)).save(root/f"{stem}_rgb.png")
                cloud=np.column_stack([rng.uniform(-5,5,(256,2)),rng.uniform(8,14,256),rng.uniform(0,1,256)]).astype(np.float32)
                np.save(root/f"{stem}_points.npy",cloud)
                mask=np.zeros((64,64),np.uint8); mask[20:44,20:44]=255
                Image.fromarray(mask).save(root/f"{stem}_mask.png")
                (root/f"{stem}_calibration.json").write_text(json.dumps({"K":[[50,0,32],[0,50,32],[0,0,1]],"T_lidar_to_camera":np.eye(4).tolist()}))
                row={"id":stem,"group":split,"split":split,"rgb":f"{stem}_rgb.png","points":f"{stem}_points.npy",
                     "calibration":f"{stem}_calibration.json","core" if split=="train" else "reference":f"{stem}_mask.png"}
                rows.append(row)
            manifest=root/"manifest.jsonl"; manifest.write_text("\n".join(json.dumps(r) for r in rows))
            c=config(); c.update(device="cuda",epochs=2,warmup_epochs=1,allow_random_image_init=True,amp=False)
            run=train(c,manifest,root/"run")
            checkpoint=torch.load(run/"epoch_002.pt",map_location="cpu",weights_only=True)
            self.assertIsNotNone(checkpoint["teacher"])
            self.assertEqual(checkpoint["global_step"],2)
            resumed=train(c,manifest,root/"resumed",run/"epoch_001.pt")
            replay=torch.load(resumed/"epoch_002.pt",map_location="cpu",weights_only=True)
            self.assertEqual(replay["global_step"],2)
            self.assertEqual(replay["scheduler"],checkpoint["scheduler"])
            # Compare resume trajectory with uninterrupted training, allowing CUDA rounding.
            self.assertTrue(torch.allclose(replay["model"]["segmentation.base_logits.weight"],
                                           checkpoint["model"]["segmentation.base_logits.weight"],atol=1e-6,rtol=1e-5))
            model,saved_config=load_deployment(run/"epoch_002.pt",torch.device("cuda"))
            self.assertFalse(any("reconstruction" in k for k in model.state_dict()))
            deployment=root/"deployment.pt"
            torch.save({"deployment":True,"config":saved_config,"state_dict":model.state_dict()},deployment)
            out=predict(deployment,manifest,"val",root/"predictions","cuda",True)
            self.assertEqual(len(list(out.glob("*.png"))),1)
            self.assertEqual(len(list(out.glob("*.npy"))),1)
            self.assertTrue((run/"validation_002"/"per_image.csv").is_file())


if __name__ == "__main__":
    unittest.main()
