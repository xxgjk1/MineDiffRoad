import csv
import datetime as dt
import hashlib
import json
import logging
import platform
import random
from pathlib import Path
import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader
from .data import RoadDataset, collate, to_device
from .model import MineDiffRoad, SegmentationPath, EMATeacher
from .supervision import build_states
from .losses import total_loss
from .metrics import postprocess, confusion, from_confusion, boundary_f1, recovery_metrics


def load_config(path, _seen=None):
    path = Path(path).resolve()
    seen = set() if _seen is None else set(_seen)
    if path in seen:
        raise ValueError("Cyclic config inheritance")
    seen.add(path)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if "base" in config:
        parent = load_config(path.parent / config.pop("base"), seen)
        for key, value in config.items():
            parent[key] = {**parent[key], **value} if isinstance(value,dict) and isinstance(parent.get(key),dict) else value
        config = parent
    if config["image_size"] % 4 or config["image_size"] < 32:
        raise ValueError("image_size must be >=32 and divisible by four")
    if config["num_points"] < 1 or config["voxel_size"] <= 0 or config["coordinate_noise_scale_m"] <= 0:
        raise ValueError("Point count/voxel size/coordinate noise scale must be positive")
    if config.get("image_encoder") != "mit_b0_four_stage":
        raise ValueError("This release requires image_encoder: mit_b0_four_stage; legacy stage-one checkpoints are incompatible")
    if "inner" in config["selection"]:
        raise ValueError("The paper ring directly adjoins the core; legacy selection.inner is unsupported")
    if config.get("resize_mode") not in {"pad", "crop"}:
        raise ValueError("resize_mode must be pad or crop")
    if config.get("learning_rate_schedule") != "constant":
        raise ValueError("learning_rate_schedule must be constant for this protocol")
    if config.get("batch_size", 0) < 1:
        raise ValueError("batch_size must be positive")
    return config


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Sparse convolution / CUDA scatter can still have rounding nondeterminism.


def select_device(name):
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable. LitePT requires CUDA; see CPU unit-test command.")
    return device


def new_output(path):
    path = Path(path) if path else Path("runs")/dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path.mkdir(parents=True,exist_ok=False)
    return path


def make_loader(dataset,config,train=False):
    return DataLoader(dataset,batch_size=config["batch_size"],shuffle=train,
                      num_workers=config["workers"],collate_fn=collate,
                      pin_memory=config["device"].startswith("cuda"),drop_last=False,
                      persistent_workers=False)


def forward_view(model,batch,view="student"):
    data = batch[view]
    return model(data["image"],data["points"],batch["K"],batch["T"])


def train_step(model,batch,epoch,config,teacher=None):
    warmup = epoch <= config["warmup_epochs"]
    if warmup:
        p1=p2=None
    else:
        if teacher is None:
            raise ValueError("Teacher must be initialized at warmup transition")
        with torch.no_grad():
            p1 = forward_view(teacher.model,batch,"weak1")["prob"]
            p2 = forward_view(teacher.model,batch,"weak2")["prob"]
    output = forward_view(model,batch)
    states = build_states(batch["core"],batch["omega"],output["rv"],output["ra"],
                          p1,p2,warmup=warmup,area_weights=batch.get("area_weights"),
                          original_valid_area=batch.get("original_valid_area"),
                          point_support=output["point_support_full"],**config["selection"])
    reconstruction = model.reconstruction(batch["student"]["image"],output,states) if states.candidates.any() else None
    loss,terms = total_loss(output,states,batch["quality"],reconstruction,config["loss_weights"],
                           config["skeleton_iterations"],config["confident_threshold"])
    counts = {"positive_pixels":int(states.positive.sum()),"negative_pixels":int(states.negative.sum()),
              "candidate_pixels":int(states.candidates.sum()),"unknown_pixels":int(states.unknown.sum()),
              "ring_admitted_pixels":int(states.admitted.sum()),"exterior_background_pixels":int(states.exterior.sum()),
              "area_restored_pixels":int(states.restored.sum()),"max_area_erosions":int(states.erosion_steps.max())}
    return loss,terms,counts


@torch.no_grad()
def evaluate_loader(model,loader,config,device,save_dir=None):
    model.eval()
    rows=[]
    totals=np.zeros(4,dtype=np.int64)
    for batch in loader:
        if "reference" not in batch:
            raise ValueError("Evaluation requires reference masks in every val/test record")
        batch = to_device(batch,device)
        output = forward_view(model,batch)
        for i,frame_id in enumerate(batch["id"]):
            p=output["prob"][i,0].cpu().numpy()
            y=batch["reference"][i,0].cpu().numpy().astype(bool)
            valid=batch["omega"][i,0].cpu().numpy().astype(bool)
            if not valid.any():
                raise ValueError(f"{frame_id}: no valid evaluation pixels")
            prediction=postprocess(p,config["threshold"],config["min_component_area"],valid)
            counts=confusion(prediction,y,valid)
            totals+=counts
            row={"id":frame_id,"scene":batch["scene"][i],"group":batch["group"][i],"seed":config["seed"],**from_confusion(counts),
                 "boundary_f1":boundary_f1(prediction,y,valid,config["boundary_tolerance"])}
            row.update(recovery_metrics(prediction,y,batch["evaluation_core"][i,0].cpu().numpy(),valid)
                       if "evaluation_core" in batch else {"mr":None,"ep":None})
            rows.append(row)
    if not rows:
        raise ValueError("Empty evaluation set")
    summary={"images":len(rows),"global":from_confusion(totals),"per_image_mean":{
        key:float(np.mean([r[key] for r in rows])) for key in ("road_iou","miou","dice","precision","recall","accuracy","boundary_f1")}}
    summary["recovery"] = {key:{"mean":float(np.mean(values)) if values else None,"valid_frames":len(values)}
                           for key in ("mr","ep") for values in [[r[key] for r in rows if r[key] is not None]]}
    if save_dir is not None:
        save_dir=Path(save_dir)
        save_dir.mkdir(parents=True,exist_ok=False)
        write_csv(save_dir/"per_image.csv",rows)
        (save_dir/"metrics.json").write_text(json.dumps(summary,indent=2))
    return summary


def write_csv(path,rows):
    with Path(path).open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def rng_state():
    n=np.random.get_state()
    return {"python":random.getstate(),"numpy":(n[0],n[1].tolist(),n[2],n[3],n[4]),
            "torch":torch.get_rng_state(),"cuda":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    n=state["numpy"]
    np.random.set_state((n[0],np.array(n[1],dtype=np.uint32),n[2],n[3],n[4]))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def train(config,manifest,output=None,resume=None):
    if config.get("learning_rate") is None or config.get("weight_decay") is None:
        raise ValueError("Set learning_rate and weight_decay before training. See configs/example.yaml for a run configuration example.")
    if config["learning_rate"] <= 0 or config["weight_decay"] < 0:
        raise ValueError("Invalid learning rate or weight decay")
    if config.get("learning_rate_schedule", "constant") != "constant":
        raise ValueError("This training protocol requires a constant learning rate")
    device=select_device(config["device"])
    if device.type != "cuda":
        raise RuntimeError("Real LitePT training requires CUDA")
    seed_everything(config["seed"])
    dataset=RoadDataset(manifest,"train",config)
    from .data import read_manifest
    all_rows=read_manifest(manifest)
    val_dataset=RoadDataset(manifest,"val",config) if any(r["split"]=="val" for r in all_rows) else None
    model=MineDiffRoad(config).to(device)
    init_info={"kind":"random_explicitly_enabled"}
    if not resume:
        if config["imagenet_checkpoint"]:
            init_info=model.segmentation.image_encoder.load_imagenet(config["imagenet_checkpoint"],config["imagenet_sha256"])
        elif not config["allow_random_image_init"]:
            raise ValueError("Set imagenet_checkpoint to official MiT-B0 ImageNet weights; random init requires an explicit config change")
    optimizer=torch.optim.AdamW(model.parameters(),lr=config["learning_rate"],weight_decay=config["weight_decay"])
    # Keep the optimizer LR constant; cosine_offset controls diffusion noise only.
    scaler=torch.amp.GradScaler("cuda",enabled=config["amp"])
    teacher=None
    start_epoch,best_iou,best_epoch,global_step=1,-1.,None,0
    if resume:
        checkpoint=torch.load(resume,map_location="cpu",weights_only=True)
        if checkpoint["config"] != config:
            raise ValueError("Resume config differs from the saved config; resume requires identical settings")
        if checkpoint["manifest_sha256"] != hashlib.sha256(Path(manifest).read_bytes()).hexdigest():
            raise ValueError("Resume manifest differs from the recorded training manifest")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint.get("scheduler") is not None:
            raise ValueError("Cannot resume a checkpoint that used a learning-rate scheduler")
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch=checkpoint["epoch"]+1
        global_step=checkpoint["global_step"]
        best_iou,best_epoch=checkpoint["best_iou"],checkpoint["best_epoch"]
        init_info=checkpoint["initialization"]
        if checkpoint["teacher"] is not None:
            teacher=EMATeacher(model,config["ema_decay"])
            teacher.model.load_state_dict(checkpoint["teacher"])
        restore_rng(checkpoint["rng"])
    if start_epoch > config["epochs"]:
        raise ValueError("Checkpoint already completed the configured training budget")
    out=new_output(output)
    logger=logging.getLogger(f"minediffroad.{out.resolve()}")
    logger.setLevel(logging.INFO)
    handler=logging.FileHandler(out/"train.log")
    logger.addHandler(handler)
    logger.addHandler(logging.StreamHandler())
    manifest_digest=hashlib.sha256(Path(manifest).read_bytes()).hexdigest()
    metadata={"python":platform.python_version(),"torch":str(torch.__version__),
              "cuda":torch.version.cuda,"cudnn":torch.backends.cudnn.version(),
              "gpu":torch.cuda.get_device_name(device),"initialization":init_info,
              "manifest_sha256":manifest_digest,"resume_from":str(resume) if resume else None}
    metadata["source_sha256"] = {p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in sorted(Path(__file__).parent.glob("*.py"))}
    (out/"config.yaml").write_text(yaml.safe_dump(config,sort_keys=False))
    (out/"environment.json").write_text(json.dumps(metadata,indent=2))
    with (out/"steps.csv").open("w",newline="") as log:
        writer=None
        for epoch in range(start_epoch,config["epochs"]+1):
            dataset.epoch=epoch
            model.train()
            if epoch > config["warmup_epochs"] and teacher is None:
                teacher=EMATeacher(model,config["ema_decay"])
            for batch in make_loader(dataset,config,train=True):
                batch=to_device(batch,device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda",dtype=torch.float16,enabled=config["amp"]):
                    loss,terms,counts=train_step(model,batch,epoch,config,teacher)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite loss at epoch {epoch}, step {global_step}")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if config["grad_clip_norm"] > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(),config["grad_clip_norm"])
                old_scale=scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                stepped=scaler.get_scale() >= old_scale
                if teacher is not None and stepped:
                    teacher.update(model)
                global_step+=1
                row={"epoch":epoch,"step":global_step,"loss":float(loss.detach()),
                     "lr":optimizer.param_groups[0]["lr"],"optimizer_step":int(stepped),
                     **{k:float(v.detach()) for k,v in terms.items()},**counts}
                if writer is None:
                    writer=csv.DictWriter(log,fieldnames=row.keys()); writer.writeheader()
                writer.writerow(row)
            log.flush()
            if val_dataset is not None:
                summary=evaluate_loader(model,make_loader(val_dataset,config),config,device,out/f"validation_{epoch:03d}")
                score=summary["global"]["road_iou"]
                # Within 0.02 percentage points prefer the earlier epoch.
                if score > best_iou+0.0002:
                    best_iou,best_epoch=score,epoch
            checkpoint={"format_version":3,"epoch":epoch,"global_step":global_step,"model":model.state_dict(),
                        "teacher":teacher.model.state_dict() if teacher else None,
                        "optimizer":optimizer.state_dict(),"scheduler":None,
                        "scaler":scaler.state_dict(),"rng":rng_state(),"config":config,
                        "manifest_sha256":manifest_digest,"initialization":init_info,
                        "best_iou":best_iou,"best_epoch":best_epoch}
            # New run directory + new filename each epoch: never overwrite checkpoints.
            torch.save(checkpoint,out/f"epoch_{epoch:03d}.pt")
            (out/"checkpoint_index.json").write_text(json.dumps({"last_epoch":epoch,"best_epoch":best_epoch,
                  "best_road_iou":best_iou if best_epoch else None,"resume_source":str(resume) if resume else None},indent=2))
            logger.info("epoch=%d last_loss=%.6f negatives=%d candidates=%d",epoch,row["loss"],row["negative_pixels"],row["candidate_pixels"])
            if config["empty_cache_each_epoch"]:
                torch.cuda.empty_cache()
    for active_handler in list(logger.handlers):
        active_handler.close()
        logger.removeHandler(active_handler)
    return out


def load_deployment(checkpoint_path,device):
    checkpoint=torch.load(checkpoint_path,map_location="cpu",weights_only=True)
    config=checkpoint["config"]
    if config.get("image_encoder") != "mit_b0_four_stage":
        raise ValueError("Legacy first-stage checkpoints cannot be loaded as the four-stage paper-aligned model")
    seed_everything(config["seed"])
    model=SegmentationPath(config).to(device)
    if checkpoint.get("deployment",False):
        state=checkpoint["state_dict"]
    else:
        state={k.removeprefix("segmentation."):v for k,v in checkpoint["model"].items() if k.startswith("segmentation.")}
    model.load_state_dict(state,strict=True)
    return model.eval(),config


@torch.no_grad()
def predict(checkpoint_path,manifest,split,output,device_name,save_probabilities=False):
    device=select_device(device_name)
    model,config=load_deployment(checkpoint_path,device)
    config={**config,"device":device_name}
    dataset=RoadDataset(manifest,split,config)
    out=new_output(output)
    for batch in make_loader(dataset,config):
        batch=to_device(batch,device)
        probabilities=forward_view(model,batch)["prob"].cpu().numpy()
        for i,frame_id in enumerate(batch["id"]):
            # User IDs can contain path separators; a hash prevents collisions/traversal.
            stem=hashlib.sha256(frame_id.encode()).hexdigest()[:16]
            mask=postprocess(probabilities[i,0],config["threshold"],config["min_component_area"],batch["omega"][i,0].cpu().numpy())
            Image.fromarray(mask.astype(np.uint8)*255).save(out/f"{stem}.png")
            if save_probabilities:
                np.save(out/f"{stem}.npy",probabilities[i,0])
    (out/"index.json").write_text(json.dumps({r["id"]:hashlib.sha256(r["id"].encode()).hexdigest()[:16]+".png" for r in dataset.rows},indent=2))
    return out
