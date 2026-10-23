import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from .engine import load_config,train,select_device,load_deployment,evaluate_loader,make_loader,predict,new_output
from .data import (read_manifest,RoadDataset,verify_paper_segmine,
                   verify_paper_core_equality,verify_natural_degradation_subset)
from .geometry import farthest_points


def main():
    parser=argparse.ArgumentParser(description="MineDiffRoad: reliability-guided road segmentation")
    subs=parser.add_subparsers(dest="command",required=True)
    p=subs.add_parser("train")
    p.add_argument("--config",default="configs/paper.yaml")
    p.add_argument("--manifest",required=True)
    p.add_argument("--output")
    p.add_argument("--resume")
    for command in ("evaluate","predict"):
        p=subs.add_parser(command)
        p.add_argument("--checkpoint",required=True)
        p.add_argument("--manifest",required=True)
        p.add_argument("--split",choices=["val","test","predict"],default="test")
        p.add_argument("--device",default="cuda")
        p.add_argument("--output",required=True)
        if command == "predict":
            p.add_argument("--save-probabilities",action="store_true")
    p=subs.add_parser("export")
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--output",required=True)
    p=subs.add_parser("verify-data")
    p.add_argument("--manifest",required=True)
    p.add_argument("--paper-segmine",action="store_true",help="check the reported SegMine split, scene, and group counts")
    p.add_argument("--paper-core-equality",action="store_true",help="pixelwise audit that held-out core and evaluation_core are identical")
    p.add_argument("--natural-degradation-ids",help="JSON list of the 40 natural-degradation frame IDs; require all to be held-out test frames")
    p.add_argument("--output",help="write the JSON audit to a new file")
    p=subs.add_parser("prepare-fps")
    p.add_argument("--manifest",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--num-points",type=int,default=16384)
    p.add_argument("--device",default="cuda")
    p=subs.add_parser("statistics",help="Paired seed t and crossed scene/seed intervals from per-frame counts")
    p.add_argument("--reference",required=True)
    p.add_argument("--candidate",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--resamples",type=int,default=2000)
    p.add_argument("--seed",type=int,default=17)
    args=parser.parse_args()
    if args.command == "statistics":
        from .statistics import read_counts, paired_iou_intervals
        result = paired_iou_intervals(read_counts(args.reference),read_counts(args.candidate),args.resamples,args.seed)
        output = Path(args.output)
        output.parent.mkdir(parents=True,exist_ok=True)
        with output.open("x",encoding="utf-8") as stream:
            json.dump(result,stream,indent=2)
        print(json.dumps(result,indent=2))
    elif args.command == "train":
        print(train(load_config(args.config),args.manifest,args.output,args.resume))
    elif args.command == "verify-data":
        rows=read_manifest(args.manifest)
        result=(verify_paper_segmine(rows) if args.paper_segmine else
                {"records":len(rows),"splits":{s:sum(r["split"]==s for r in rows) for s in sorted({r["split"] for r in rows})}})
        if args.paper_core_equality:
            result["paper_core_equality"]=verify_paper_core_equality(rows)
        if args.natural_degradation_ids:
            frame_ids=json.loads(Path(args.natural_degradation_ids).read_text(encoding="utf-8"))
            result["natural_degradation"]=verify_natural_degradation_subset(rows,frame_ids)
        if args.output:
            out=Path(args.output)
            out.parent.mkdir(parents=True,exist_ok=True)
            with out.open("x",encoding="utf-8") as stream:
                json.dump(result,stream,indent=2)
        print(json.dumps(result,indent=2))
    elif args.command == "export":
        out=Path(args.output)
        if out.exists():
            raise FileExistsError(out)
        checkpoint=torch.load(args.checkpoint,map_location="cpu",weights_only=True)
        if checkpoint['config'].get('image_encoder') != 'mit_b0_four_stage':
            raise ValueError('Legacy first-stage checkpoints cannot be exported as the four-stage model')
        if checkpoint.get("deployment",False):
            state=checkpoint["state_dict"]
        else:
            state={k.removeprefix("segmentation."):v for k,v in checkpoint["model"].items() if k.startswith("segmentation.")}
        if not state:
            raise ValueError("Checkpoint contains no segmentation weights")
        out.parent.mkdir(parents=True,exist_ok=True)
        torch.save({"deployment":True,"config":checkpoint["config"],"state_dict":state,
                    "source_checkpoint_sha256":hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()},out)
        print(out)
    elif args.command == "predict":
        print(predict(args.checkpoint,args.manifest,args.split,args.output,args.device,args.save_probabilities))
    elif args.command == "evaluate":
        device=select_device(args.device)
        model,config=load_deployment(args.checkpoint,device)
        config={**config,"device":args.device}
        data=RoadDataset(args.manifest,args.split,config)
        result=evaluate_loader(model,make_loader(data,config),config,device,args.output)
        print(json.dumps(result,indent=2))
    elif args.command == "prepare-fps":
        if args.num_points < 1:
            raise ValueError("num-points must be positive")
        device=select_device(args.device)
        rows=read_manifest(args.manifest)
        out=new_output(args.output).resolve()
        metadata=[]
        for row in rows:
            points=np.load(row["points"],allow_pickle=False)
            if points.ndim != 2 or points.shape[1] != 4 or not np.isfinite(points).all():
                raise ValueError(f"Invalid point cloud: {row['points']}")
            ids=farthest_points(torch.tensor(points[:,:3],device=device,dtype=torch.float32),args.num_points)
            filename=hashlib.sha256(row["id"].encode()).hexdigest()[:16]+".npy"
            np.save(out/filename,ids.cpu().numpy())
            row["fps_indices"]=str(out/filename)
            row["points_sha256"]=hashlib.sha256(Path(row["points"]).read_bytes()).hexdigest()
            metadata.append({"id":row["id"],"point_file_sha256":row["points_sha256"],
                             "count":len(ids),"fps_indices":filename})
        (out/"manifest.jsonl").write_text("\n".join(json.dumps(r) for r in rows)+"\n")
        (out/"fps_metadata.json").write_text(json.dumps(metadata,indent=2))
        print(out/"manifest.jsonl")


if __name__ == "__main__":
    main()
