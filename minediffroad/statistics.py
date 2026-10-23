"""Paired seed t intervals and crossed scene/seed bootstrap from real counts.
"""
import csv
from pathlib import Path
import numpy as np
from scipy.stats import t


def read_counts(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def paired_iou_intervals(reference_rows, candidate_rows, resamples=2000, seed=17):
    if resamples < 1:
        raise ValueError("resamples must be positive")
    def index(rows):
        result = {}
        for row in rows:
            key = tuple(str(row[field]).strip() for field in ("seed","scene","id"))
            if not all(key) or key in result:
                raise ValueError("Unique nonempty seed/scene/id records are required")
            counts = np.asarray([float(row[field]) for field in ("tp","fp","fn","tn")])
            if not np.isfinite(counts).all() or (counts < 0).any() or (counts != np.floor(counts)).any():
                raise ValueError("Confusion counts must be finite nonnegative integers")
            result[key] = counts
        return result
    reference, candidate = index(reference_rows), index(candidate_rows)
    if not reference or reference.keys() != candidate.keys():
        raise ValueError("Both methods must have identical paired frame records")
    seeds = sorted({key[0] for key in reference})
    scenes = sorted({key[1] for key in reference})
    if len(seeds) < 2 or len(scenes) < 2:
        raise ValueError("At least two seeds and scenes are required")
    common = {(s,i) for k,s,i in reference if k == seeds[0]}
    for k in seeds:
        if {(s,i) for run,s,i in reference if run == k} != common:
            raise ValueError("Every seed must cover the same scene/frame panel")
    arrays = np.zeros((2,len(seeds),len(scenes),4),dtype=np.float64)
    for key, counts in reference.items():
        other = candidate[key]
        if counts.sum() != other.sum() or counts[0]+counts[2] != other[0]+other[2]:
            raise ValueError("Paired methods must use identical valid pixels and references")
        a,b = seeds.index(key[0]),scenes.index(key[1])
        arrays[0,a,b] += counts
        arrays[1,a,b] += other
    def iou(counts):
        denominator = counts[..., :3].sum(-1)
        return np.divide(counts[...,0],denominator,out=np.ones_like(denominator),where=denominator>0)
    per_seed = iou(arrays.sum(2))
    differences = (per_seed[1]-per_seed[0])*100
    mean = float(differences.mean())
    halfwidth = float(t.ppf(.975,len(seeds)-1)*differences.std(ddof=1)/np.sqrt(len(seeds)))
    rng = np.random.default_rng(seed)
    boot = np.empty(resamples)
    for n in range(resamples):
        seed_ids = rng.integers(len(seeds),size=len(seeds))
        scene_ids = rng.integers(len(scenes),size=len(scenes))
        # Use the same draws for both methods. Re-pool counts per sampled seed.
        scores = iou(arrays[:,seed_ids][:,:,scene_ids].sum(2)).mean(1)
        boot[n] = (scores[1]-scores[0])*100
    return {"gain_pp":mean,"paired_t_95_pp":[mean-halfwidth,mean+halfwidth],
            "crossed_bootstrap_95_pp":np.quantile(boot,[.025,.975]).tolist(),
            "reference_mean_iou":float(per_seed[0].mean()),"candidate_mean_iou":float(per_seed[1].mean()),
            "seeds":len(seeds),"scenes":len(scenes),"resamples":resamples,"random_seed":seed,
            "unit":"scene and seed (crossed, paired between methods)",
            "statistic":"pooled pixel IoU per seed, then mean across seeds"}
