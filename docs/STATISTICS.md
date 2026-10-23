# Statistical interface

Compute paired seed t intervals and crossed scene/seed bootstrap intervals from per-frame confusion counts.

## Inputs

Run evaluate for each reporting seed, then concatenate its per_image.csv records for each method, preserving one header. Required columns:

| Column | Meaning |
| --- | --- |
| seed | Actual training seed |
| scene | Prespecified scene identifier, consistent across methods and seeds |
| id | Frame identifier |
| tp, fp, fn, tn | Integer confusion counts after fixed postprocessing and valid-mask exclusion |

group controls train/validation/test separation. scene controls scene resampling. They are not silently treated as the same field. Add scene to the data manifest before evaluation. Both methods must have identical paired (seed, scene, id) records, and every seed must cover the same scene/frame panel. At least two seeds/scenes are needed; the paper uses five seeds. Valid-pixel totals and reference-positive totals must agree for paired records.

```bash
python -m minediffroad statistics --reference runs/conf_all_seeds.csv --candidate runs/ours_all_seeds.csv --output runs/paired_intervals.json --resamples 2000 --seed 17
```

The random seed here controls resampling only.

## Calculation

1. Pool tp/fp/fn within each seed to compute road IoU.
2. Form paired candidate-minus-reference IoU differences across seeds and calculate a two-sided 95% Student t interval.
3. Draw seeds and scenes independently with replacement (crossed bootstrap), using the same draws for both methods.
4. Re-pool sampled scene counts separately within each sampled seed, then average the seed IoUs.
5. Report percentile 2.5%/97.5% bounds over 2,000 draws. Differences and intervals are in percentage points.

The CLI writes a new JSON file and refuses to overwrite an existing result. It does not apply multiplicity correction. IoU counts are re-pooled for each resample using the supplied scene and seed identifiers.

## Selection audit

metrics.selection_diagnostics(A, S, Y_star, Y_eval_core, Omega) returns fractions:

- FNS = |S intersect (Y_star minus Y_eval_core)| / |S|.
- BR = |S intersect background| / |A intersect background|.

A is the common stable-background pool. S is the selected subset before area control. Exterior background B_out and restored background F⁻ are excluded. Invalid-domain selections are rejected. Empty denominators yield None and must be excluded from per-image means; report the number of valid images alongside the mean.

The ranking function takes only domain and scores. Complete references enter the audit after selection. fixed_quota_selection uses floor(fraction × number of candidates), stable descending sorting, and row-major tie breaking.

recovery_metrics returns MR and EP relative to the frozen evaluation core. The evaluator reports their means and valid-frame counts when evaluation_core is present. It never synthesizes a core from complete labels inside training.
