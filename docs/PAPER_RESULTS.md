# Paper results

[Back to README](../README.md)

Tables 1–6 are transcribed from the supplied 2026-10-11 manuscript. Numerical cells preserve the source values. Percentages are used unless units are specified; pp denotes percentage points. Bold follows the manuscript. Sources and SHA-256 hashes are recorded in [ASSET_PROVENANCE.json](ASSET_PROVENANCE.json).

The main online comparison and fixed-trajectory replay use different experimental controls. In particular, 68.57% and 68.76% IoU belong to these two different settings. See each table note before comparing rows.

**Metrics:** IoU is road intersection over union; MR is omitted-road recall; EP is expansion precision; FNS is erroneous negative supervision; BR is background retention. LPR is long-path recall, XCR measures incorrect cross-component connections, and endpoint FCR measures false connections conditional on endpoint coverage.

## Table 1: Overall comparison

| Method | IoU ↑ | MR ↑ | EP ↑ |
| --- | ---: | ---: | ---: |
| Partial BCE, trusted positives only | 60.84 | 55.27 | 41.63 |
| Stable background + hard teacher targets | 62.33 | 64.18 | 46.72 |
| nnPU adaptation | 61.92 | 47.36 | 48.51 |
| GCE robust loss | 62.16 | 49.83 | 48.92 |
| SCE robust loss | 62.51 | 51.24 | 49.38 |
| Mean Teacher, multimodal adaptation | 63.47 | 50.65 | 49.84 |
| UniMatch, multimodal adaptation | 64.21 | 54.38 | 50.61 |
| CorrMatch, multimodal adaptation | 64.79 | 57.26 | 51.17 |
| UniMatch V2, protocol matched | 65.62 | 60.48 | 51.83 |
| CSL adaptive selection | 65.94 | 63.56 | 52.42 |
| UniMatch V2, stronger image backbone † | 66.52 | 66.31 | 52.87 |
| CSL, stronger image backbone † | 66.73 | 67.84 | 53.16 |
| Concat, matched architecture | 63.58 | 71.42 | 46.93 |
| Conf, matched architecture | 64.94 | 73.68 | 50.87 |
| Reliability selection only | 67.61 | 76.84 | 53.73 |
| **MineDiffRoad** | **68.57** | **79.13** | **55.12** |
| MineDiffRoad with complete training labels | 69.21 | 80.42 | 55.79 |

All rows except the complete-label reference use conservative annotations. † Stronger image backbones are DINOv2-Base for UniMatch V2 and SegFormer-B5 for CSL, with LitePT-S and late fusion. The complete-label reference has a different auxiliary update schedule.

[CSV](results/table_1_overall.csv)

## Table 2(a): Negative placement at a fixed quota

| Rule | Quota (%) | FNS ↓ | BR ↑ | IoU ↑ | MR ↑ | EP ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Random | 60.00 | 4.62 | 60.03 | 65.44 | 70.83 | 47.92 |
| Confidence | 60.00 | 3.81 | 60.52 | 66.14 | 72.61 | 49.84 |
| MC uncertainty, eight passes | 60.00 | 3.18 | 60.93 | 66.91 | 74.83 | 51.42 |
| Reliability | 60.00 | **2.91** | **61.08** | **68.76** | **78.21** | **54.13** |

Teacher outputs, candidate domain, positive candidates, and the negative quota are shared. FNS and BR are computed per image and averaged. Entropy and margin induce the same ranking as Confidence in this binary stable-background protocol. MC dropout uses eight passes for ranking only and is disabled during student optimization.

[CSV](results/table_2a_negative_placement.csv)

## Table 2(b): Fusion and selection

| Feature fusion | Negative ranking | IoU ↑ |
| --- | ---: | ---: |
| Concatenation | Confidence | 64.93 |
| Concatenation | Reliability | 67.92 |
| Reliability weighted | Confidence | 66.14 |
| Reliability weighted | Reliability | **68.76** |

Reliability heads and losses are identical across the factorial cells. Averaging the cells gives a selection effect of 2.81 percentage points, a fusion effect of 1.03 points, and an interaction of −0.37 points, as reported in the paper.

[CSV](results/table_2b_fusion_selection.csv)

## Table 2(c): Feature feedback

| Ranking signal | FNS ↓ | IoU ↑ |
| --- | ---: | ---: |
| Teacher background confidence | 3.81 | 66.14 |
| Low level texture, geometry, and support | 3.17 | 67.16 |
| Joint features with stopped input gradients | 3.03 | 67.62 |
| Default reliability heads | **2.91** | **68.76** |

Stopping input gradients restricts feedback to conditioning features but retains the semantic information carried by those features.

[CSV](results/table_2c_feature_feedback.csv)

## Table 3(a): Candidate use and residual bound

| Recovery rule | IoU ↑ | EP ↑ |
| --- | ---: | ---: |
| Direct supervision with hard candidates | 67.09 | 50.67 |
| Direct supervision with soft candidates | 67.37 | 52.27 |
| Reconstruction on the valid set + bounded residual | **68.76** | **54.13** |
| Reconstruction on the valid set, residual unbounded | 67.59 | 51.47 |

Teacher outputs, candidates, auxiliary steps, and capacity are fixed. The 68.76% IoU in this replay setting is distinct from the 68.57% result under online training.

[CSV](results/table_3a_candidate_use.csv)

## Table 3(b): Noise levels and time conditioning

| Recovery training variant | Standard IoU ↑ | Mean over seven conditions ↑ | Training seconds/epoch ↓ |
| --- | ---: | ---: | ---: |
| Direct soft candidates | 67.49 | 62.71 | 64 |
| Ordinary denoising | 67.86 | 63.23 | 82 |
| Multiple noise levels, no time input | 68.09 | 63.57 | 111 |
| Reconstruction conditioned on diffusion time | **68.57** | **64.05** | 114 |

These runs use independently evolving online teachers. Training time measures the additional reconstruction work; the reconstruction branch is removed at inference.

[CSV](results/table_3b_noise_conditioning.csv)

## Table 4: Other annotation and domain settings

| Evaluation | MineDiffRoad IoU ↑ | Reference method | Reference IoU ↑ | Difference |
| --- | ---: | ---: | ---: | ---: |
| Human conservative masks | 67.57 | Concat | 65.82 | +1.75 pp |
| Mine-B, no adaptation | 63.27 | CSL | 62.13 | +1.14 pp |
| RELLIS-3D surface task | 63.49 | CSL | 62.41 | +1.08 pp |
| Natural degradation mean | 64.05 | Conf | 60.72 | +3.33 pp |

IoU and differences are averaged over five seeds. Human masks use a separate batch from the principal mine. Mine-B uses frozen SegMine checkpoints without adaptation or threshold adjustment. RELLIS-3D models are trained independently with asphalt, concrete, dirt, mud, and grass as foreground. The seven natural degradation conditions are equally weighted.

[CSV](results/table_4_generalization.csv)

## Table 5: Connectivity

| Method | clDice ↑ | LPR ↑ | XCR ↓ | Endpoint FCR ↓ |
| --- | ---: | ---: | ---: | ---: |
| Concat | 73.13 | 42.83 | 58.02 | 100.00 |
| CorrMatch | 73.87 | 46.51 | 48.03 | 99.19 |
| MineDiffRoad | 74.83 | 53.13 | 41.37 | 99.06 |
| Complete label model | **76.51** | **60.03** | **38.01** | **96.94** |

LPR evaluates 2,480 pairs within the same reference component; XCR evaluates 5,000 pairs from components that should remain separate. Endpoint FCR uses 840 broader negative endpoint tasks, is conditional on both endpoints being covered, and pools five seeds. Its denominator differs from XCR and the human tasks. Raw MineDiffRoad endpoint FCR is 99.06%, exposing substantial errors on this task set.

[CSV](results/table_5a_connectivity.csv)

## Table 5(b): Independent human review

| Method | Success when connection is required ↑ | False bridge when separation is required ↓ |
| --- | ---: | ---: |
| Concat | 45.11 | 20.21 |
| CorrMatch | 49.31 | 16.43 |
| MineDiffRoad | **56.22** | **11.81** |
| Complete label model | 62.81 | 8.71 |

Independent review uses 1,000 connection tasks and 1,000 separation tasks. The complete-label model uses different supervision. In a separate optional graph-pruning analysis, MineDiffRoad endpoint FCR changes to 30.00%, while endpoint coverage falls from 76.19% to 71.43% and LPR falls from 53.13% to 50.63%. This optional graph constraint is outside the supplied segmentation CLI.

[CSV](results/table_5b_human_connectivity.csv)

## Table 6(a): Model size

| Configuration | Parameters | GFLOPs |
| --- | ---: | ---: |
| Deployed model | 35.6 M | 92.4 |
| Training model | 41.2 M unique | — |

Training parameters are unique learned parameters; deployment excludes the reconstruction branch. These values describe the model configuration measured in the paper. The source adapter uses the MiT-B0 stride-4 first-stage interface described in IMPLEMENTATION_NOTES.md; resource use depends on the implementation and environment.

[CSV](results/table_6a_parameters.csv)

## Table 6(b): Runtime

| Timed scope | Mean (ms) ↓ | P95 (ms) ↓ | FPS ↑ |
| --- | ---: | ---: | ---: |
| Network forward pass | 45.3 | 52.1 | 22.1 |
| Loading, alignment, forward pass, postprocessing | 59.1 | 70.4 | 16.9 |
| With optional graph constraint | 64.8 | 77.6 | 15.4 |

The paper uses FP16, batch size one, RTX 4090 and Xeon Platinum 8380, with 200 initialization runs and 500 synchronized runs. Sensor acquisition and transmission are excluded. Optional graph pruning is a separate processing step.

[CSV](results/table_6b_runtime.csv)

