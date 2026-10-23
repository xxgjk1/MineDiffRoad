# Paper results

[Back to README](../README.md)

Results and numbering follow the supplied 40-page manuscript. All rates are percentages unless noted. Sources and hashes are in [ASSET_PROVENANCE.json](ASSET_PROVENANCE.json). Table B.6 defines supervision sets and is represented in the [method map](EQUATION_MAP.md).

IoU pools pixel counts within each seed, then averages five seeds. MR/EP and FNS/BR are per-image ratios averaged over defined denominators. Full online training and frozen-teacher replay have different controls.

## Table 1: Overall comparison

| Method | IoU ↑ | MR ↑ | EP ↑ |
| --- | --- | --- | --- |
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
| MineDiffRoad | 68.57 | 79.13 | 55.12 |
| MineDiffRoad with complete training labels | 69.21 | 80.42 | 55.79 |

Full online training. Means over five seeds. The complete-label model is a separate supervision reference, not a conservative-label competitor. Larger-backbone references use DINOv2-Base for UniMatch V2 and SegFormer-B5 for CSL. MR and EP are relative to the frozen evaluation core.

[CSV](results/table_1_overall.csv)

## Table 2(a): Negative selection at a fixed quota

| Rule | Quota (%) | FNS ↓ | BR ↑ | IoU ↑ | MR ↑ | EP ↑ |
| --- | --- | --- | --- | --- | --- | --- |
| Random | 60.00 | 2.13 | 60.03 | 65.44 | 70.83 | 47.92 |
| Confidence | 60.00 | 2.16 | 60.52 | 66.14 | 72.61 | 49.84 |
| MC uncertainty, eight passes | 60.00 | 2.03 | 60.93 | 66.91 | 74.83 | 51.42 |
| Reliability | 60.00 | 1.72 | 61.08 | 68.76 | 78.21 | 54.13 |

Frozen teacher snapshots, common candidate domain A, positive candidates, and a 60% quota. FNS/BR average per-image ratios. FNS is false negative supervision: selected negatives overlapping omitted reference road divided by all selected negatives. BR divides selected true background by eligible true background. B_out and F⁻ are excluded. MC uncertainty uses eight passes for ranking. Reliability ranks min(rv,ra).

[CSV](results/table_2a_negative_placement.csv)

## Table 2(b): Reliability-input controls

| Ranking signal | IoU ↑ |
| --- | --- |
| Teacher background confidence | 66.14 |
| Low level texture, geometry, and support | 67.16 |
| Joint features with stopped input gradients | 67.62 |
| Default reliability heads | 68.76 |

Same frozen replay protocol. The current manuscript reports IoU only in this panel. Stopping gradients does not erase semantic information already encoded in joint features.

[CSV](results/table_2b_reliability_inputs.csv)

## Table 3: Fusion and selection

| Feature fusion | Negative ranking | IoU ↑ |
| --- | --- | --- |
| Concatenation | Confidence | 64.93 |
| Concatenation | Reliability | 67.92 |
| Reliability weighted | Confidence | 66.14 |
| Reliability weighted | Reliability | 68.76 |

The CSV uses long form for the same four factorial cells. Ranking gains are 2.99 points under concatenation and 2.62 under weighted fusion. Reported average ranking/fusion effects are 2.81/1.03 points, with interaction −0.37 points. These effects concern this controlled replay, not every task.

[CSV](results/table_3_fusion_selection.csv)

## Table 4: Candidate supervision and bounded refinement

| Recovery rule | IoU ↑ | EP ↑ |
| --- | --- | --- |
| Direct supervision with hard candidates | 67.09 | 50.67 |
| Direct supervision with soft candidates | 67.37 | 52.27 |
| Reconstruction on the valid set + bounded residual | 68.76 | 54.13 |
| Reconstruction on the valid set, residual unbounded | 67.59 | 51.47 |

Frozen teacher outputs, candidate sets, auxiliary steps, and model capacity. The 68.76% replay result is distinct from the 68.57% full-training result.

[CSV](results/table_4_candidate_use.csv)

## Table 5(a): Continuity and separation

| Method | clDice ↑ | LPR ↑ | XCR ↓ | Endpoint FCR ↓ |
| --- | --- | --- | --- | --- |
| Concat | 73.13 | 42.83 | 58.02 | 100.00 |
| CorrMatch | 73.87 | 46.51 | 48.03 | 99.19 |
| MineDiffRoad | 74.83 | 53.13 | 41.37 | 99.06 |
| Complete label model | 76.51 | 60.03 | 38.01 | 96.94 |

LPR means local path recovery: 2,480 pairs within the same reference component. XCR is the connection rate across components: 5,000 pairs from different components. Endpoint FCR uses a separate 840-task set, pools five seeds, and conditions on both endpoints being covered. Its denominator differs from XCR. The complete-label row is a separate reference.

[CSV](results/table_5a_connectivity.csv)

## Table 5(b): Independent human connectivity review

| Method | Success when connection is required ↑ | False bridge when separation is required ↓ |
| --- | --- | --- |
| Concat | 45.11 | 20.21 |
| CorrMatch | 49.31 | 16.43 |
| MineDiffRoad | 56.22 | 11.81 |
| Complete label model | 62.81 | 8.71 |

1,000 required-connection tasks and 1,000 required-separation tasks. Optional graph pruning reduces MineDiffRoad conditional endpoint FCR from 99.06% to 30.00%, while coverage falls from 76.19% to 71.43% and LPR from 53.13% to 50.63%.

[CSV](results/table_5b_human_connectivity.csv)

## Table G.7: Model size

| Configuration | Parameters | GFLOPs |
| --- | --- | --- |
| Deployed model | 35.6 M | 92.4 |
| Training model | 41.2 M unique | — |

Training counts unique learned parameters. Deployment excludes reconstruction.

[CSV](results/table_g7_parameters.csv)

## Figure 8: Reconstruction benefit and training cost

| Recovery training variant | Standard IoU ↑ | Mean over seven conditions ↑ | Training seconds/epoch ↓ |
| --- | --- | --- | --- |
| Direct soft candidates | 67.49 | 62.71 | 64 |
| Ordinary denoising | 67.86 | 63.23 | 82 |
| Multiple noise levels, no time input | 68.09 | 63.57 | 111 |
| Reconstruction conditioned on diffusion time | 68.57 | 64.05 | 114 |

Full training with independently evolving teachers, not Table 4's frozen replay. Degraded IoU is the equally weighted mean of seven natural conditions. Time measures complete training time per epoch in the reported setup, not isolated inference or just the reconstruction operator.

[CSV](results/figure_08_reconstruction.csv)

## Figure 9: Other annotation and domain settings

| Evaluation | MineDiffRoad IoU ↑ | Reference method | Reference IoU ↑ | Difference |
| --- | --- | --- | --- | --- |
| Human conservative masks | 67.57 | Concat | 65.82 | +1.75 pp |
| Mine-B, no adaptation | 63.27 | CSL | 62.13 | +1.14 pp |
| RELLIS-3D surface task | 63.49 | CSL | 62.41 | +1.08 pp |
| Natural degradation mean | 64.05 | Conf | 60.72 | +3.33 pp |

Five-seed means. Human conservative masks use a separate batch from the principal mine. Mine-B uses frozen SegMine checkpoints without target adaptation or threshold tuning. RELLIS-3D is independently trained with asphalt, concrete, dirt, mud, and grass as foreground. Natural degradation averages seven conditions equally. Each gain uses its named comparator in its own task, not one baseline across datasets.

[CSV](results/figure_09_generalization.csv)

## Figure 10: Runtime

| Timed scope | Mean (ms) ↓ | P95 (ms) ↓ | FPS ↑ |
| --- | --- | --- | --- |
| Network forward pass | 45.3 | 52.1 | 22.1 |
| Loading, alignment, forward pass, postprocessing | 59.1 | 70.4 | 16.9 |
| With optional graph constraint | 64.8 | 77.6 | 15.4 |

Paper workstation: RTX 4090 24 GB, Intel Core i9-13900K, 64 GB RAM, Ubuntu 22.04, PyTorch 2.1.2, CUDA 12.1, cuDNN 8.9. FP16, batch one, 200 warm-up passes and 500 synchronized measurements. Acquisition/transmission excluded. FPS is reciprocal mean latency. P95 values are scope totals, not additive segments.

[CSV](results/figure_10_runtime.csv)

