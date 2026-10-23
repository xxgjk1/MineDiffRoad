# MineDiffRoad

## Reliability-Guided Mine Road Segmentation from Conservative Annotations

[中文说明](README.zh-CN.md) · [Paper results](docs/PAPER_RESULTS.md) · [17-figure gallery](docs/FIGURES.md) · [Implementation](docs/IMPLEMENTATION_NOTES.md) · [Validation](docs/SOFTWARE_VALIDATION.md)

Conservative annotations omit some true road. Treating all unannotated pixels as background can penalize recovery. MineDiffRoad uses RGB and measured LiDAR reliability to select local background targets, retain uncertain regions, and recover supported road candidates.

![Figure 3. MineDiffRoad training and inference.](docs/figures/figure_03.png)

## Method

Full four-stage MiT-B0 and LitePT-S feed a **256 × 256 fusion grid**. Input/evaluation masks use **1024 × 1024**, with aspect-preserving resize then padding or cropping. Calibration follows the same transform. Padding is excluded from supervision and evaluation.

Both reliability heads read the joint image, point, and support features. Effective point reliability is zero without point support. Relative weights scale the modalities before **concatenation and projection**, not feature summation.

| Set | Construction | Role |
| --- | --- | --- |
| A_geom | D25(M⁺) minus M⁺, within valid image Ω | Ring directly adjoining the core |
| B_out | Ω minus (M⁺ union A_geom) | Exterior hard background |
| S⁻ | Ring pixels with both teacher probabilities < 0.10 and both reliability scores > 0.50 | Reliability-admitted background |
| U | A_geom minus S⁻ | Provisional unknown |
| Mᵘ | Minimum outer-side contractions of U to satisfy the 30% original-valid-image-area cap, preserving the core-adjacent inner band | Retained unknown |
| F⁻ | U minus Mᵘ | Restored ordinary background |
| M⁻ | B_out union S⁻ union F⁻ | Complete hard-background set |
| Q⁺ | In Mᵘ, both teacher probabilities > 0.90, difference < 0.05, either reliability > 0.50 | Detached mean teacher target |

D25 is a square kernel **width**, not a radius. There is no inner exclusion band. The supervised set is K = M⁺ union M⁻ union Q⁺. The area cap precedes Q⁺ selection and uses original valid image area, not ring area. Fill values outside K are not labels.
The cap removes outer distance layers only; admitted-background holes do not erode the inner edge. If the protected inner band alone exceeds 30%, the implementation stops with an explicit error.

![Figure 2. From conservative annotation to training supervision.](docs/figures/figure_02.png)

Training-only reconstruction uses noisy valid targets and sensor features. Inference retains direct prediction and bounded residual refinement, without reverse diffusion. The residual logit correction is bounded by 0.1.

[Equation map](docs/EQUATION_MAP.md) · [Implementation choices](docs/IMPLEMENTATION_NOTES.md)

## Paper results

Metrics are percentages. IoU pools pixel counts within each seed before averaging over five seeds. MR and EP average valid frames, excluding undefined denominators.

### Full training — Table 1

| Method | IoU ↑ | MR ↑ | EP ↑ |
| --- | ---: | ---: | ---: |
| Concat | 63.58 | 71.42 | 46.93 |
| Conf | 64.94 | 73.68 | 50.87 |
| Reliability admission only | 67.61 | 76.84 | 53.73 |
| MineDiffRoad | **68.57** | **79.13** | **55.12** |
| Complete-label reference | 69.21 | 80.42 | 55.79 |

The complete-label model uses different supervision. The reported gain over Conf is 3.63 points, with paired 95% interval [3.44, 3.82] and crossed scene/seed interval [3.32, 3.94].

### Frozen-teacher replay — Table 2(a)

Teachers, candidate domains, positive candidates, and quota are shared. Every selector retains 60% of the common domain.

| Ranking | FNS ↓ | BR ↑ | IoU ↑ |
| --- | ---: | ---: | ---: |
| Random | 2.13 | 60.03 | 65.44 |
| Confidence | 2.16 | 60.52 | 66.14 |
| MC uncertainty, eight passes | 2.03 | 60.93 | 66.91 |
| Reliability | **1.72** | **61.08** | **68.76** |

FNS audits selected ring negatives against omitted reference road, not the full M⁻ set. BR is selected true background divided by eligible true background. Both average per-image ratios. The 68.76% replay result differs from 68.57% under full online training.

![Figure 7. Fixed-quota selection before unknown-area control.](docs/figures/figure_07.png)

Gray means unselected candidates, not the final unknown mask. Purple is an offline complete-reference diagnosis.

[All result tables and protocol notes](docs/PAPER_RESULTS.md) · [CSV index](docs/results/README.md)

## Segmentation examples

![Figure F.13. SegMine overlays and measured-point projections.](docs/figures/figure_13.png)

![Figure F.17. RELLIS-3D overlays and measured-point projections.](docs/figures/figure_17.png)

Each dataset has four sample groups. Point views project image predictions onto measured returns, not a separate 3D-accuracy evaluation. RELLIS-3D models are trained independently for the paper's surface task. The [figure gallery](docs/FIGURES.md) follows the supplied 40-page manuscript, including its appendix labels.

## Run the source

Use Python 3.10+ and compatible PyTorch. Real LitePT requires CUDA spconv on an NVIDIA GPU. CPU supports software checks.

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python -m minediffroad --help
# Also install compatible CUDA spconv/cumm builds for real training.
```

Use PyTorch ≥2.4 and CUDA-compatible spconv/cumm builds. See [software checks](docs/SOFTWARE_VALIDATION.md).
The manuscript's PyTorch 2.1.2/CUDA 12.1 entry describes the latency-measurement workstation; this source release has not been validated as a byte-identical rebuild of that benchmark environment.

Prepare real paired inputs using [DATA_FORMAT.md](docs/DATA_FORMAT.md). Complete reference labels are prohibited in training records.

The manuscript's SegMine working subset contains **749 paired frames** from 26 scenes and 12 acquisition groups: 524/75/150 train/validation/test frames. These are the repository frames with corresponding point clouds used by the supplied manuscript, not a size claim for every SegMine data product. The optional paper-protocol check verifies these counts against a supplied manifest; ordinary data loading remains manifest-driven.

```bash
python -m minediffroad verify-data --manifest data/manifest.jsonl
python -m minediffroad verify-data --manifest data/manifest.jsonl --paper-segmine
python -m minediffroad verify-data --manifest data/manifest.jsonl --paper-segmine --paper-core-equality --output core_audit.json
python -m minediffroad verify-data --manifest data/manifest.jsonl --natural-degradation-ids data/natural40.json --output natural40_audit.json
python -m minediffroad prepare-fps --manifest data/manifest.jsonl --output data/fps_cache --device cuda
```

[paper.yaml](configs/paper.yaml) contains the default model and training settings. [example.yaml](configs/example.yaml) inherits these defaults. Provide an official full MiT-B0 ImageNet checkpoint before training.

| Default training parameter | Value |
| --- | --- |
| Optimizer | AdamW |
| Learning rate | 1 × 10⁻³ (`0.001`) |
| Weight decay | 1 × 10⁻⁴ (`0.0001`) |
| Learning-rate schedule | Constant (no annealing) |
| Training epochs | 400 |
| Batch size | 2 |

The cosine setting elsewhere in the configuration controls diffusion noise, not the optimizer learning rate.

Conf and Concat configurations inherit the same optimizer settings.

```bash
python -m minediffroad train --config configs/example.yaml --manifest data/fps_cache/manifest.jsonl --output runs/example
# Replace epoch_NNN.pt with the validation-selected best_epoch from checkpoint_index.json.
python -m minediffroad evaluate --checkpoint runs/example/epoch_NNN.pt --manifest data/manifest.jsonl --split test --output runs/example_test
python -m minediffroad predict --checkpoint runs/example/epoch_NNN.pt --manifest data/manifest.jsonl --split test --output runs/example_predictions
python -m minediffroad export --checkpoint runs/example/epoch_NNN.pt --output checkpoints/deployment.pt
python -m minediffroad statistics --reference runs/conf_all_seeds.csv --candidate runs/ours_all_seeds.csv --output runs/paired_intervals.json
python -m unittest discover -s tests -v
```

Outputs must be new. Training saves resolved settings, environment, manifest hash, optimizer/EMA/RNG state, and validation counts. Old first-stage-only checkpoints are incompatible.

The reported Conf control is [conf.yaml](configs/conf.yaml) (teacher road probability below 0.05). A separate [conf_matched.yaml](configs/conf_matched.yaml) uses the same teacher road threshold, 0.10, as the reliability method and requires its own training/evaluation before any result is reported. [concat.yaml](configs/concat.yaml) controls fusion. These configurations inherit the same geometry and area cap.

Statistics require real paired frame counts with seed and scene identifiers. The implementation uses paired seed t intervals and 2,000 crossed scene/seed bootstrap resamples, re-pooling IoU counts. [Statistics contract](docs/STATISTICS.md).

The online trainer, Conf admission, unweighted fusion, fixed-quota ranking, and FNS/BR audit primitives are implemented.

## Provenance

[Source information](docs/SOURCE_SPEC.json) · [Asset hashes](docs/ASSET_PROVENANCE.json) · [Third-party sources](third_party/README.md) · [License](LICENSE)

Old figures are retained in [the historical directory](docs/legacy/20261011/README.md), outside the active gallery and release ZIP.
