# Figures from the paper

[Back to README](../README.md)

Figures 1–8 are reproduced from `MineDiffRoad_20261011_UpdatedFigures_English.pdf`. Complete panel groups, in-panel labels, and legends are retained. Surrounding page text is excluded; captions below summarize the original captions. These assets were rendered from the supplied PDF, without regenerating predictions or altering masks.

The source PDF and LaTeX hashes, page numbers, crop coordinates, image sizes, and image hashes are recorded in [ASSET_PROVENANCE.json](ASSET_PROVENANCE.json). The full manuscript is not bundled. Underlying dataset imagery retains its original attribution and distribution terms.

## Figure 1: Problem: conservative road annotations

![Problem: conservative road annotations](figures/problem.png)

RGB, conservative core, omitted road, and complete reference in one scene. White pixels in the omitted-road panel mark road missing from the core. Complete references support evaluation and the offline illustration, not partial-label training.

Source: PDF page 2.

## Figure 2: Method overview

![Method overview](figures/architecture.png)

Multimodal encoding, reliability-guided fusion, deterministic prediction, and the training-only supervision/reconstruction branch. Background admission requires both modalities to be reliable; positive recovery accepts either modality. The base prediction is refined with a residual bounded by 0.1 in logit space.

Source: PDF page 6.

## Figure 3: Supervision states

![Supervision states](figures/supervision_states.png)

The geometric ring, stable background domain, trusted positives, admitted negatives, and unknown region. Complete labels are used only in the final panel for the offline error audit. The implementation consumes a supplied valid mask; it does not derive sky exclusion from complete training labels.

Source: PDF page 9.

## Figure 4: Recovery of omitted road

![Recovery of omitted road](figures/road_recovery.png)

Full views and shared crops compare RGB, reference, Concat, Conf, MineDiffRoad, and the complete-label model. Deep green denotes trusted/correct road; light green recovered road; amber missed road; purple false expansion. The paper specifies seed 0, validation-selected checkpoints, threshold 0.50, and common postprocessing.

Source: PDF page 16.

## Figure 5: Comparison with CSL: samples 1–4

![Comparison with CSL: samples 1–4](figures/comparison_samples_1_4.png)

Columns show RGB, complete reference, CSL with SegFormer-B5, and MineDiffRoad. Deep green denotes road. Both methods use the same spatial extent and rendering convention.

Source: PDF page 17.

## Figure 6: Comparison with CSL: samples 5–8

![Comparison with CSL: samples 5–8](figures/comparison_samples_5_8.png)

Four additional samples using the same column order and rendering as Figure 5. Complete references are used for evaluation and display.

Source: PDF page 18.

## Figure 7: Negative selection at a fixed quota

![Negative selection at a fixed quota](figures/negative_selection.png)

The common candidate domain and random, confidence, and reliability selections, followed by a shared crop. Each selector chooses 409 of 682 candidates. Purple identifies omitted road admitted as negative, based on an offline reference audit.

Source: PDF page 21.

## Figure 8: Failure case

![Failure case](figures/failure_case.png)

False expansion into terrain resembling road. The full-image ROI is followed by the reference, Concat, Conf, MineDiffRoad, and complete-label model crops. Colors follow Figure 4; observation quality alone does not settle semantic ambiguity.

Source: PDF page 29.

