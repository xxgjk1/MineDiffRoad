# Paper figure gallery

[Back to README](../README.md)

These 17 images are exact crops from the supplied 40-page English PDF. They retain the PDF's panel labels, color legends, and captions. Main-text figures are numbered 1–10; the remaining labels are E.11 and F.12–F.17. The [figure manifest](figures/manifest.json) records source pages, crop rectangles, and image hashes. Cropping does not regenerate masks, predictions, or plotted data.

## Figure 1: Conservative annotation

![Figure 1: Conservative annotation](figures/figure_01.png)

PDF page 4.

## Figure 2: From conservative annotation to training supervision

![Figure 2: Supervision mask generation](figures/figure_02.png)

PDF page 8. The diagram distinguishes the ring, admitted background, retained unknown set, and soft road evidence.

## Figure 3: MineDiffRoad training and inference

![Figure 3: Training and inference](figures/figure_03.png)

PDF page 9. Training-only branches are marked separately from inference.

## Figure 4: Local supervision changes

![Figure 4: Local supervision changes](figures/figure_04.png)

PDF page 11. The lower row shows the corresponding wheel-rut crop.

## Figure 5: Road recovery and boundary expansion

![Figure 5: SegMine road recovery](figures/figure_05.png)

PDF page 16. Rows show the full image and shared crops A and B. This is the supplied PDF rendering; no pixel-level correction to its original masks or overlay was made in this repository.

## Figure 6: Confidence and reliability diagnostics

![Figure 6: Confidence and reliability](figures/figure_06.png)

PDF page 19. Candidate counts, omitted-road fraction, and selected fractions are shown on the shared score domain.

## Figure 7: Fixed-quota negative selection

![Figure 7: Negative selection](figures/figure_07.png)

PDF page 20. Gray denotes unselected candidates before unknown-area control, not the final unknown mask. Purple uses complete-reference labels for offline diagnosis only.

## Figure 8: Reconstruction benefit and training cost

![Figure 8: Reconstruction variants](figures/figure_08.png)

PDF page 21. Standard IoU, degraded-condition mean IoU, and full training time per epoch are reported separately.

## Figure 9: Other annotation and domain settings

![Figure 9: Gains across settings](figures/figure_09.png)

PDF page 22. Each gain is relative to the named comparator in its own setting.

## Figure 10: Latency and throughput

![Figure 10: Runtime](figures/figure_10.png)

PDF page 24. P95 markers are total scope latencies, not additive segments.

## Figure E.11: SegMine RGB and measured point-cloud inputs

![Figure E.11: SegMine paired inputs](figures/figure_11.png)

PDF page 31. Points are colored by range.

## Figure F.12: Validation trajectories

![Figure F.12: Validation trajectories](figures/figure_12.png)

PDF page 33. Thin curves represent five seeds; thick curves are their means.

## Figure F.13: SegMine segmentation examples

![Figure F.13: SegMine segmentation examples](figures/figure_13.png)

PDF page 34. Four groups pair image overlays with measured-point projections.

## Figure F.14: Left-edge recovery ROI

![Figure F.14: Left-edge recovery ROI](figures/figure_14.png)

PDF page 34. The lower row enlarges the boxed region in each full image.

## Figure F.15: Quota and false negative supervision

![Figure F.15: Selection quota and FNS](figures/figure_15.png)

PDF page 35. At 60% quota, mean FNS is 2.13% for random, 2.16% for confidence, and 1.72% for reliability selection.

## Figure F.16: Scene-wise gains

![Figure F.16: Scene-wise IoU gains](figures/figure_16.png)

PDF page 36. Panels separate open terrain, complex boundaries, and degraded observations.

## Figure F.17: RELLIS-3D segmentation examples

![Figure F.17: RELLIS-3D segmentation examples](figures/figure_17.png)

PDF page 37. Four groups pair image overlays with measured LiDAR returns; these are independently trained surface-task models, not SegMine zero-shot transfer.

Point-cloud panels visualize image predictions on measured returns, not independent 3D accuracy.
