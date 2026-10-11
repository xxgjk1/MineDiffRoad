# Third-party provenance

LitePT source: https://github.com/prs-eth/LitePT

Pinned upstream commit: `436d04801c8151faebe66a1b2d368a9711e7e6aa`.

The following upstream files are included under `minediffroad/_vendor/litept/`:

- `litept/model.py` → `model.py`
- `litept/serialization/*.py` → `serialization/*.py`
- `libs/pointrope/pointrope_torch.py` → `pointrope.py`

Upstream MIT license and copyright are retained in `LitePT_LICENSE.txt`. No datasets, checkpoint files, training results, upstream Git history or unrelated framework code are bundled.

Local modifications:

1. Replace optional `torch_scatter.segment_csr` by PyTorch `torch.segment_reduce` over identical CSR segments.
2. Replace FlashAttention's packed variable-length call by per-segment PyTorch scaled dot-product attention with the same Q/K/V, scale and dropout. This can be slower than the dedicated FlashAttention operator; no speed claim is made.
3. Replace the timm DropPath import by a small local implementation. Shapes and stochastic-depth probabilities are unchanged.
4. Use upstream's own pure PyTorch PointROPE implementation; bound its auxiliary trig cache to two entries to avoid unbounded memory growth under augmentation.
5. Move optional `colorhash` visualization dependency into the visualization-only function.
6. Adapter `LitePTEncoder` disables serialization shuffling throughout the model in eval mode. Training keeps upstream shuffling.

Sparse convolution still uses real spconv CUDA operators. The topology, encoder/decoder widths, blocks and learned parameter shapes are preserved. Compatibility operators are validated on synthetic tensors; this does not establish experimental equivalence across all numerical kernels.

Image backbone architecture reference: https://github.com/NVlabs/SegFormer

`backbones.py` independently implements the MiT-B0 stride-4 stage with official checkpoint-compatible parameter names. NVIDIA source files and pretrained weights are not bundled. External pretrained weights remain subject to their original distribution terms.
