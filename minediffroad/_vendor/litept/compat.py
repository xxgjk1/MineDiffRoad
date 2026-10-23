"""Native PyTorch equivalents for optional LitePT operator dependencies.

No architecture or parameter shapes are changed. Sparse convolution still uses
spconv and requires CUDA. SDPA can select PyTorch's fused CUDA implementation.
"""
import torch
from torch import nn
from torch.nn import functional as F


def segment_csr(src, indptr, reduce="sum"):
    return torch.segment_reduce(src, reduce, lengths=indptr[1:]-indptr[:-1])


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0:
            return x
        keep = 1-self.drop_prob
        mask = x.new_empty((len(x),)+(1,)*(x.ndim-1)).bernoulli_(keep)
        return x*mask/keep


def flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens, max_seqlen,
                                   dropout_p=0., softmax_scale=None):
    outputs = []
    bounds = cu_seqlens.tolist()
    for start, end in zip(bounds[:-1], bounds[1:]):
        q, k, v = qkv[start:end].unbind(1)
        out = F.scaled_dot_product_attention(
            q.transpose(0,1).unsqueeze(0), k.transpose(0,1).unsqueeze(0),
            v.transpose(0,1).unsqueeze(0), dropout_p=dropout_p, scale=softmax_scale)
        outputs.append(out.squeeze(0).transpose(0,1))
    return torch.cat(outputs, dim=0)
