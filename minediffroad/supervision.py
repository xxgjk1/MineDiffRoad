from dataclasses import dataclass
import torch
import torch.nn.functional as F


def dilate(x, kernel=3):
    return F.max_pool2d(x.float(), kernel, stride=1, padding=kernel//2)


def erode(x, kernel=3):
    # Explicit zero padding: neighborhoods extending outside the image are invalid.
    p = kernel//2
    return -F.max_pool2d(-F.pad(x.float(), (p,p,p,p), value=0), kernel, stride=1)


@dataclass
class States:
    positive: torch.Tensor
    ring: torch.Tensor
    stable_background: torch.Tensor
    negative: torch.Tensor
    unknown: torch.Tensor
    candidates: torch.Tensor
    valid: torch.Tensor
    target: torch.Tensor


@torch.no_grad()
def build_states(core, omega, rv, ra, teacher1=None, teacher2=None, warmup=False,
                 outer=25, inner=11, tau_n=0.1, tau_r=0.5,
                 tau_p=0.9, delta=0.05, tau_q=0.5):
    if outer <= inner or outer % 2 != 1 or inner % 2 != 1:
        raise ValueError("Ring kernels must be odd widths with outer > inner")
    omega = omega.bool()
    positive = core.bool() & omega
    ring = dilate(positive, outer).bool() & ~dilate(positive, inner).bool() & ~positive & omega
    if warmup:
        stable = ring
        negative = ring
        candidates = torch.zeros_like(positive)
    else:
        if teacher1 is None or teacher2 is None:
            raise ValueError("Two aligned EMA teacher maps are required after warmup")
        stable = ring & (torch.maximum(teacher1, teacher2) < tau_n)
        negative = stable & (rv > tau_r) & (ra > tau_r)
        unknown = omega & ~(positive | negative)
        candidates = unknown & (torch.minimum(teacher1, teacher2) > tau_p)
        candidates &= (teacher1-teacher2).abs() < delta
        candidates &= torch.maximum(rv, ra) > tau_q
    unknown = omega & ~(positive | negative)
    valid = positive | negative | candidates
    target = torch.full_like(rv, 0.5)
    target[positive] = 1.0
    target[negative] = 0.0
    if not warmup:
        target[candidates] = ((teacher1+teacher2)/2)[candidates]
    return States(positive, ring, stable, negative, unknown, candidates, valid, target)


@torch.no_grad()
def fixed_quota_selection(domain, score, fraction=0.6):
    """Per-image ranking; stable sorting gives row/column tie-breaking."""
    if not 0 <= fraction <= 1:
        raise ValueError("fraction must be in [0,1]")
    selected = torch.zeros_like(domain, dtype=torch.bool)
    for b in range(len(domain)):
        indices = domain[b].flatten().nonzero(as_tuple=True)[0]
        order = torch.argsort(score[b].flatten()[indices], descending=True, stable=True)
        selected[b].view(-1)[indices[order[:int(len(indices)*fraction)]]] = True
    return selected
