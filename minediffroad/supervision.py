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
    exterior: torch.Tensor
    admitted: torch.Tensor
    provisional_unknown: torch.Tensor
    restored: torch.Tensor
    erosion_steps: torch.Tensor


@torch.no_grad()
def cap_unknown(unknown, omega, core, outer=25, fraction=.30, kernel=3,
                area_weights=None, original_valid_area=None):
    """Peel the core-centred ring from its outer edge, per image.

    Each weight counts original valid pixel centres mapped to a working pixel.
    Cropped-out original pixels remain in the denominator; padding has no weight.
    One step removes a (kernel-1)/2-pixel outer band in Chebyshev distance;
    the core-adjacent first band is never removed. An infeasible cap is an error.
    """
    if not 0 <= fraction <= 1 or kernel < 3 or kernel % 2 != 1 or outer < 3 or outer % 2 != 1:
        raise ValueError("Area cap must be in [0,1]; ring and erosion kernels must be odd and >=3")
    if unknown.shape != omega.shape or core.shape != omega.shape or unknown.ndim != 4:
        raise ValueError("Unknown, valid domain, and core must have equal NCHW shapes")
    core = core.bool() & omega.bool()
    ring = dilate(core, outer).bool() & ~core & omega.bool()
    if (unknown.bool() & ~ring).any():
        raise ValueError("Unknown pixels must lie in the core-centred ring")
    if (area_weights is None) != (original_valid_area is None):
        raise ValueError("area_weights and original_valid_area must be supplied together")
    weights = omega.float() if area_weights is None else area_weights.to(unknown.device).float()
    if weights.shape != unknown.shape or not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Invalid original-area weights")
    total = weights.flatten(1).sum(1) if original_valid_area is None else torch.as_tensor(
        original_valid_area, device=unknown.device).reshape(-1).float()
    if total.shape != (len(unknown),) or not torch.isfinite(total).all() or (total <= 0).any():
        raise ValueError("Every image needs positive original valid area")
    if (weights.flatten(1).sum(1) > total + .5).any():
        raise ValueError("Area weights exceed the original valid image area")
    retained = unknown.bool() & omega.bool()
    steps = torch.zeros(len(unknown), device=unknown.device, dtype=torch.long)
    radius = (outer-1)//2
    radius_step = (kernel-1)//2
    active = (retained * weights).flatten(1).sum(1) > fraction * total
    if not active.any():
        return retained, steps
    # Build the core-distance layers once using small pooling kernels.
    distance = torch.full(core.shape, radius+1, device=core.device, dtype=torch.int32)
    reached = core
    for layer in range(1, radius+1):
        expanded = dilate(reached, 3).bool()
        distance[expanded & ~reached] = layer
        reached = expanded
    while True:
        active = (retained * weights).flatten(1).sum(1) > fraction * total
        if not active.any():
            return retained, steps
        radius -= radius_step
        if radius < 1:
            raise ValueError("Unknown-area cap cannot be met while preserving the core-adjacent band")
        retained = torch.where(active[:, None, None, None], retained & (distance <= radius), retained)
        steps += active.long()


@torch.no_grad()
def build_states(core, omega, rv, ra, teacher1=None, teacher2=None, warmup=False,
                 outer=25, tau_n=0.1, tau_r=0.5,
                 tau_p=0.9, delta=0.05, tau_q=0.5,
                 unknown_area_cap=.30, erosion_kernel=3,
                 area_weights=None, original_valid_area=None,
                 negative_rule="reliability", confidence_threshold=.95, point_support=None):
    if outer < 1 or outer % 2 != 1:
        raise ValueError("Ring dilation kernel must be a positive odd width")
    if negative_rule not in {"reliability", "confidence", "confidence_matched"}:
        raise ValueError("negative_rule must be reliability, confidence, or confidence_matched")
    omega = omega.bool()
    positive = core.bool() & omega
    ring = dilate(positive, outer).bool() & ~positive & omega
    exterior = omega & ~(positive | ring)
    if warmup:
        stable = ring
        admitted = ring
    else:
        if teacher1 is None or teacher2 is None:
            raise ValueError("Two aligned EMA teacher maps are required after warmup")
        stable = ring & (torch.maximum(teacher1, teacher2) < tau_n)
        if negative_rule == "reliability":
            admitted = stable & (rv > tau_r) & (ra > tau_r)
        elif negative_rule == "confidence":
            if point_support is None:
                raise ValueError("Conf admission requires an explicit point-support mask")
            admitted = ring & point_support.bool() & (1-torch.maximum(teacher1, teacher2) > confidence_threshold)
        else:
            if point_support is None:
                raise ValueError("Matched Conf admission requires an explicit point-support mask")
            admitted = stable & point_support.bool()
    provisional = ring & ~admitted
    unknown, steps = cap_unknown(provisional, omega, positive, outer, unknown_area_cap,
                                 erosion_kernel, area_weights, original_valid_area)
    restored = provisional & ~unknown
    negative = exterior | admitted | restored
    candidates = torch.zeros_like(positive)
    if not warmup:
        candidates = unknown & (torch.minimum(teacher1, teacher2) > tau_p)
        candidates &= (teacher1-teacher2).abs() < delta
        candidates &= torch.maximum(rv, ra) > tau_q
    valid = positive | negative | candidates
    target = torch.full_like(rv, 0.5)
    target[positive] = 1.0
    target[negative] = 0.0
    if not warmup:
        target[candidates] = ((teacher1+teacher2)/2)[candidates]
    return States(positive, ring, stable, negative, unknown, candidates, valid, target,
                  exterior, admitted, provisional, restored, steps)


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
