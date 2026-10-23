"""Every spatial loss explicitly tracks its domain of valid supervision."""
import torch
import torch.nn.functional as F
from .supervision import dilate, erode


def masked_mean(x, mask):
    return (x * mask).sum() / mask.sum().clamp_min(1)


def balanced_segmentation(logit, target, mask, eps=1e-6):
    target, mask = target.detach().float(), mask.bool()
    losses = []
    for s, y, m in zip(logit.float(), target, mask):
        if not m.any():
            continue
        error = F.binary_cross_entropy_with_logits(s, y, reduction="none")
        parts = [error[c].mean() for c in (m & (y > .5), m & (y <= .5)) if c.any()]
        p = s.sigmoid()
        dice = 1 - (2*(p*y)[m].sum()+eps) / (p[m].sum()+y[m].sum()+eps)
        losses.append(torch.stack(parts).mean()+dice)
    return torch.stack(losses).mean() if losses else logit.sum()*0


def reconstruction_l1(pred, target, positive, negative):
    # Equation: mean of the two set means, with each empty set equal to zero.
    error = (pred.float()-target.detach()).abs()
    parts = []
    for e, p, n in zip(error, positive, negative):
        if (p | n).any():
            parts.append(.5*(masked_mean(e,p)+masked_mean(e,n)))
    return torch.stack(parts).mean() if parts else pred.sum()*0


def skeleton(x, valid, iterations=10):
    """Soft erosion/opening with conservative validity propagation at EVERY op.

    Returned support is the intersection of all iteration supports. This is
    intentionally conservative: narrow admitted regions may have no valid skeleton.
    """
    e = erode(x)
    ve = erode(valid).bool()
    opened, vo = dilate(e), erode(ve).bool()
    sk = F.relu(x-opened)
    support = valid.bool() & vo
    for _ in range(iterations):
        x, valid = e, ve
        e, ve = erode(x), erode(valid).bool()
        opened, vo = dilate(e), erode(ve).bool()
        delta = F.relu(x-opened)
        sk = sk + F.relu(delta-sk*delta)
        support &= valid & vo
    return sk, support


def structural_losses(prob, target, valid, iterations=10, eps=1e-6):
    target, valid = target.detach(), valid.bool()
    boundary_valid = erode(valid).bool()
    pb = (dilate(prob)-erode(prob)).clamp(0,1)
    yb = (dilate(target)-erode(target)).clamp(0,1)
    boundary = masked_mean((pb-yb).abs(), boundary_valid)
    sp, vp = skeleton(prob, valid, iterations)
    sy, vy = skeleton(target, valid, iterations)
    m = vp & vy
    losses = []
    for p,y,s,t,k in zip(prob,target,sp,sy,m):
        if not k.any():
            continue
        precision = ((s*y)[k].sum()+eps)/(s[k].sum()+eps)
        sensitivity = ((t*p)[k].sum()+eps)/(t[k].sum()+eps)
        losses.append(1-2*precision*sensitivity/(precision+sensitivity+eps))
    cldice = torch.stack(losses).mean() if losses else prob.sum()*0
    return boundary, cldice


def total_loss(output, states, quality, reconstruction, weights, skeleton_iterations=10,
               confident_threshold=.9):
    s, p = output["logit"].float(), output["prob"].float()
    y, valid = states.target, states.valid
    terms = {"seg": balanced_segmentation(s,y,states.positive | states.negative)}
    terms["rel"] = sum(masked_mean(F.binary_cross_entropy_with_logits(
        output[f"quality_logit_{m}"].float(), quality[f"q_{m}"], reduction="none"),
        quality[f"valid_{m}"]) for m in ("v","a"))
    terms["b"], terms["s"] = structural_losses(p,y,valid,skeleton_iterations)
    pb = output["base_prob"].detach().float()
    confident = ((pb >= confident_threshold) | (pb <= 1-confident_threshold)) & valid
    terms["c"] = masked_mean((p-pb).square(), confident)
    terms["delta"] = masked_mean(output["delta"].float().square(), valid)
    if reconstruction is not None and states.candidates.any():
        terms["rec"] = balanced_segmentation(s,reconstruction["x0"].detach(),valid)
        terms["diff"] = masked_mean((reconstruction["noise_pred"].float()-reconstruction["noise"]).square(),valid)
        terms["x0"] = reconstruction_l1(reconstruction["x0"],y,
                                         states.positive | states.candidates,states.negative)
    else:
        terms.update({k:s.sum()*0 for k in ("rec","diff","x0")})
    total = sum(weights[k]*v for k,v in terms.items())
    return total, terms
