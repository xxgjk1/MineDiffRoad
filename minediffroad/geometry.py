"""Calibrated projection, visibility, feature aggregation and deterministic FPS."""
import torch
import torch.nn.functional as F


def resize(x, size, discrete=False):
    return F.interpolate(x.float(), size=size, mode="nearest" if discrete else "bilinear",
                         **({} if discrete else {"align_corners": False}))


@torch.no_grad()
def farthest_points(xyz, count):
    """No random start, no duplicate padding; retain all points if fewer than count."""
    n = len(xyz)
    if n <= count:
        return torch.arange(n, device=xyz.device)
    chosen = torch.empty(count, dtype=torch.long, device=xyz.device)
    distances = torch.full((n,), float("inf"), device=xyz.device)
    farthest = ((xyz - xyz.mean(0)) ** 2).sum(1).argmax()
    for i in range(count):
        chosen[i] = farthest
        distances = torch.minimum(distances, ((xyz - xyz[farthest]) ** 2).sum(1))
        distances[chosen[:i+1]] = -1
        farthest = distances.argmax()
    return chosen


@torch.no_grad()
def project(points, K, T, size, tolerance=0.5):
    """T maps LiDAR metres to camera (x right, y down, z forward).

    K may include the image resize/crop/flip homography. Return visible point
    indices and flattened integer pixel indices. Visibility is resolved BEFORE
    stride-four aggregation, with an inclusive depth tolerance in metres.
    """
    h, w = size
    xyz = points[:, :3].float() @ T[:3, :3].float().T + T[:3, 3].float()
    uvz = xyz @ K.float().T
    uv = uvz[:, :2] / uvz[:, 2:].clamp_min(1e-8)
    xy = torch.floor(uv + 0.5).long()
    valid = (xyz[:, 2] > 1e-6) & torch.isfinite(uv).all(1)
    valid &= (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
    ids = valid.nonzero(as_tuple=True)[0]
    pixels = xy[ids, 1] * w + xy[ids, 0]
    depth = xyz[ids, 2]
    z = depth.new_full((h*w,), float("inf"))
    z.scatter_reduce_(0, pixels, depth, reduce="amin", include_self=True)
    keep = depth <= z[pixels] + tolerance
    return ids[keep], pixels[keep]


def cell_indices(pixels, size, grid):
    h, w = size
    gh, gw = grid
    return (pixels.div(w, rounding_mode="floor") * gh // h) * gw + (pixels % w) * gw // w


def aggregate(features, points, K, T, size, grid, tolerance=0.5):
    ids, pixels = project(points, K, T, size, tolerance)
    cells = cell_indices(pixels, size, grid)
    n, channels = grid[0]*grid[1], features.shape[1]
    counts = features.new_zeros(n)
    counts.index_add_(0, cells, features.new_ones(len(cells)))
    sums = features.new_zeros(n, channels).index_add(0, cells, features[ids])
    maxima = features.new_full((n, channels), -float("inf"))
    maxima = maxima.scatter_reduce(0, cells[:, None].expand(-1, channels), features[ids],
                                   reduce="amax", include_self=True)
    maxima = torch.where(counts[:, None] > 0, maxima, torch.zeros_like(maxima))
    combined = torch.cat([sums / counts.clamp_min(1)[:, None], maxima], dim=1)
    return (combined.T.reshape(1, 2*channels, *grid),
            (counts > 0).reshape(1, 1, *grid), counts.reshape(1, 1, *grid))


@torch.no_grad()
def projected_counts(points, K, T, size, grid, tolerance=0.5):
    _, pixels = project(points, K, T, size, tolerance)
    cells = cell_indices(pixels, size, grid)
    return torch.bincount(cells, minlength=grid[0]*grid[1]).reshape(1, *grid).float()
