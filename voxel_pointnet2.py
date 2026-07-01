"""
voxel_pointnet2.py

A standard PointNet++ "Set Abstraction" feature learner, applied per-voxel
to LiDAR point clouds that have already been voxelized into fixed-capacity
buckets -- the same input format produced by spconv / mmdet3d / OpenPCDet
style voxelization: a [num_voxels, max_points_per_voxel, C] tensor plus a
[num_voxels] count of how many points are actually valid in each voxel.

This is a pure-PyTorch re-implementation of the core PointNet++ building
blocks (farthest point sampling, ball-query grouping, and the shared-MLP +
max-pool "Set Abstraction" layer), extended with padding-mask support so
the variable number of real points per voxel doesn't corrupt sampling,
grouping, or pooling.

Use it wherever you currently turn "points in a voxel" into "one feature
vector per voxel" -- e.g. right before a BEV scatter step, or as the
node-feature extractor for the LiDAR side of a heterogeneous graph. It is a
strictly richer alternative to a simple max-pool VFE (VoxelNet/PointPillars
style), since it captures multi-scale local geometry within each voxel
instead of collapsing all points through a single shared MLP + max-pool.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Core PointNet++ ops (pure PyTorch, no custom CUDA kernels required)
# --------------------------------------------------------------------------- #

def square_distance(src, dst):
    """Pairwise squared Euclidean distance.
    src: [B, N, C], dst: [B, M, C] -> [B, N, M]
    """
    return torch.sum((src[:, :, None, :] - dst[:, None, :, :]) ** 2, dim=-1)


def index_points(points, idx):
    """Gather points by index.
    points: [B, N, C]
    idx:    [B, S] or [B, S, K]
    -> [B, S, C] or [B, S, K, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
    return points[batch_indices, idx, :]


def farthest_point_sample(xyz, npoint, valid_mask=None):
    """Iterative farthest point sampling.
    xyz: [B, N, 3]
    valid_mask: optional [B, N] bool, True = real point, False = padding.
                Padded points are never selected as centroids.
    -> [B, npoint] long indices
    """
    device = xyz.device
    B, N, _ = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.full((B, N), 1e10, device=device)

    if valid_mask is not None:
        init_scores = torch.rand(B, N, device=device)
        init_scores = init_scores.masked_fill(~valid_mask, -1.0)
        farthest = torch.max(init_scores, dim=-1)[1]
    else:
        farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)

    batch_indices = torch.arange(B, dtype=torch.long, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        distance = torch.min(distance, dist)
        if valid_mask is not None:
            select_from = distance.masked_fill(~valid_mask, -1.0)
        else:
            select_from = distance
        farthest = torch.max(select_from, dim=-1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz, valid_mask=None):
    """Group up to `nsample` neighbors of each query point within `radius`.
    xyz: [B, N, 3], new_xyz: [B, S, 3]
    valid_mask: optional [B, N] bool over the source points being queried.
    -> [B, S, nsample] long indices into the N source points.
    """
    B, N, _ = xyz.shape
    S = new_xyz.shape[1]
    sqrdists = square_distance(new_xyz, xyz)  # [B, S, N]
    if valid_mask is not None:
        sqrdists = sqrdists.masked_fill(~valid_mask[:, None, :], 1e10)
    group_idx = torch.arange(N, dtype=torch.long, device=xyz.device).view(1, 1, N).repeat(B, S, 1)
    group_idx = group_idx.masked_fill(sqrdists > radius ** 2, N)  # N = "out of range" sentinel
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat(1, 1, nsample)
    empty_mask = group_idx == N
    group_idx = torch.where(empty_mask, group_first, group_idx)
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points, valid_mask=None):
    """FPS -> ball query -> group local neighborhoods (relative to their centroid).
    xyz: [B, N, 3], points: [B, N, D] or None
    Returns:
      new_xyz:            [B, npoint, 3]
      new_points:         [B, npoint, nsample, 3 (+D)]
      grouped_valid_mask: [B, npoint, nsample] bool, or None
      center_valid_mask:  [B, npoint] bool, or None
    """
    B, N, C = xyz.shape
    fps_idx = farthest_point_sample(xyz, npoint, valid_mask)
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz, valid_mask)
    grouped_xyz = index_points(xyz, idx)
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, npoint, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz_norm

    grouped_valid_mask = None
    center_valid_mask = None
    if valid_mask is not None:
        vm = valid_mask.float().unsqueeze(-1)
        grouped_valid_mask = index_points(vm, idx).squeeze(-1).bool()
        center_valid_mask = index_points(vm, fps_idx).squeeze(-1).bool()
    return new_xyz, new_points, grouped_valid_mask, center_valid_mask


def sample_and_group_all(xyz, points, valid_mask=None):
    """Group ALL points into a single global region (used for the final SA layer).
    -> new_xyz: [B, 1, 3] (zeros), new_points: [B, 1, N, 3(+D)], grouped_valid_mask: [B, 1, N] or None
    """
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C, device=xyz.device, dtype=xyz.dtype)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    grouped_valid_mask = valid_mask.view(B, 1, N) if valid_mask is not None else None
    return new_xyz, new_points, grouped_valid_mask


# --------------------------------------------------------------------------- #
# Set Abstraction layers
# --------------------------------------------------------------------------- #

class PointNetSetAbstraction(nn.Module):
    """Single-scale grouping (SSG) Set Abstraction layer -- the core
    PointNet++ block: sample centroids, group local neighborhoods, encode
    each neighborhood with a shared MLP, max-pool over the neighborhood.
    """

    def __init__(self, npoint, radius, nsample, in_channel, mlp_channels, group_all=False):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp_channels:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz, points, valid_mask=None):
        """
        xyz:    [B, N, 3]
        points: [B, N, D] or None
        valid_mask: [B, N] bool or None
        Returns: new_xyz [B, S, 3], new_points [B, S, C_out], center_valid_mask [B, S] or None
        """
        if self.group_all:
            new_xyz, new_points, grouped_valid_mask = sample_and_group_all(xyz, points, valid_mask)
            center_valid_mask = None
        else:
            new_xyz, new_points, grouped_valid_mask, center_valid_mask = sample_and_group(
                self.npoint, self.radius, self.nsample, xyz, points, valid_mask)

        # [B, S, K, C] -> [B, C, S, K] for Conv2d
        new_points = new_points.permute(0, 3, 1, 2)
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            new_points = F.relu(bn(conv(new_points)))

        if grouped_valid_mask is not None:
            new_points = new_points.masked_fill(~grouped_valid_mask.unsqueeze(1), float('-inf'))

        new_points = torch.max(new_points, dim=-1)[0]  # [B, C_out, S]
        new_points = torch.nan_to_num(new_points, neginf=0.0)  # guard fully-empty groups
        new_points = new_points.permute(0, 2, 1)  # [B, S, C_out]
        return new_xyz, new_points, center_valid_mask


class PointNetSetAbstractionMsg(nn.Module):
    """Multi-scale grouping (MSG) Set Abstraction layer: groups each centroid's
    neighborhood at several (radius, nsample) scales in parallel and
    concatenates the resulting features. Useful when point density varies a
    lot within a voxel (sparse vs. dense LiDAR returns).
    """

    def __init__(self, npoint, radii, nsamples, in_channel, mlp_channels_list):
        super().__init__()
        assert len(radii) == len(nsamples) == len(mlp_channels_list)
        self.npoint = npoint
        self.radii = radii
        self.nsamples = nsamples
        self.branches = nn.ModuleList()
        for mlp_channels in mlp_channels_list:
            convs, bns = nn.ModuleList(), nn.ModuleList()
            last_channel = in_channel
            for out_channel in mlp_channels:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.branches.append(nn.ModuleDict({"convs": convs, "bns": bns}))

    def forward(self, xyz, points, valid_mask=None):
        B, N, C = xyz.shape
        fps_idx = farthest_point_sample(xyz, self.npoint, valid_mask)
        new_xyz = index_points(xyz, fps_idx)

        center_valid_mask = None
        if valid_mask is not None:
            center_valid_mask = index_points(valid_mask.float().unsqueeze(-1), fps_idx).squeeze(-1).bool()

        branch_outputs = []
        for radius, nsample, branch in zip(self.radii, self.nsamples, self.branches):
            idx = query_ball_point(radius, nsample, xyz, new_xyz, valid_mask)
            grouped_xyz = index_points(xyz, idx) - new_xyz.view(B, self.npoint, 1, C)
            if points is not None:
                grouped_points = torch.cat([grouped_xyz, index_points(points, idx)], dim=-1)
            else:
                grouped_points = grouped_xyz

            grouped_valid_mask = None
            if valid_mask is not None:
                grouped_valid_mask = index_points(valid_mask.float().unsqueeze(-1), idx).squeeze(-1).bool()

            feat = grouped_points.permute(0, 3, 1, 2)  # [B, C, S, K]
            for conv, bn in zip(branch["convs"], branch["bns"]):
                feat = F.relu(bn(conv(feat)))
            if grouped_valid_mask is not None:
                feat = feat.masked_fill(~grouped_valid_mask.unsqueeze(1), float('-inf'))
            feat = torch.max(feat, dim=-1)[0]
            feat = torch.nan_to_num(feat, neginf=0.0)
            branch_outputs.append(feat)

        new_points = torch.cat(branch_outputs, dim=1).permute(0, 2, 1)  # [B, S, sum(C_out)]
        return new_xyz, new_points, center_valid_mask


# --------------------------------------------------------------------------- #
# Per-voxel wrapper
# --------------------------------------------------------------------------- #

def build_valid_mask(num_points, max_points):
    """num_points: [V] long -> valid_mask: [V, max_points] bool"""
    ar = torch.arange(max_points, device=num_points.device).unsqueeze(0)
    return ar < num_points.unsqueeze(1)


class VoxelPointNetEncoder(nn.Module):
    """Standard PointNet++ Set Abstraction stack, applied independently to
    every voxel in a batch, to produce one learned feature vector per voxel.

    Input format matches standard voxelization output (spconv / mmdet3d /
    OpenPCDet style):
        voxel_points: [V, P, C]  (V voxels, up to P points each,
                                   C = 3 (xyz) + extra features e.g. intensity)
        num_points:   [V]        (how many of the P slots are real; the rest
                                   is assumed zero-padded, as voxel generators
                                   already produce)

    Output:
        voxel_features: [V, out_dim]

    Assumes every voxel passed in has at least 1 valid point (filter out
    empty voxels upstream, same as you'd do for any VFE).
    """

    def __init__(self, extra_feature_dim=1, center_xyz=True,
                 sa1_npoint=16, sa1_radius=0.6, sa1_nsample=16, sa1_mlp=(32, 64),
                 sa2_mlp=(64, 128, 256)):
        super().__init__()
        self.center_xyz = center_xyz
        sa1_mlp = list(sa1_mlp)
        sa2_mlp = list(sa2_mlp)
        self.sa1 = PointNetSetAbstraction(
            npoint=sa1_npoint, radius=sa1_radius, nsample=sa1_nsample,
            in_channel=3 + extra_feature_dim, mlp_channels=sa1_mlp, group_all=False)
        self.sa2 = PointNetSetAbstraction(
            npoint=None, radius=None, nsample=None,
            in_channel=3 + sa1_mlp[-1], mlp_channels=sa2_mlp, group_all=True)
        self.out_dim = sa2_mlp[-1]

    def forward(self, voxel_points, num_points):
        V, P, C = voxel_points.shape
        valid_mask = build_valid_mask(num_points, P)

        xyz = voxel_points[..., :3].clone()
        feats = voxel_points[..., 3:] if C > 3 else None

        if self.center_xyz:
            # Center each voxel's points on the mean of its OWN valid points,
            # so the encoder sees translation-invariant local geometry
            # regardless of where the voxel sits in the scene.
            sums = (xyz * valid_mask.unsqueeze(-1)).sum(dim=1)
            centers = sums / num_points.clamp(min=1).unsqueeze(-1).to(xyz.dtype)
            xyz = (xyz - centers.unsqueeze(1)) * valid_mask.unsqueeze(-1)

        l1_xyz, l1_points, l1_valid = self.sa1(xyz, feats, valid_mask)
        _, l2_points, _ = self.sa2(l1_xyz, l1_points, l1_valid)
        return l2_points.squeeze(1)  # [V, out_dim]


# --------------------------------------------------------------------------- #
# Sanity check
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    torch.manual_seed(0)
    V, P = 64, 32            # 64 voxels, capacity 32 points each
    extra_dim = 1            # e.g. intensity
    num_points = torch.randint(1, P + 1, (V,))  # each voxel has between 1 and P real points

    voxel_points = torch.randn(V, P, 3 + extra_dim)
    mask = build_valid_mask(num_points, P)
    voxel_points = voxel_points * mask.unsqueeze(-1)  # zero out padding, as real preprocessing would

    encoder = VoxelPointNetEncoder(extra_feature_dim=extra_dim)
    out = encoder(voxel_points, num_points)
    print("output shape:", out.shape)
    assert out.shape == (V, encoder.out_dim)
    assert torch.isfinite(out).all()

    out.sum().backward()
    grad_ok = all(p.grad is not None and torch.isfinite(p.grad).all() for p in encoder.parameters())
    print("gradients finite:", grad_ok)
    assert grad_ok

    # Edge case: a voxel with only 1 valid point
    num_points[0] = 1
    voxel_points2 = torch.randn(V, P, 3 + extra_dim)
    mask2 = build_valid_mask(num_points, P)
    voxel_points2 = voxel_points2 * mask2.unsqueeze(-1)
    out2 = encoder(voxel_points2, num_points)
    assert torch.isfinite(out2).all()
    print("single-point-voxel edge case OK")

    # Edge case: every voxel has exactly P points (no padding at all)
    num_points_full = torch.full((V,), P, dtype=torch.long)
    voxel_points3 = torch.randn(V, P, 3 + extra_dim)
    out3 = encoder(voxel_points3, num_points_full)
    assert torch.isfinite(out3).all()
    print("fully-dense-voxel edge case OK")

    print("All sanity checks passed.")
