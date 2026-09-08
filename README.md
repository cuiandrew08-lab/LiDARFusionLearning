# LiDARFusionLearning

3D object detection for autonomous driving on the [nuScenes](https://www.nuscenes.org/) dataset, comparing a LiDAR-only baseline against a LiDAR-camera fusion model built on a heterogeneous graph.

## Table of Contents

- [Overview](#overview)
- [Status](#status)
- [Architecture](#architecture)
- [About This Repo](#about-this-repo)
- [Results](#results)
- [References](#references)

## Overview

This repository contains two models for 3D object detection, trained on nuScenes:

- **Baseline** - a **PointNet++** backbone encodes BEV(Bird's Eye View) LiDAR data into a 2D feature map, which is fed into a **CenterPoint** head to produce 3D bounding boxes.
- **Fusion (WIP)** - combines point cloud and camera image features via message passing on a **heterogeneous graph**. Fused LiDAR nodes are mapped back into a 3D feature volume, passed through a sparse convolutional backbone to produce a 2D BEV feature map, and then through the same CenterPoint head to produce final 3D bounding boxes.

See [Results](#results) for a comparison between the two models.

## Status

**WIP** The baseline model is complete; the fusion model's architecture is partially implemented and under active development. Evaluation of the models is pending. Details below subject to change.

## Architecture

- **PointNet++** is used as the voxel feature extractor (VFE) for the point cloud, mapping voxels to feature vectors that become the LiDAR nodes of the heterogeneous graph.
- **ResNet-18** (ImageNet-pretrained) extracts features from the camera images, mapping pixel patches to camera nodes.
- **HAN** (Heterogeneous graph Attention Network) performs message passing across the graph.
- **CenterPoint** is then used to find the final 3D bounding boxes.

The following meta-paths are defined for HAN message passing (`L` = LiDAR, `C` = Camera):

- `L-L`
- `C-C`
- `L-C-L`
- `C-L-C`

**Graph construction:**

1. Camera nodes are connected to their spatially adjacent camera-pixel nodes.
2. Each LiDAR node is associated with its voxel's centroid, and nodes whose centroids fall within a radius `r` of one another are connected.
3. For cross-modal edges, LiDAR centroids are projected onto the image plane and connected to their `k` nearest pixel-patch nodes.
4. Top-*k* pruning is applied when building cross-modal meta-paths, to limit the neighbor count of the (much denser) LiDAR nodes.

Message passing then propogates features between LiDAR and camera nodes. Each LiDAR node is mapped back to its centroid and placed into a 3D $(C \times W \times H)$ feature map:

- Cells receiving multiple LiDAR nodes are combined via max-pooling.
- Cells with no LiDAR nodes are zero-filled.

Camera nodes are **not** included directly in the final feature map - their information reaches it only indirectly through message passing into LiDAR nodes.

The 3D feature map is then passed through a sparse convolutional backbone to project it to a 2D BEV feature map, which is then passed through CenterPoint for the final bounding boxes.

## About This Repo

Colab notebooks were used **for development and testing only**; code there may not run as-is. All final model and training code lives in the `.py` files.

## Results

| Model | mAP |
|---|---|
| LiDAR-only baseline (PointNet++ + CenterPoint) | `TODO` |
| LiDAR-camera fusion (HAN) | `TODO` |
