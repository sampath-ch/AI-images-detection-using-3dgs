# Brandenburg Gate Geometry Verification (3DGS + 6DGS)

This repository contains a geometry-aware pipeline to verify whether a single query image of the Brandenburg Gate (real or AI-generated) is consistent with a physically plausible camera viewpoint of a real 3D scene.

## Core workflow
1) Train a 3D Gaussian Splatting (3DGS) scene from real calibrated images (COLMAP-style).
2) Estimate a 6-DoF camera pose for a query image using 6DGS.
3) Render the 3DGS scene from the predicted pose (optionally with small yaw variants).
4) Compare render vs query using feature alignment and pixel/gradient error metrics (SIFT + RANSAC homography + heatmaps).

## Notes
- Designed to run on a Linux HPC environment (conda env: `v_6dgs`).
- Large assets (datasets, trained models, renders) are not tracked in git; paths are provided via command-line flags.
- Do **not** commit `.ply` / checkpoints / datasets to git (GitHub blocks >100MB files).

---

# Phase 1: 3DGS Scene Training (Brandenburg Gate)

**Goal:** Train a 3DGS scene from real calibrated images. The trained 3DGS model becomes the ground-truth 3D prior for everything downstream.

**Training guide (my notes):**
- Phase 1 Training Guide: https://drive.google.com/file/d/19GVurOG6A5L6prPUVwE3A-BjBDNYEOUh/view?usp=sharing

## What to run (in the official 3DGS repo)

In the official `graphdeco-inria/gaussian-splatting` codebase, the optimizer is run via:
- `train.py` with `--source_path/-s` (dataset root) and optionally `--model_path/-m` (output dir). The default training length is `--iterations 30000`.  

Reference usage from the upstream README:
- `python train.py -s <path to COLMAP or NeRF Synthetic dataset>`
- `--source_path / -s`, `--model_path / -m`, `--images / -i`, `--iterations (30_000 default)`

## Example (matches our Brandenburg Gate layout)

From inside the 3DGS repo:

```bash
python train.py \
  -s /scratch/schettip/gaussian-splatting/data/brandenburg_gate \
  -m /scratch/schettip/gaussian-splatting/output/5db67e12-1 \
  --iterations 30000
