This repository contains a geometry-aware pipeline to verify whether a single query image of the Brandenburg Gate (real or AI-generated) is consistent with a physically plausible camera viewpoint of a real 3D scene.

Core workflow:
1) Train a 3D Gaussian Splatting (3DGS) scene from real calibrated images.
2) Estimate a 6-DoF camera pose for a query image using 6DGS.
3) Render the 3DGS scene from the predicted pose (optionally with small yaw variants).
4) Compare render vs query using feature alignment and pixel/gradient error metrics (SIFT + RANSAC homography + heatmaps).

Notes:
- Designed to run on a Linux HPC environment (conda env: v_6dgs).
- Large assets (datasets, trained models, renders) are not tracked in git; paths are provided via command-line flags.