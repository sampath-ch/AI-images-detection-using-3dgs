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

**Training guide:**
- Phase 1 Training .ply file: https://drive.google.com/file/d/19GVurOG6A5L6prPUVwE3A-BjBDNYEOUh/view?usp=sharing

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
```
---

## Baseline Localization Experiments (HLoc)

*After training the 3DGS scene, we attempted to establish a baseline for 6-DoF pose estimation using HLoc. The following scripts document these initial attempts and the technical challenges (specifically feature mismatches) that led us to the 6DGS-consistent methods used in Phase 2.*

### 1. Pose Attempt with HLoc
**Script:** `find_pose.py`
* **Goal:** Estimate a 6-DoF pose for a query image using HLoc (SuperPoint + SuperGlue) against the COLMAP model.
* **Output:** `localization_results.txt` (pose) and `queries.txt` (intrinsics).
* **Outcome:** Produced a pose, but required "clipping" hacks to run and ultimately proved unreliable for this specific COLMAP model setup.

### 2. Rendering Attempts (Diagnosis)
**Attempt A: Raw HLoc Pose (Failed)**
* **Script:** `gaussian_render_custom.py`
* **Goal:** Render the scene using the raw HLoc pose and estimated intrinsics.
* **Outcome:** **Visually incorrect.** The render failed because it bypassed the internal 3DGS scene normalization (scale/translation) and coordinate system alignment.

**Attempt B: Native Scene Loading (Success)**
* **Script:** `gaussian_render_scene_native.py`
* **Goal:** Sanity check the model by rendering a known training view using the official 3DGS `Scene` and `Camera` objects.
* **Outcome:** **Correct.** The render matched the ground truth, confirming the trained 3DGS model and `.ply` file were valid. The error was strictly in the HLoc → Render pipeline alignment.

### 3. Why HLoc Failed (Root Cause)
We frequently encountered `IndexError` or bad poses due to a **feature mismatch**:
* The **COLMAP model** was built using native **SIFT** keypoints.
* **HLoc** extracted and matched **SuperPoint** features.
* **Result:** `localize_sfm` attempted to map SuperPoint match indices into COLMAP's SIFT-based `points3D_ids`, causing index out-of-bounds errors. This confirmed that HLoc cannot directly localize against a pre-built SIFT COLMAP model without rebuilding the 3D features.

### 4. COLMAP image registration attempt (SIFT baseline)
We tried registering `query_image.jpg` into the existing COLMAP reconstruction by creating a fresh project folder (`$SCENE/colmap_reg`), symlinking all training images + the query into `images/`, copying `sparse/0` into `model/0`, then building a new `database.db` with `colmap feature_extractor` + `colmap exhaustive_matcher`. :contentReference[oaicite:0]{index=0}  
When we ran `colmap image_registrator --database_path database.db --input_path model/0 --output_path out`, COLMAP aborted with `Check failed: existing_image.Name() == image.second.Name() (... vs. query_image.jpg)`, i.e., the new database did not match the image entries expected by the existing reconstruction (database/model inconsistency). :contentReference[oaicite:1]{index=1}

## Phase 2: 6DGS Pose Estimation & Consistent Rendering

Phase 2 replaces the feature-mismatched HLoc pipeline with **6DGS**, a method *natively consistent with the trained 3DGS scene*. 6DGS predicts a camera pose relative to the Gaussian scene, allowing us to render directly from that pose. 

**References:**
- **6DGS:** [Paper](https://arxiv.org/abs/2407.15484) | [Code](https://github.com/mbortolon97/6dgs)
- **Focal Length + Pose:** [FocalPose++](https://arxiv.org/abs/2312.02985) (future direction for unknown intrinsics)

### Challenge: Unknown Intrinsics (AI Images)
6DGS expects known camera intrinsics. Since AI images lack metadata, we:
1.  Assume an approximate Field of View (e.g., `--fov_deg 60`).
2.  Apply small yaw corrections (+/- 20°) to compensate for angular error.

---

### (A) Diagnostic: Native Scene Render + Optimization
**Script:** `gaussian_render_scene_native_v2.py`
**Goal:** Sanity check the scene and optionally optimize a known camera's view toward the query image using photometric loss.

```bash
python gaussian_render_scene_native_v2.py \
  --source_path /scratch/schettip/gaussian-splatting/data/brandenburg_gate \
  --model_path  /scratch/schettip/gaussian-splatting/output/5db67e12-1 \
  --iteration   30000 \
  --image_name  00289298_7642283248.jpg \
  --out         output/native_init.png \
  --query_image /scratch/schettip/gaussian-splatting/Hierarchical-Localization/query_image.jpg \
  --opt_steps   400 \
  --opt_lr      1e-3
```

### (B) Train 6DGS Identification Module (Stage 1)
**Key Metrics** Look for low translation error and decent angular error (~10–13°).
**Outputs** id_module.th (trained weights) inside the experiment folder.

```bash
python pretrain_eval_attention.py \
  --exp_path /scratch/schettip/gaussian-splatting/output \
  --out_path brdbg_results.json \
  --data_type all
```

### (C) Estimate Query Pose (Stage 2)
**Script:** estimate_pose_from_image.py 
**Goal:** Predict the Camera-to-World (c2w) matrix for the query image using the trained ID module and an assumed FoV.

```bash
python estimate_pose_from_image.py \
  --exp_dir    /scratch/schettip/gaussian-splatting/output/5db67e12-1 \
  --ply_path   /scratch/schettip/gaussian-splatting/output/5db67e12-1/point_cloud/iteration_30000/point_cloud.ply \
  --query_image /scratch/schettip/gaussian-splatting/Hierarchical-Localization/query_image.jpg \
  --fov_deg    60 \
  --out_pose   /scratch/schettip/gaussian-splatting/output/brdbg_query_pose.npy
```

### (D) Render from Predicted 6DGS Pose
**Script:** render_from_6dgs.py 
**Goal:** Convert the predicted c2w pose to w2c, create a compatible camera, and render the scene.

```bash
python render_from_6dgs.py \
  --source_path /scratch/schettip/gaussian-splatting/data/brandenburg_gate \
  --model_path  /scratch/schettip/gaussian-splatting/output/5db67e12-1 \
  --iteration   30000 \
  --pose_npy    /scratch/schettip/gaussian-splatting/output/brdbg_query_pose.npy \
  --out         /scratch/schettip/gaussian-splatting/output/brdbg_query_render.png
```

### (E) Yaw Correction & High-Res Rendering
**Script:** render_from_6dgs_with_more_res.py Goal: Fix common heading errors (~10–20°) and render at higher resolution for better feature inspection.
**Yaw Variants:** Script automatically saves *_yawm30.png (-30°) and *_yawp30.png (+30°).
**Resolution:** --render_scale 1.5 improves detail (columns, silhouette) without inflating Gaussian geometry.

```bash
python render_from_6dgs_with_more_res.py \
  --source_path /scratch/schettip/gaussian-splatting/data/brandenburg_gate \
  --model_path  /scratch/schettip/gaussian-splatting/output/5db67e12-1 \
  --iteration   30000 \
  --pose_npy    /scratch/schettip/gaussian-splatting/output/brdbg_query_pose.npy \
  --out         /scratch/schettip/gaussian-splatting/output/brdbg_query_render_hr.png \
  --render_scale 1.5 \
  --scale_modifier 1
  ```

## Phase 3: Render-to-Query Comparisons (Verification)

Once a pose-consistent view is rendered from the 3DGS scene (Phase 2), we employ two comparison baselines: (1) learned local-feature matches (SuperPoint+SuperGlue) for qualitative sanity checks, and (2) a controlled SIFT+RANSAC homography alignment for quantitative pixel/gradient error metrics.

### 1. SuperPoint + SuperGlue Visualization (Qualitative)
**Script:** `superglue_compare_views.py`

**What it does:**
Extracts SuperPoint keypoints and descriptors, matches them using SuperGlue, and visualizes correspondences between the **Render ↔ Real** and **Render ↔ AI** pairs.

**Why use it:**
This verifies if "some structure matches." However, we observed that SuperGlue is often *too* robust—it can find visually plausible matches even when high-detail geometry (like the Quadriga statue in GPT images) is hallucinated or incorrect. It serves as a visual sanity check rather than a strict geometric verifier.

```bash
python superglue_compare_views.py \
  --real_img   /scratch/schettip/gaussian-splatting/data/brandenburg_gate/images/00289298_7642283248.jpg \
  --render_img /scratch/schettip/gaussian-splatting/output/brdbg_query_render_hr.png \
  --ai_img     /scratch/schettip/gaussian-splatting/Hierarchical-Localization/query_image.jpg \
  --out_dir    /scratch/schettip/gaussian-splatting/output/superglue_matches
```

### 2. SIFT + RANSAC Homography & Metrics (Main Baseline)
**Script:** `pixel_compare_homography.py`

**What it does:**
1.  **Normalization:** Resizes Render, Real, and AI images to a consistent dimension (`--max_long_edge`).
2.  **Feature Matching:** Detects and matches features (SIFT preferred, ORB fallback). 
3.  **Alignment:** Estimates a homography using **RANSAC** to map the query image into the render's perspective.
4.  **Warping:** Warps the query image to align with the render.
5.  **Metrics:** Computes full-reference metrics (MSE, MAE, PSNR) and an edge-sensitive metric (Gradient-MSE via Sobel filters).
6.  **Output:** Saves warped images, difference heatmaps, and side-by-side comparisons.

**Interpretation:**
Real images typically produce a significantly higher number of geometrically consistent inliers under RANSAC and lower pixel/gradient errors. AI images often align poorly, resulting in fewer inliers and higher residual errors, particularly in complex geometry-rich regions.

```bash
python pixel_compare_homography.py \
  --render_img /scratch/schettip/gaussian-splatting/output/brdbg_query_render_hr.png \
  --real_img   /scratch/schettip/gaussian-splatting/data/brandenburg_gate/images/00289298_7642283248.jpg \
  --ai_img     /scratch/schettip/gaussian-splatting/Hierarchical-Localization/query_image.jpg \
  --out_dir    /scratch/schettip/gaussian-splatting/output/pixel_compare1 \
  --max_long_edge 1400
```