#!/usr/bin/env python3
import argparse
from pathlib import Path
import math

import cv2
import numpy as np

# Load image in BGR and optionally resize
def load_color(path, max_long_edge=None):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    if max_long_edge is not None:
        h, w = img.shape[:2]
        scale = max_long_edge / float(max(h, w))
        if scale < 1.0:
            new_w = int(round(w * scale))
            new_h = int(round(h * scale))
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img

# Detect and match features using SIFT (preferred) or ORB
def detect_and_match(img_ref_gray, img_qry_gray, max_matches=2000):
    if hasattr(cv2, "SIFT_create"):
        detector = cv2.SIFT_create()
        use_sift = True
        print("Using SIFT for detection.")
    else:
        detector = cv2.ORB_create(5000)
        use_sift = False
        print("SIFT unavailable, using ORB.")

    kp1, des1 = detector.detectAndCompute(img_ref_gray, None)
    kp2, des2 = detector.detectAndCompute(img_qry_gray, None)

    if des1 is None or des2 is None or len(kp1) == 0 or len(kp2) == 0:
        print("No features found.")
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)

    if use_sift:
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
    else:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    matches = matcher.knnMatch(des1, des2, k=2)
    good = []
    ratio = 0.7 if use_sift else 0.75
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good.append(m)
    
    matches = sorted(good, key=lambda m: m.distance)[:max_matches]

    pts_ref = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts_qry = np.float32([kp2[m.trainIdx].pt for m in matches])

    print(f"Matches found: {len(matches)}")
    return pts_ref, pts_qry

# Estimate homography matrix with RANSAC
def find_homography(pts_src, pts_dst, ransac_thresh=3.0):
    if len(pts_src) < 4:
        print("Not enough points for homography.")
        return None, np.zeros(len(pts_src), dtype=bool)

    pts_src_ = pts_src.reshape(-1, 1, 2)
    pts_dst_ = pts_dst.reshape(-1, 1, 2)

    H, mask = cv2.findHomography(pts_src_, pts_dst_, cv2.RANSAC, ransac_thresh)
    if mask is None:
        print("Homography estimation failed.")
        return None, np.zeros(len(pts_src), dtype=bool)

    inliers = mask.ravel().astype(bool)
    print(f"Homography inliers: {inliers.sum()}/{len(inliers)}")
    return H, inliers

# Compute MSE, MAE, PSNR, and Gradient MSE
def compute_metrics(ref, warped):
    ref_f = ref.astype(np.float32) / 255.0
    warped_f = warped.astype(np.float32) / 255.0

    diff = ref_f - warped_f
    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(np.abs(diff)))

    if mse > 0:
        psnr = 10.0 * math.log10(1.0 / mse)
    else:
        psnr = float('inf')

    # Gradient difference on grayscale
    ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    war_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    gx_ref = cv2.Sobel(ref_gray, cv2.CV_32F, 1, 0, ksize=3)
    gy_ref = cv2.Sobel(ref_gray, cv2.CV_32F, 0, 1, ksize=3)
    gx_war = cv2.Sobel(war_gray, cv2.CV_32F, 1, 0, ksize=3)
    gy_war = cv2.Sobel(war_gray, cv2.CV_32F, 0, 1, ksize=3)

    grad_mse = float(np.mean((gx_ref - gx_war) ** 2 + (gy_ref - gy_war) ** 2))

    return {"mse": mse, "mae": mae, "psnr": psnr, "grad_mse": grad_mse}

# Create heatmap of pixel differences 
def make_diff_heatmap(ref, warped):
    ref_f = ref.astype(np.float32)
    war_f = warped.astype(np.float32)

    diff = np.mean(np.abs(ref_f - war_f), axis=2)
    diff_norm = diff / (diff.max() + 1e-8)
    diff_uint8 = (diff_norm * 255).astype(np.uint8)

    heat = cv2.applyColorMap(diff_uint8, cv2.COLORMAP_JET)
    return heat

# Align images using homography and compute metrics
def process_pair(ref_color, other_color, name, out_dir):
    h_ref, w_ref = ref_color.shape[:2]
    other_gray = cv2.cvtColor(other_color, cv2.COLOR_BGR2GRAY)
    ref_gray = cv2.cvtColor(ref_color, cv2.COLOR_BGR2GRAY)

    pts_ref, pts_other = detect_and_match(ref_gray, other_gray)

    # Calculate homography mapping 'other' -> 'ref'
    H, inliers = find_homography(pts_other, pts_ref)

    if H is None:
        print(f"Skipping metrics for {name} (no Homography).")
        return

    warped_other = cv2.warpPerspective(other_color, H, (w_ref, h_ref))

    metrics = compute_metrics(ref_color, warped_other)
    print(f"\nResult: {name}")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}")
    print("")

    heat = make_diff_heatmap(ref_color, warped_other)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(out_dir / f"warped_{name}.png"), warped_other)
    cv2.imwrite(str(out_dir / f"diff_heatmap_{name}.png"), heat)
    side = np.concatenate([ref_color, warped_other, heat], axis=1)
    cv2.imwrite(str(out_dir / f"side_by_side_{name}.png"), side)

def main():
    ap = argparse.ArgumentParser("Pixel-level comparison: RENDER vs REAL/AI")
    ap.add_argument("--render_img", required=True, help="Path to 3DGS render")
    ap.add_argument("--real_img", required=True, help="Path to real photo")
    ap.add_argument("--ai_img", required=True, help="Path to AI-generated image")
    ap.add_argument("--out_dir", required=True, help="Output directory")
    ap.add_argument("--max_long_edge", type=int, default=1400, help="Resize max dimension")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    render = load_color(args.render_img, args.max_long_edge)
    real = load_color(args.real_img, args.max_long_edge)
    ai = load_color(args.ai_img, args.max_long_edge)

    print(f"Render size: {render.shape[1]}x{render.shape[0]}")
    print(f"Real size: {real.shape[1]}x{real.shape[0]}")
    print(f"AI size: {ai.shape[1]}x{ai.shape[0]}")

    print("\nProcessing RENDER vs REAL...")
    process_pair(render, real, "render_real", out_dir)

    print("\nProcessing RENDER vs AI...")
    process_pair(render, ai, "render_ai", out_dir)

if __name__ == "__main__":
    main()