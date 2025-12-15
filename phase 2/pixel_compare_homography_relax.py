#!/usr/bin/env python3
import argparse
from pathlib import Path
import math

import cv2
import numpy as np


def load_color(path, max_long_edge=None):
    """
    Load image in color (BGR). Optionally resize so longest side = max_long_edge.
    """
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


def detect_and_match(img_ref_gray, img_qry_gray, max_matches=2000):
    """
    Detect keypoints and match between ref and query using SIFT or ORB.

    Returns:
      pts_ref: [N,2]
      pts_qry: [N,2]
    """
    if hasattr(cv2, "SIFT_create"):
        detector = cv2.SIFT_create()
        use_sift = True
        print("[INFO] Using SIFT for feature detection.")
    else:
        detector = cv2.ORB_create(5000)
        use_sift = False
        print("[INFO] SIFT not available, falling back to ORB.")

    kp1, des1 = detector.detectAndCompute(img_ref_gray, None)
    kp2, des2 = detector.detectAndCompute(img_qry_gray, None)

    if des1 is None or des2 is None or len(kp1) == 0 or len(kp2) == 0:
        print("[WARN] No features found in one of the images.")
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)

    if use_sift:
        index_params = dict(algorithm=1, trees=5)  # FLANN + KD-tree
        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
    else:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    if use_sift:
        matches = matcher.knnMatch(des1, des2, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)
        matches = good
    else:
        matches = matcher.knnMatch(des1, des2, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)
        matches = good

    matches = sorted(matches, key=lambda m: m.distance)
    matches = matches[:max_matches]

    pts_ref = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts_qry = np.float32([kp2[m.trainIdx].pt for m in matches])

    print(f"[INFO] Raw matches: {len(matches)}")
    return pts_ref, pts_qry


def find_homography(pts_src,
                    pts_dst,
                    src_shape,
                    ransac_thresh=3.0,
                    min_inliers=40,
                    min_span_frac=0.25,
                    max_span_frac=4.0):
    """
    Estimate homography that maps pts_src -> pts_dst using RANSAC.
    Also performs sanity checks to reject degenerate homographies.

    src_shape: shape of source image (H,W,3) used to test corner warp.
    """
    if len(pts_src) < 4:
        print("[WARN] Not enough points to estimate homography.")
        return None, np.zeros(len(pts_src), dtype=bool)

    pts_src_ = pts_src.reshape(-1, 1, 2)
    pts_dst_ = pts_dst.reshape(-1, 1, 2)

    H, mask = cv2.findHomography(pts_src_, pts_dst_, cv2.RANSAC, ransac_thresh)
    if mask is None or H is None:
        print("[WARN] Homography estimation failed.")
        return None, np.zeros(len(pts_src), dtype=bool)

    inliers = mask.ravel().astype(bool)
    num_inliers = int(inliers.sum())
    print(f"[INFO] Homography inliers: {num_inliers}/{len(inliers)}")

    if num_inliers < min_inliers:
        print(f"[WARN] Too few inliers ({num_inliers} < {min_inliers}), "
              f"treating homography as unreliable.")
        return None, inliers

    # ---- Degeneracy checks via corner warp ----
    h_src, w_src = src_shape[:2]
    corners = np.array(
        [[0, 0],
         [w_src - 1, 0],
         [w_src - 1, h_src - 1],
         [0, h_src - 1]],
        dtype=np.float32
    ).reshape(-1, 1, 2)

    proj = cv2.perspectiveTransform(corners, H).reshape(-1, 2)

    if not np.isfinite(proj).all():
        print("[WARN] Homography produced non-finite corner projections; "
              "treating as degenerate.")
        return None, inliers

    # measure how spread out the projected corners are
    span = proj.max(axis=0) - proj.min(axis=0)
    max_span = float(np.linalg.norm(span))
    src_diag = float(np.linalg.norm([w_src, h_src]))

    if max_span < min_span_frac * src_diag:
        print("[WARN] Homography collapses the image (span too small); "
              "treating as degenerate.")
        return None, inliers

    if max_span > max_span_frac * src_diag:
        print("[WARN] Homography expands the image excessively (span too large); "
              "treating as degenerate.")
        return None, inliers

    return H, inliers


def compute_metrics(ref, warped):
    """
    Compute pixel-level metrics between two color images (same size).
    ref, warped: uint8 BGR, same HxW.
    """
    ref_f = ref.astype(np.float32) / 255.0
    warped_f = warped.astype(np.float32) / 255.0

    diff = ref_f - warped_f
    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(np.abs(diff)))

    if mse > 0:
        psnr = 10.0 * math.log10(1.0 / mse)
    else:
        psnr = float('inf')

    # gradient difference (on grayscale)
    ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    war_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    gx_ref = cv2.Sobel(ref_gray, cv2.CV_32F, 1, 0, ksize=3)
    gy_ref = cv2.Sobel(ref_gray, cv2.CV_32F, 0, 1, ksize=3)
    gx_war = cv2.Sobel(war_gray, cv2.CV_32F, 1, 0, ksize=3)
    gy_war = cv2.Sobel(war_gray, cv2.CV_32F, 0, 1, ksize=3)

    grad_mse = float(np.mean((gx_ref - gx_war) ** 2 + (gy_ref - gy_war) ** 2))

    return {
        "mse": mse,
        "mae": mae,
        "psnr": psnr,
        "grad_mse": grad_mse,
    }


def make_diff_heatmap(ref, warped):
    """
    Build a color heatmap of per-pixel absolute difference (L1 per pixel).
    """
    ref_f = ref.astype(np.float32)
    war_f = warped.astype(np.float32)

    diff = np.mean(np.abs(ref_f - war_f), axis=2)  # [H,W]
    diff_norm = diff / (diff.max() + 1e-8)
    diff_uint8 = (diff_norm * 255).astype(np.uint8)

    heat = cv2.applyColorMap(diff_uint8, cv2.COLORMAP_JET)
    return heat


def process_pair(ref_color, other_color, name, out_dir, min_inliers):
    """
    Align 'other_color' to 'ref_color' using homography, then compute metrics.
    min_inliers: homography RANSAC inlier threshold to accept H.
    """
    h_ref, w_ref = ref_color.shape[:2]
    other_gray = cv2.cvtColor(other_color, cv2.COLOR_BGR2GRAY)
    ref_gray = cv2.cvtColor(ref_color, cv2.COLOR_BGR2GRAY)

    # detect & match features (reference vs other)
    pts_ref, pts_other = detect_and_match(ref_gray, other_gray)

    # homography mapping other -> ref
    H, inliers = find_homography(
        pts_other,
        pts_ref,
        src_shape=other_color.shape,
        ransac_thresh=3.0,
        min_inliers=min_inliers,
        min_span_frac=0.25,
        max_span_frac=4.0,
    )

    if H is None:
        print(f"[WARN] Skipping pixel metrics for {name} (homography unreliable).")
        return

    warped_other = cv2.warpPerspective(other_color, H, (w_ref, h_ref))

    metrics = compute_metrics(ref_color, warped_other)
    print(f"\n[RESULT] {name}")
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
    ap = argparse.ArgumentParser(
        "Pixel-level comparison of RENDER vs REAL and RENDER vs AI "
        "using homography + per-pixel metrics."
    )
    ap.add_argument("--render_img", required=True,
                    help="Path to 3DGS render (reference frame)")
    ap.add_argument("--real_img", required=True,
                    help="Path to real photo")
    ap.add_argument("--ai_img", required=True,
                    help="Path to AI-generated image")
    ap.add_argument("--out_dir", required=True,
                    help="Directory to save outputs")
    ap.add_argument("--max_long_edge", type=int, default=1400,
                    help="Resize longest side of all images to this (for speed/stability)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load images & resize to same scale
    render = load_color(args.render_img, args.max_long_edge)
    real = load_color(args.real_img, args.max_long_edge)
    ai = load_color(args.ai_img, args.max_long_edge)

    print(f"[INFO] render size: {render.shape[1]}x{render.shape[0]}")
    print(f"[INFO] real   size: {real.shape[1]}x{real.shape[0]}")
    print(f"[INFO] ai     size: {ai.shape[1]}x{ai.shape[0]}")

    print("\n=== RENDER vs REAL ===")
    process_pair(render, real, "render_real", out_dir, min_inliers=40)

    print("\n=== RENDER vs AI ===")
    # allow weaker homography here; content is very different
    process_pair(render, ai, "render_ai", out_dir, min_inliers=8)


if __name__ == "__main__":
    main()
