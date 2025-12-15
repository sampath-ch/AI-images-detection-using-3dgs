#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

# Point this to your cloned SuperGlue repo
SUPERGLUE_ROOT = "/scratch/schettip/gaussian-splatting/SuperGluePretrainedNetwork"
sys.path.append(str(Path(SUPERGLUE_ROOT)))

from models.superpoint import SuperPoint
from models.superglue import SuperGlue

# Load image in color (BGR) and resize
def load_color_resize(path, max_long_edge=1024):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    h, w = img.shape[:2]
    scale = max_long_edge / float(max(h, w))
    if scale < 1.0:
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img

# Convert grayscale image to torch tensor [1,1,H,W]
def to_tensor_gray(img_gray, device):
    t = torch.from_numpy(img_gray).float() / 255.0
    t = t.unsqueeze(0).unsqueeze(0)
    return t.to(device)

# Align 'other_color' to 'render_color' using ORB features and homography
def align_image_to_render(render_color, other_color, min_matches=10):
    h_ref, w_ref = render_color.shape[:2]

    ref_gray = cv2.cvtColor(render_color, cv2.COLOR_BGR2GRAY)
    oth_gray = cv2.cvtColor(other_color, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(2000)
    kp1, des1 = orb.detectAndCompute(ref_gray, None)
    kp2, des2 = orb.detectAndCompute(oth_gray, None)

    if des1 is None or des2 is None:
        print("ORB descriptors missing, skipping alignment.")
        return other_color

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    if len(matches) < min_matches:
        print(f"Not enough matches ({len(matches)} < {min_matches}), skipping alignment.")
        return other_color

    matches = sorted(matches, key=lambda x: x.distance)
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)
    if H is None:
        print("Homography estimation failed, skipping alignment.")
        return other_color

    aligned = cv2.warpPerspective(other_color, H, (w_ref, h_ref))
    return aligned

# Run SuperPoint and SuperGlue on a pair of grayscale images
def run_superpoint_superglue(img0_gray, img1_gray, sp, sg, device):
    t0 = to_tensor_gray(img0_gray, device)
    t1 = to_tensor_gray(img1_gray, device)

    # SuperPoint
    with torch.no_grad():
        pred0 = sp({'image': t0})
        pred1 = sp({'image': t1})

    def normalize_sp_pred(pred):
        k = pred['keypoints']
        d = pred['descriptors']
        s = pred['scores']

        if isinstance(k, (list, tuple)): k = k[0]
        if isinstance(d, (list, tuple)): d = d[0]
        if isinstance(s, (list, tuple)): s = s[0]

        if k.ndim == 2: k = k.unsqueeze(0)
        if d.ndim == 2: d = d.unsqueeze(0)
        if s.ndim == 1: s = s.unsqueeze(0)

        return k, d, s

    k0_b, d0_b, s0_b = normalize_sp_pred(pred0)
    k1_b, d1_b, s1_b = normalize_sp_pred(pred1)

    data = {
        'image0': t0,
        'image1': t1,
        'keypoints0': k0_b,
        'keypoints1': k1_b,
        'descriptors0': d0_b,
        'descriptors1': d1_b,
        'scores0': s0_b,
        'scores1': s1_b,
    }

    # SuperGlue
    with torch.no_grad():
        pred = sg(data)

    matches0 = pred['matches0']
    scores0 = pred['matching_scores0']

    if isinstance(matches0, (list, tuple)): matches0 = matches0[0]
    if isinstance(scores0, (list, tuple)): scores0 = scores0[0]

    if matches0.ndim == 2: matches0 = matches0[0]
    if scores0.ndim == 2: scores0 = scores0[0]

    matches0 = matches0.cpu().numpy()
    scores0 = scores0.cpu().numpy()

    kpts0 = k0_b[0].cpu().numpy()
    kpts1 = k1_b[0].cpu().numpy()

    valid = matches0 > -1
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches0[valid]]
    mscores = scores0[valid]

    return mkpts0, mkpts1, mscores

# Visualization helper
def draw_matches_side_by_side_color(img0_color, img1_color, kpts0, kpts1, scores, max_draw=200):
    h0, w0 = img0_color.shape[:2]
    h1, w1 = img1_color.shape[:2]
    H = max(h0, h1)
    W = w0 + w1

    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    canvas[:h0, :w0, :] = img0_color
    canvas[:h1, w0:w0 + w1, :] = img1_color

    if len(scores) > 0:
        s_min, s_max = float(scores.min()), float(scores.max())
        if s_max > s_min:
            norm_scores = (scores - s_min) / (s_max - s_min)
        else:
            norm_scores = np.ones_like(scores)
    else:
        norm_scores = scores

    order = np.argsort(-norm_scores)
    order = order[:max_draw]

    for i in order:
        x0, y0 = kpts0[i]
        x1, y1 = kpts1[i]
        x1_shift = x1 + w0

        s = float(norm_scores[i])
        color = (0, int(255 * s), 255 - int(255 * s))
        p0 = (int(round(x0)), int(round(y0)))
        p1 = (int(round(x1_shift)), int(round(y1)))

        cv2.circle(canvas, p0, 3, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, p1, 3, color, -1, lineType=cv2.LINE_AA)
        cv2.line(canvas, p0, p1, color, 1, lineType=cv2.LINE_AA)

    return canvas

def main():
    parser = argparse.ArgumentParser("Compare RENDER vs AI/REAL using SuperPoint+SuperGlue")
    parser.add_argument("--real_img", required=True, help="Path to real photo")
    parser.add_argument("--render_img", required=True, help="Path to 3DGS render")
    parser.add_argument("--ai_img", required=True, help="Path to AI-generated image")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument("--max_long_edge", type=int, default=1024, help="Resize max dimension")
    parser.add_argument("--max_draw", type=int, default=200, help="Max matches to draw")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # 
    sp_config = {
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': 1024,
    }
    sg_config = {
        'weights': 'outdoor',
        'sinkhorn_iterations': 20,
        'match_threshold': 0.2,
    }

    sp = SuperPoint(sp_config).eval().to(device)
    sg = SuperGlue(sg_config).eval().to(device)
    print("Loaded SuperPoint and SuperGlue models.")

    render_color = load_color_resize(args.render_img, args.max_long_edge)
    real_color = load_color_resize(args.real_img, args.max_long_edge)
    ai_color = load_color_resize(args.ai_img, args.max_long_edge)

    # Align images
    print("Aligning REAL to RENDER...")
    real_aligned = align_image_to_render(render_color, real_color)
    print("Aligning AI to RENDER...")
    ai_aligned = align_image_to_render(render_color, ai_color)

    cv2.imwrite(str(out_dir / "real_aligned_to_render.png"), real_aligned)
    cv2.imwrite(str(out_dir / "ai_aligned_to_render.png"), ai_aligned)

    render_gray = cv2.cvtColor(render_color, cv2.COLOR_BGR2GRAY)
    real_gray_aligned = cv2.cvtColor(real_aligned, cv2.COLOR_BGR2GRAY)
    ai_gray_aligned = cv2.cvtColor(ai_aligned, cv2.COLOR_BGR2GRAY)

    # 1. RENDER vs AI
    print("Matching RENDER <-> AI...")
    k0, k1, sc = run_superpoint_superglue(render_gray, ai_gray_aligned, sp, sg, device)
    canvas_render_ai = draw_matches_side_by_side_color(
        render_color, ai_aligned, k0, k1, sc, max_draw=args.max_draw
    )
    out_render_ai = out_dir / "matches_render_ai_color.png"
    cv2.imwrite(str(out_render_ai), canvas_render_ai)
    print(f"Saved AI matches: {out_render_ai} ({len(sc)} matches)")

    # 2. RENDER vs REAL
    print("Matching RENDER <-> REAL...")
    k0, k1, sc = run_superpoint_superglue(render_gray, real_gray_aligned, sp, sg, device)
    canvas_render_real = draw_matches_side_by_side_color(
        render_color, real_aligned, k0, k1, sc, max_draw=args.max_draw
    )
    out_render_real = out_dir / "matches_render_real_color.png"
    cv2.imwrite(str(out_render_real), canvas_render_real)
    print(f"Saved REAL matches: {out_render_real} ({len(sc)} matches)")

if __name__ == "__main__":
    main()