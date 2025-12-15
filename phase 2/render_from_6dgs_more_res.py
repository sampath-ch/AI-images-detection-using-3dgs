#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from gaussian_renderer import render
from scene.gaussian_model import GaussianModel
from arguments import ModelParams, PipelineParams
import scene

# Wrapper to unify render output (dict vs tuple)
def call_render(view_cam, gaussians, pipe, bg_color, scale_mod: float):
    out = render(
        view_cam,
        gaussians,
        pipe,
        bg_color,
        scaling_modifier=scale_mod,
        separate_sh=False,
        override_color=None,
        use_trained_exp=False,
    )

    if isinstance(out, dict):
        img = out["render"]
        radii = out.get("radii", None)
        depth = out.get("depth", None)
    else:
        img, radii, depth = out

    return img, radii, depth

# Initialize 3DGS scene and model
def build_model_and_pipeline(source_path: Path, model_path: Path, iteration: int):
    import argparse as _argparse

    parser = _argparse.ArgumentParser(add_help=False)
    mp_group = ModelParams(parser)
    pp_group = PipelineParams(parser)

    argv = [
        f"--source_path={str(source_path)}",
        f"--model_path={str(model_path)}",
    ]
    args = parser.parse_args(argv)

    mp = mp_group.extract(args)
    pp = pp_group.extract(args)

    gaussians = GaussianModel(sh_degree=mp.sh_degree)
    SceneCls = scene.Scene

    try:
        scn = SceneCls(
            mp,
            gaussians,
            load_iteration=iteration,
            shuffle=False,
            resolution_scale=1.0,
        )
    except TypeError:
        scn = SceneCls(
            mp,
            gaussians,
            load_iteration=iteration,
            shuffle=False,
        )

    return scn, gaussians, pp

# Minimal camera wrapper with support for custom resolution and pose
class PoseCamera:
    def __init__(self, base_cam, w2c_matrix: torch.Tensor, render_scale: float = 1.0):
        device = w2c_matrix.device

        self.FoVx = float(base_cam.FoVx)
        self.FoVy = float(base_cam.FoVy)

        self.image_width = int(round(base_cam.image_width * render_scale))
        self.image_height = int(round(base_cam.image_height * render_scale))

        self.world_view_transform = w2c_matrix.to(device)
        self.projection_matrix = base_cam.projection_matrix.clone().to(device)

    @property
    def full_proj_transform(self):
        return self.world_view_transform @ self.projection_matrix

    @property
    def camera_center(self):
        W = self.world_view_transform
        W_inv = torch.inverse(W)
        return W_inv[3, :3]

# Utility for yaw rotation around world Y axis
def rotation_y(yaw_deg: float) -> np.ndarray:
    yaw = np.deg2rad(yaw_deg)
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=np.float32,
    )

def main():
    parser = argparse.ArgumentParser("Render 3DGS view from predicted pose (and +/- yaw)")
    parser.add_argument("--source_path", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--pose_npy", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--render_scale", type=float, default=1.0, help="Resolution scale factor")
    parser.add_argument("--scale_modifier", type=float, default=1.0, help="Gaussian scale modifier")
    args = parser.parse_args()

    source_path = Path(args.source_path)
    model_path = Path(args.model_path)
    pose_path = Path(args.pose_npy)
    out_path = Path(args.out)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Source: {source_path}")
    print(f"Model: {model_path}")
    print(f"Pose: {pose_path}")

    # 1. Build Scene
    scn, gaussians, pipeline = build_model_and_pipeline(
        source_path, model_path, args.iteration
    )

    # 2. Get template camera for intrinsics
    if hasattr(scn, "getTrainCameras"):
        cams = scn.getTrainCameras()
    elif hasattr(scn, "train_cameras"):
        cams = scn.train_cameras
    else:
        cams = getattr(scn, "cameras", [])

    if not cams:
        raise SystemExit("No train cameras found.")

    base_cam = cams[0]
    print("Using first train camera as template.")

    # 3. Load predicted pose
    c2w = np.load(pose_path)
    if c2w.shape != (4, 4):
        raise SystemExit(f"Invalid c2w shape: {c2w.shape}")

    w2c = np.linalg.inv(c2w)
    print(f"Loaded c2w:\n{c2w}")
    print(f"Derived w2c:\n{w2c}")
    w2c_torch = torch.from_numpy(w2c).float().to(device)

    # 4. Render main view
    pose_cam = PoseCamera(base_cam, w2c_torch, render_scale=args.render_scale)
    background = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=device)

    img_t, _, _ = call_render(pose_cam, gaussians, pipeline, background, args.scale_modifier)
    img = img_t.clamp(0.0, 1.0).permute(1, 2, 0).detach().cpu().numpy()
    img = (img * 255.0 + 0.5).astype(np.uint8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(out_path)
    print(f"Saved main render: {out_path}")

    # 5. Render yaw variants (+/- 30 degrees)
    stem = out_path.stem
    suffix = out_path.suffix or ".png"
    out_left = out_path.with_name(f"{stem}_yawm30{suffix}")
    out_right = out_path.with_name(f"{stem}_yawp30{suffix}")

    for yaw_deg, out_extra in [(-30.0, out_left), (30.0, out_right)]:
        print(f"Rendering yaw {yaw_deg:+.1f} deg...")

        R_y = rotation_y(yaw_deg)
        c2w_yaw = c2w.copy()
        c2w_yaw[:3, :3] = c2w[:3, :3] @ R_y
        w2c_yaw = np.linalg.inv(c2w_yaw)

        w2c_yaw_torch = torch.from_numpy(w2c_yaw).float().to(device)
        yaw_cam = PoseCamera(base_cam, w2c_yaw_torch, render_scale=args.render_scale)

        img_t_yaw, _, _ = call_render(
            yaw_cam, gaussians, pipeline, background, args.scale_modifier
        )
        img_yaw = img_t_yaw.clamp(0.0, 1.0).permute(1, 2, 0).detach().cpu().numpy()
        img_yaw = (img_yaw * 255.0 + 0.5).astype(np.uint8)

        Image.fromarray(img_yaw).save(out_extra)
        print(f"Saved yaw variant: {out_extra}")

if __name__ == "__main__":
    main()