#!/usr/bin/env python3
import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from gaussian_renderer import render
from scene.gaussian_model import GaussianModel
from arguments import ModelParams, PipelineParams
import scene

# Wrapper to unify render output
def call_render(view_cam, gaussians, pipe, bg_color):
    out = render(view_cam, gaussians, pipe, bg_color)

    if isinstance(out, dict):
        img = out["render"]
        radii = out.get("radii", None)
        depth = out.get("depth", None)
    else:
        img, radii, depth = out

    return img, radii, depth

# Initialize 3DGS scene
def build_model_and_pipeline(source_path: Path, model_path: Path, iteration: int):
    import argparse as _argparse

    parser = _argparse.ArgumentParser(add_help=False)
    mp_group = ModelParams(parser)
    pp_group = PipelineParams(parser)

    argv = [
        f'--source_path={str(source_path)}',
        f'--model_path={str(model_path)}',
    ]
    args = parser.parse_args(argv)

    mp = mp_group.extract(args)
    pp = pp_group.extract(args)

    gaussians = GaussianModel(sh_degree=mp.sh_degree)
    SceneCls = scene.Scene

    try:
        scn = SceneCls(mp, gaussians, load_iteration=iteration, shuffle=False, resolution_scale=1.0)
    except TypeError:
        scn = SceneCls(mp, gaussians, load_iteration=iteration, shuffle=False)

    return scn, gaussians, pp

# Minimal camera wrapper
class PoseCamera:
    def __init__(self, base_cam, w2c_matrix: torch.Tensor):
        device = w2c_matrix.device

        self.FoVx = float(base_cam.FoVx)
        self.FoVy = float(base_cam.FoVy)
        self.image_width = int(base_cam.image_width)
        self.image_height = int(base_cam.image_height)

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

def main():
    parser = argparse.ArgumentParser("Render 3DGS view from 6DGS-predicted pose")
    parser.add_argument("--source_path", required=True, help="Dataset root")
    parser.add_argument("--model_path", required=True, help="3DGS experiment path")
    parser.add_argument("--iteration", type=int, default=30000, help="Training iteration")
    parser.add_argument("--pose_npy", required=True, help="Path to c2w numpy pose")
    parser.add_argument("--out", required=True, help="Output PNG path")
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
    scn, gaussians, pipeline = build_model_and_pipeline(source_path, model_path, args.iteration)

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

    # 3. Load pose and convert to w2c
    c2w = np.load(pose_path)
    print(f"Loaded c2w:\n{c2w}")

    w2c = np.linalg.inv(c2w)
    print(f"Derived w2c:\n{w2c}")

    w2c_torch = torch.from_numpy(w2c).float().to(device)

    # 4. Setup camera and render
    pose_cam = PoseCamera(base_cam, w2c_torch)
    background = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=device)

    img_t, _, _ = call_render(pose_cam, gaussians, pipeline, background)
    img = img_t.clamp(0.0, 1.0).permute(1, 2, 0).detach().cpu().numpy()
    img = (img * 255.0 + 0.5).astype(np.uint8)

    # 5. Save output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(out_path)
    print(f"Saved render to: {out_path}")

if __name__ == "__main__":
    main()