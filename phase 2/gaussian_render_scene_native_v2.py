#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image

from gaussian_renderer import render
from scene.gaussian_model import GaussianModel
from arguments import ModelParams, PipelineParams
import scene

# Load and preprocess query image as tensor
def load_query_image_tensor(path, target_hw=None, device="cuda"):
    img = Image.open(path).convert("RGB")
    if target_hw is not None:
        H, W = target_hw
        img = img.resize((W, H), Image.BILINEAR)
    arr = np.array(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    return tensor

# Wrapper for camera to optimize its pose while keeping intrinsics fixed
class OptimizedViewCamera(nn.Module):
    def __init__(self, base_cam):
        super().__init__()
        device = base_cam.world_view_transform.device

        self.FoVx = float(base_cam.FoVx)
        self.FoVy = float(base_cam.FoVy)
        self.image_width = int(base_cam.image_width)
        self.image_height = int(base_cam.image_height)

        self.world_view = nn.Parameter(
            base_cam.world_view_transform.clone().to(device)
        )

        self.proj = base_cam.projection_matrix.clone().to(device)

    @property
    def world_view_transform(self):
        return self.world_view

    @property
    def full_proj_transform(self):
        return self.world_view @ self.proj

    @property
    def camera_center(self):
        W = self.world_view
        W_inv = torch.inverse(W)
        return W_inv[3, :3]

# Run optimization loop for camera pose
def optimize_camera_pose(gaussians, pipe, init_cam, query_tensor,
                         steps=400, lr=1e-3, device="cuda", bg_color=None):
    if bg_color is None:
        bg_color = torch.tensor([0.0, 0.0, 0.0], device=device)

    # Align camera resolution with query
    _, _, Hq, Wq = query_tensor.shape
    init_cam.image_height = Hq
    init_cam.image_width = Wq

    opt_cam = OptimizedViewCamera(init_cam).to(device)
    optimizer = torch.optim.Adam(opt_cam.parameters(), lr=lr)

    print(f"Starting camera optimization for {steps} steps...")
    for step in range(steps):
        optimizer.zero_grad()

        render_pkg = render(
            opt_cam,
            gaussians,
            pipe,
            bg_color,
            scaling_modifier=1.0,
            separate_sh=False,
            override_color=None,
            use_trained_exp=False
        )
        rendered = render_pkg["render"]
        rendered_b = rendered.unsqueeze(0)

        # Loss: L1 + small regularization on matrix values
        photometric = F.l1_loss(rendered_b, query_tensor)
        reg = 1e-4 * (opt_cam.world_view ** 2).mean()
        loss = photometric + reg

        loss.backward()
        optimizer.step()

        if (step + 1) % 50 == 0 or step == 0:
            print(f"Step {step+1}/{steps}, loss={loss.item():.6f}, L1={photometric.item():.6f}")

    print("Finished optimization.")
    return opt_cam

# Initialize model and pipeline params
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

    Scene = scene.Scene
    try:
        scn = Scene(mp, gaussians, load_iteration=iteration, shuffle=False, resolution_scale=1.0)
    except TypeError:
        scn = Scene(mp, gaussians, load_iteration=iteration, shuffle=False)

    return scn, gaussians, pp

def main():
    parser = argparse.ArgumentParser("Render native Scene (optional pose optimization)")
    parser.add_argument("--source_path", required=True, help="Dataset root")
    parser.add_argument("--model_path", required=True, help="Training output folder")
    parser.add_argument("--iteration", type=int, default=30000, help="Iteration to load")
    parser.add_argument("--image_name", required=True, help="Training image name (initial pose)")
    parser.add_argument("--out", default="native_render.png", help="Output PNG")
    parser.add_argument("--query_image", type=str, default=None, help="Image to optimize pose against")
    parser.add_argument("--opt_steps", type=int, default=400, help="Optimization steps")
    parser.add_argument("--opt_lr", type=float, default=1e-3, help="Optimization Learning Rate")
    parser.add_argument("--opt_out", type=str, default=None, help="Output path for optimized render")
    args = parser.parse_args()

    source_path = Path(args.source_path)
    model_path = Path(args.model_path)
    out_path = Path(args.out)
    image_key = Path(args.image_name).name

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Source: {source_path}")
    print(f"Model: {model_path}")
    print(f"Iteration: {args.iteration}")
    print(f"Init Image: {image_key}")

    scn, gaussians, pipeline = build_model_and_pipeline(source_path, model_path, args.iteration)

    # Find matching training camera
    if hasattr(scn, "getTrainCameras"):
        cams = scn.getTrainCameras()
    elif hasattr(scn, "train_cameras"):
        cams = scn.train_cameras
    else:
        cams = getattr(scn, "cameras", [])

    if not cams:
        raise SystemExit("No train cameras found on Scene object.")

    target_cam = None
    for c in cams:
        name = getattr(c, "image_name", None)
        if name is None and hasattr(c, "image_path"):
            name = Path(c.image_path).name
        if name == image_key:
            target_cam = c
            break

    if target_cam is None:
        sample = [Path(c.image_path).name for c in cams[:20] if hasattr(c, "image_path")]
        raise SystemExit(f"Camera {image_key} not found. Samples: {sample}")

    print(f"Found camera for {image_key}")

    # Render initial view
    background = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=device)
    render_pkg = render(target_cam, gaussians, pipeline, background)
    img = render_pkg["render"].clamp(0.0, 1.0).permute(1, 2, 0).detach().cpu().numpy()
    img = (img * 255.0 + 0.5).astype(np.uint8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(out_path)
    print(f"Saved native Scene render to: {out_path}")

    # Optimize pose if query image is provided
    if args.query_image is not None:
        print(f"Query image: {args.query_image}")
        H0 = getattr(target_cam, "image_height", img.shape[0])
        W0 = getattr(target_cam, "image_width", img.shape[1])
        query_tensor = load_query_image_tensor(args.query_image, target_hw=(H0, W0), device=device)

        opt_cam = optimize_camera_pose(
            gaussians=gaussians,
            pipe=pipeline,
            init_cam=target_cam,
            query_tensor=query_tensor,
            steps=args.opt_steps,
            lr=args.opt_lr,
            device=str(device),
            bg_color=background,
        )

        opt_pkg = render(opt_cam, gaussians, pipeline, background)
        opt_img = opt_pkg["render"].clamp(0.0, 1.0).permute(1, 2, 0).detach().cpu().numpy()
        opt_img = (opt_img * 255.0 + 0.5).astype(np.uint8)

        if args.opt_out is None:
            opt_out_path = out_path.with_name(out_path.stem + "_opt.png")
        else:
            opt_out_path = Path(args.opt_out)

        opt_out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(opt_img).save(opt_out_path)
        print(f"Saved optimized render to: {opt_out_path}")

        W = opt_cam.world_view_transform.detach().cpu().numpy()
        W_inv = np.linalg.inv(W)
        C = W_inv[3, :3]
        print(f"Final world_view_transform:\n{W}")
        print(f"Approx camera center C: {C}")

if __name__ == "__main__":
    main()