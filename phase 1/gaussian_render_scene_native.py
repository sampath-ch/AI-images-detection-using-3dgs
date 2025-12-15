#!/usr/bin/env python3
import argparse
from pathlib import Path
import torch
import numpy as np
from PIL import Image

from gaussian_renderer import render
from scene.gaussian_model import GaussianModel
from arguments import ModelParams, PipelineParams
import scene

def build_model_and_pipeline(source_path: Path, model_path: Path, iteration: int):
    import argparse as _argparse

    parser = _argparse.ArgumentParser(add_help=False)
    mp_group = ModelParams(parser)
    pp_group = PipelineParams(parser)

    # Emulate CLI arguments for ModelParams
    argv = [
        f'--source_path={str(source_path)}',
        f'--model_path={str(model_path)}',
    ]
    args = parser.parse_args(argv)

    mp = mp_group.extract(args)
    pp = pp_group.extract(args)

    gaussians = GaussianModel(sh_degree=mp.sh_degree)

    # Initialize Scene (handles loading from checkpoint/iteration)
    Scene = scene.Scene
    try:
        scn = Scene(mp, gaussians, load_iteration=iteration, shuffle=False, resolution_scale=1.0)
    except TypeError:
        # Fallback for older forks without resolution_scale
        scn = Scene(mp, gaussians, load_iteration=iteration, shuffle=False)

    return scn, gaussians, pp

def main():
    parser = argparse.ArgumentParser("Render a training view using native Scene")
    parser.add_argument("--source_path", required=True, help="Dataset root")
    parser.add_argument("--model_path", required=True, help="Path to training output folder")
    parser.add_argument("--iteration", type=int, default=30000, help="Iteration to load")
    parser.add_argument("--image_name", required=True, help="Training image name")
    parser.add_argument("--out", default="native_render.png", help="Output PNG")
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
    print(f"Image: {image_key}")

    # Build Scene and Gaussians using native pipeline logic
    scn, gaussians, pipeline = build_model_and_pipeline(source_path, model_path, args.iteration)

    # Retrieve cameras from the loaded scene
    if hasattr(scn, "getTrainCameras"):
        cams = scn.getTrainCameras()
    elif hasattr(scn, "train_cameras"):
        cams = scn.train_cameras
    else:
        cams = getattr(scn, "cameras", [])

    if not cams:
        raise SystemExit("Could not find any train cameras on Scene object.")

    # Find the specific camera object matching the requested image name
    target_cam = None
    for c in cams:
        name = getattr(c, "image_name", None)
        if name is None and hasattr(c, "image_path"):
            name = Path(c.image_path).name
        if name == image_key:
            target_cam = c
            break

    if target_cam is None:
        sample = []
        for c in cams[:20]:
            nm = getattr(c, "image_name", None)
            if nm is None and hasattr(c, "image_path"):
                nm = Path(c.image_path).name
            sample.append(str(nm))
        raise SystemExit(f"Could not find camera for {image_key}. Samples: {sample}")

    print(f"Found camera for {image_key}")

    # Render
    background = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=device)
    render_pkg = render(target_cam, gaussians, pipeline, background)
    img = render_pkg["render"].clamp(0.0, 1.0).permute(1, 2, 0).detach().cpu().numpy()
    img = (img * 255.0 + 0.5).astype(np.uint8)

    # Save output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(out_path)
    print(f"Saved native Scene render to: {out_path}")

if __name__ == "__main__":
    main()