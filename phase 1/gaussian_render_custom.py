#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import torch
from PIL import Image

from gaussian_renderer import render
from scene.gaussian_model import GaussianModel
from scene.cameras import Camera
from arguments import PipelineParams
from utils.graphics_utils import focal2fov

# Convert COLMAP quaternion to rotation matrix
def qvec2rotmat(qvec):
    qw, qx, qy, qz = qvec
    return np.array([
        [1 - 2 * qy * qy - 2 * qz * qz,
         2 * qx * qy - 2 * qw * qz,
         2 * qx * qz + 2 * qw * qy],
        [2 * qx * qy + 2 * qw * qz,
         1 - 2 * qx * qx - 2 * qz * qz,
         2 * qy * qz - 2 * qw * qx],
        [2 * qx * qz - 2 * qw * qy,
         2 * qy * qz + 2 * qw * qx,
         1 - 2 * qx * qx - 2 * qy * qy],
    ], dtype=np.float32)

def _norm_name(name: str) -> str:
    return Path(name).name

# Parse HLoc localization_results.txt for qvec and tvec
def load_pose_from_loc(loc_path: Path, image_name: str):
    key = _norm_name(image_name)
    with loc_path.open("r") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.split()
            if _norm_name(parts[0]) == key:
                if len(parts) < 8:
                    raise RuntimeError(f"Line for {key} malformed: {line}")
                qvec = np.array(list(map(float, parts[1:5])), dtype=np.float32)
                tvec = np.array(list(map(float, parts[5:8])), dtype=np.float32)
                return qvec, tvec
    raise RuntimeError(f"Image {key} not found in {loc_path}")

# Parse queries.txt for camera intrinsics
def load_intrinsics_from_queries(queries_path: Path, image_name: str):
    key = _norm_name(image_name)
    with queries_path.open("r") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.split()
            if _norm_name(parts[0]) == key:
                if len(parts) < 8:
                    raise RuntimeError(f"Malformed queries line for {key}: {line}")
                _, cam_model, W, H, fx, fy, cx, cy = parts[:8]
                return cam_model, int(float(W)), int(float(H)), float(fx), float(fy), float(cx), float(cy)
    raise RuntimeError(f"Image {key} not found in {queries_path}")

# Initialize default pipeline parameters
def build_default_pipeline():
    import argparse as _argparse
    pipe_parser = _argparse.ArgumentParser(add_help=False)
    pp_group = PipelineParams(pipe_parser)
    pipe_args = pipe_parser.parse_args([])
    pipeline = pp_group.extract(pipe_args)
    return pipeline

def main():
    user_parser = argparse.ArgumentParser(description="Render GS model from HLoc pose")
    user_parser.add_argument("--model_ply", required=True, help="Path to point_cloud.ply")
    user_parser.add_argument("--loc", required=True, help="Path to localization_results.txt")
    user_parser.add_argument("--queries", required=True, help="Path to queries.txt")
    user_parser.add_argument("--image_name", required=True, help="Query image name")
    user_parser.add_argument("--out", default="render_from_hloc.png", help="Output PNG path")
    user_parser.add_argument("--sh_degree", type=int, default=3, help="SH degree")

    args = user_parser.parse_args()

    model_ply = Path(args.model_ply)
    loc_path = Path(args.loc)
    queries_path = Path(args.queries)
    out_path = Path(args.out)
    img_key = _norm_name(args.image_name)

    if not model_ply.is_file():
        raise SystemExit(f"Model .ply not found: {model_ply}")
    if not loc_path.is_file():
        raise SystemExit(f"Localization file not found: {loc_path}")
    if not queries_path.is_file():
        raise SystemExit(f"Queries file not found: {queries_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Image key: {img_key}")

    # Load pose data
    qvec, tvec = load_pose_from_loc(loc_path, img_key)
    print(f"Pose loaded: qvec={qvec}, tvec={tvec}")

    R = qvec2rotmat(qvec)
    T = tvec.astype(np.float32)

    # Load intrinsics
    cam_model, W, H, fx, fy, cx, cy = load_intrinsics_from_queries(queries_path, img_key)
    print(f"Intrinsics: W={W}, H={H}, fx={fx}, fy={fy}")

    FoVx = float(focal2fov(fx, W))
    FoVy = float(focal2fov(fy, H))

    # Initialize Camera object
    resolution = (H, W)
    dummy_pil_image = Image.new("RGB", (W, H), color=(0, 0, 0))
    trans = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    
    cam = Camera(
        resolution,
        colmap_id=-1,
        R=R,
        T=T,
        FoVx=FoVx,
        FoVy=FoVy,
        depth_params=None,
        image=dummy_pil_image,
        invdepthmap=None,
        image_name=img_key,
        uid=999999,
        trans=trans,
        scale=1.0,
        data_device=str(device),
        train_test_exp=False,
        is_test_dataset=False,
        is_test_view=False,
    )

    # Load Gaussian model
    print(f"Loading Gaussian model: {model_ply}")
    gaussians = GaussianModel(args.sh_degree)
    gaussians.load_ply(model_ply)

    # Setup pipeline and background
    pipeline = build_default_pipeline()
    background = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=device)

    # Render image
    print("Rendering...")
    render_pkg = render(cam, gaussians, pipeline, background)
    render_img = render_pkg["render"]

    # Save output
    render_img = render_img.clamp(0.0, 1.0).permute(1, 2, 0).detach().cpu().numpy()
    render_img = (render_img * 255.0 + 0.5).astype(np.uint8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(render_img).save(out_path)
    print(f"Saved rendered image to: {out_path}")

if __name__ == "__main__":
    main()