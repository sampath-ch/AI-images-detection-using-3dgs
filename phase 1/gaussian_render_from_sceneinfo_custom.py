#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys
import numpy as np
import torch
from PIL import Image

from scene.dataset_readers import readColmapSceneInfo
from scene.gaussian_model import GaussianModel
from scene.cameras import Camera
from gaussian_renderer import render
from arguments import PipelineParams

# Initialize default pipeline parameters
def build_default_pipeline():
    import argparse as _argparse
    parser = _argparse.ArgumentParser(add_help=False)
    pp_group = PipelineParams(parser)
    args = parser.parse_args([])
    return pp_group.extract(args)

def main():
    ap = argparse.ArgumentParser("Render GS model from SceneInfo (training-style)")
    ap.add_argument("--dataset_root", required=True, help="Dataset root used for training")
    ap.add_argument("--model_ply", required=True, help="Path to point_cloud.ply")
    ap.add_argument("--image_name", required=True, help="Name of an existing training image")
    ap.add_argument("--out", default="render_sceneinfo_view.png", help="Output PNG")
    ap.add_argument("--sh_degree", type=int, default=3, help="SH degree (default 3)")
    ap.add_argument("--train_test_exp", action="store_true", help="Set if trained with this flag")

    args = ap.parse_args()

    dataset_root = Path(args.dataset_root)
    model_ply = Path(args.model_ply)
    out_path = Path(args.out)
    image_name = Path(args.image_name).name

    if not dataset_root.is_dir():
        raise SystemExit(f"Dataset root not found: {dataset_root}")
    if not model_ply.is_file():
        raise SystemExit(f"Model .ply not found: {model_ply}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Dataset: {dataset_root}")
    print(f"Image: {image_name}")

    # 1. Read full SceneInfo to get training camera parameters
    print("Reading SceneInfo...")
    scene_info = readColmapSceneInfo(
        path=str(dataset_root),
        images=None,
        depths="",
        eval=False,
        train_test_exp=args.train_test_exp,
        llffhold=8,
    )

    nerf_norm = scene_info.nerf_normalization
    print("nerf_normalization:", nerf_norm)

    train_cams = scene_info.train_cameras
    print(f"Train cameras found: {len(train_cams)}")

    # 2. Find the specific camera info for the requested image
    cam_info = None
    for ci in train_cams:
        if ci.image_name == image_name:
            cam_info = ci
            break
            
    if cam_info is None:
        sample_names = [ci.image_name for ci in train_cams[:15]]
        raise SystemExit(f"Image '{image_name}' not found. Examples: {sample_names}")

    print(f"Found CameraInfo: uid={cam_info.uid}, W={cam_info.width}, H={cam_info.height}")

    # 3. Load image and set up Camera object
    im_path = Path(cam_info.image_path)
    if not im_path.is_file():
        raise SystemExit(f"Image file not found: {im_path}")
    pil_img = Image.open(im_path).convert("RGB")

    H, W = cam_info.height, cam_info.width
    FoVx = float(cam_info.FovX)
    FoVy = float(cam_info.FovY)
    R = np.array(cam_info.R, dtype=np.float32)
    T = np.array(cam_info.T, dtype=np.float32)
    depth_params = cam_info.depth_params
    invdepthmap = None
    trans = np.array(nerf_norm.get("translate", [0.0, 0.0, 0.0]), dtype=np.float32)
    scale = float(nerf_norm.get("radius", 1.0))

    print(f"Using trans={trans}, scale={scale}")
    resolution = (W, H)

    cam = Camera(
        resolution,
        colmap_id=cam_info.uid,
        R=R,
        T=T,
        FoVx=FoVx,
        FoVy=FoVy,
        depth_params=depth_params,
        image=pil_img,
        invdepthmap=invdepthmap,
        image_name=cam_info.image_name,
        uid=cam_info.uid,
        trans=trans,
        scale=scale,
        data_device=str(device),
        train_test_exp=args.train_test_exp,
        is_test_dataset=False,
        is_test_view=cam_info.is_test,
    )

    # 4. Load Gaussian model
    print(f"Loading Gaussian model: {model_ply}")
    gaussians = GaussianModel(args.sh_degree)
    gaussians.load_ply(model_ply)

    # 5. Render
    pipeline = build_default_pipeline()
    background = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=device)

    print("Rendering...")
    render_pkg = render(cam, gaussians, pipeline, background)
    img = render_pkg["render"]

    # 6. Save output
    img = img.clamp(0.0, 1.0).permute(1, 2, 0).detach().cpu().numpy()
    img = (img * 255.0 + 0.5).astype(np.uint8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(out_path)
    print(f"Saved rendered image to: {out_path}")

if __name__ == "__main__":
    main()