import argparse
import os
import math
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from pose_estimation.file_utils import get_checkpoint_arguments
from pose_estimation.identification_module import IdentificationModule
from pose_estimation.sampling import generate_all_possible_rays
from pose_estimation.test import test_pose_estimation
from pose_estimation.distance_based_loss import DistanceBasedScoreLoss
from scene import GaussianModel, load_data

def load_gaussian_model(ply_path: str, sh_degree: int, device: str):
    model = GaussianModel(sh_degree)
    model.load_ply(ply_path)
    model = model.to(device)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model

def main():
    parser = argparse.ArgumentParser(description="Estimate camera pose from a single query image using 6DGS.")
    parser.add_argument("--exp_dir", type=str, required=True, help="Path to 3DGS experiment directory")
    parser.add_argument("--ply_path", type=str, required=True, help="Path to point_cloud.ply")
    parser.add_argument("--query_image", type=str, required=True, help="Path to query image")
    parser.add_argument("--fov_deg", type=float, default=60.0, help="Horizontal FOV in degrees")
    parser.add_argument("--out_pose", type=str, required=True, help="Output path for c2w .npy")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load experiment arguments and model
    checkpoint_args = get_checkpoint_arguments(args.exp_dir)

    print(f"Loading Gaussian model: {args.ply_path}")
    gs_model = load_gaussian_model(
        args.ply_path, sh_degree=checkpoint_args.sh_degree, device=device
    )

    print(f"Loading scene data: {checkpoint_args.source_path}")
    scene_info = load_data(checkpoint_args)

    # 2. Load Identification Module
    id_module_ckpt_path = os.path.join(args.exp_dir, "id_module.th")
    if not os.path.exists(id_module_ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {id_module_ckpt_path}")

    print(f"Loading IdentificationModule: {id_module_ckpt_path}")
    backbone_type = "dino"
    id_module = IdentificationModule(backbone_type=backbone_type).to(device)
    ckpt = torch.load(id_module_ckpt_path, map_location=device)
    id_module.load_state_dict(ckpt["model_state_dict"])
    id_module.eval()

    # 3. Sample rays from Gaussian model
    print("Sampling rays...")
    with torch.no_grad():
        rays_ori, rays_dirs, rays_rgb = generate_all_possible_rays(
            gs_model,
            sample_quadricell_targets=50,
        )

    # 4. Compute scene up vector
    model_up_np = np.mean(
        np.asarray(
            [train_camera.R[:3, 1] for train_camera in scene_info.train_cameras],
            dtype=np.float32,
        ),
        axis=0,
    )
    model_up = torch.from_numpy(model_up_np).to(device=device, non_blocking=True)

    # 5. Prepare query camera object
    print(f"Loading query image: {args.query_image}")
    pil_img = Image.open(args.query_image).convert("RGBA")
    width, height = pil_img.size

    template_cam = scene_info.train_cameras[0]
    fov_rad = math.radians(args.fov_deg)

    # Create namespace compatible with test_pose_estimation
    query_cam = SimpleNamespace(
        R=template_cam.R,
        T=template_cam.T,
        FovX=fov_rad,
        FovY=fov_rad,
        width=width,
        height=height,
        image=pil_img,
    )

    # 6. Run estimation
    loss_fn = DistanceBasedScoreLoss()
    cameras_info = [query_cam]

    print("Running 6DGS pose estimation...")
    with torch.no_grad():
        results, _, _, _, _ = test_pose_estimation(
            cameras_info,
            id_module,
            rays_ori,
            rays_dirs,
            rays_rgb,
            model_up,
            sequence_id="brandenburg",
            category_id="brandenburg",
            loss_fn=loss_fn,
            save=False,
            save_all=False,
        )

    if len(results) == 0:
        raise RuntimeError("No result returned for query image.")

    pred_c2w = np.array(results[0]["pred_c2w"], dtype=np.float32)
    print("Predicted c2w matrix:")
    print(pred_c2w)

    # 7. Save result
    out_dir = os.path.dirname(args.out_pose)
    if out_dir != "":
        os.makedirs(out_dir, exist_ok=True)
    np.save(args.out_pose, pred_c2w)
    print(f"Saved predicted pose to: {args.out_pose}")

if __name__ == "__main__":
    main()