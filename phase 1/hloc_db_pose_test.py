#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np

from hloc import extract_features, match_features, pairs_from_exhaustive, localize_sfm
from hloc.utils import read_write_model

# Calculate angle between two rotation matrices in degrees
def rotation_error_deg(R1, R2):
    R = R1 @ R2.T
    trace = np.clip(np.trace(R), -1.0, 3.0)
    cos_angle = (trace - 1.0) / 2.0
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))

# Parse localization_results.txt for qvec and tvec
def parse_hloc_loc(loc_path: Path, image_name: str):
    with loc_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            name = parts[0]
            if name == image_name:
                q = np.array(list(map(float, parts[1:5])), dtype=np.float64)
                t = np.array(list(map(float, parts[5:8])), dtype=np.float64)
                return q, t
    raise RuntimeError(f"Image {image_name} not found in {loc_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--colmap_model", required=True, help="Path to COLMAP sparse/0 folder")
    parser.add_argument("--images_dir", required=True, help="Path to images folder")
    parser.add_argument("--image_name", required=True, help="Name of a DB image")
    parser.add_argument("--output", default="outputs_db_test", help="Output directory")
    args = parser.parse_args()

    sfm_dir = Path(args.colmap_model)
    images_dir = Path(args.images_dir)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Read COLMAP model to find Ground Truth pose
    try:
        cameras, images, points3D = read_write_model.read_model(sfm_dir, ext=".bin")
    except Exception:
        cameras, images, points3D = read_write_model.read_model(sfm_dir, ext=".txt")

    img_entry = None
    for im in images.values():
        if im.name == args.image_name:
            img_entry = im
            break

    if img_entry is None:
        raise RuntimeError(f"{args.image_name} not found in COLMAP images")

    cam = cameras[img_entry.camera_id]

    q_gt = img_entry.qvec.astype(np.float64)
    t_gt = img_entry.tvec.astype(np.float64)
    R_gt = read_write_model.qvec2rotmat(q_gt)
    C_gt = -R_gt.T @ t_gt  # Camera center in world coordinates

    print("Ground-truth (COLMAP) pose:")
    print(f"q_gt: {q_gt}")
    print(f"t_gt: {t_gt}")
    print(f"C_gt: {C_gt}")
    print(f"Camera: {cam.model}, W={cam.width}, H={cam.height}")

    # 2. Write queries_db.txt using exact COLMAP intrinsics
    queries_path = out_dir / "queries_db.txt"
    params = cam.params

    if cam.model == "PINHOLE" and len(params) >= 4:
        fx, fy, cx, cy = params[:4]
        model_name = "PINHOLE"
    else:
        fx, fy, cx, cy = params[0], params[1], params[2], params[3]
        model_name = cam.model
        print("Warning: Non-PINHOLE model; using first 4 params.")

    with queries_path.open("w") as f:
        f.write("# name model width height params...\n")
        f.write(f"{args.image_name} {model_name} {cam.width} {cam.height} {fx} {fy} {cx} {cy}\n")

    print(f"Wrote queries file: {queries_path}")

    # 3. Run HLoc pipeline
    feature_path = out_dir / "features.h5"
    match_path = out_dir / "matches.h5"
    pairs_path = out_dir / "pairs.txt"
    results_path = out_dir / "localization_results_db.txt"

    conf_extract = extract_features.confs["superpoint_aachen"]
    conf_match = match_features.confs["superglue"]

    print("Extracting features...")
    extract_features.main(
        conf_extract,
        images_dir,
        feature_path=feature_path,
        image_list=None,
    )

    print("Building pairs...")
    db_names = [im.name for im in images.values()]
    pairs_from_exhaustive.main(
        pairs_path,
        image_list=[args.image_name],
        ref_list=db_names,
    )

    print("Matching features...")
    match_features.main(
        conf_match,
        pairs_path,
        features=feature_path,
        matches=match_path,
    )

    print("Localizing...")
    localize_sfm.main(
        sfm_dir,
        queries_path,
        pairs_path,
        feature_path,
        match_path,
        results_path,
        covisibility_clustering=False,
    )
    print(f"Localization complete: {results_path}")

    # 4. Compare HLoc pose to COLMAP GT
    q_hloc, t_hloc = parse_hloc_loc(results_path, args.image_name)
    R_hloc = read_write_model.qvec2rotmat(q_hloc)

    # Calculate camera centers for both conventions
    # Case A: t is translation (world -> cam)
    T_A = t_hloc
    C_A = -R_hloc.T @ T_A

    # Case B: t is camera center in world coords
    C_B = t_hloc
    T_B = -R_hloc @ C_B

    print("HLoc raw pose:")
    print(f"q_hloc: {q_hloc}")
    print(f"t_hloc: {t_hloc}")

    rot_err = rotation_error_deg(R_hloc, R_gt)
    trans_err_A = np.linalg.norm(C_A - C_gt)
    trans_err_B = np.linalg.norm(C_B - C_gt)

    print("Pose error (Case A: t = translation):")
    print(f"C_A: {C_A}")
    print(f"Rotation error: {rot_err:.4f} deg")
    print(f"Translation error: {trans_err_A:.4f}")

    print("Pose error (Case B: t = camera center):")
    print(f"C_B: {C_B}")
    print(f"Rotation error: {rot_err:.4f} deg")
    print(f"Translation error: {trans_err_B:.4f}")

    print("Note: The case with smaller translation error is likely correct.")

if __name__ == "__main__":
    main()