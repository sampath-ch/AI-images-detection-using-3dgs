#!/usr/bin/env python3
import argparse
from pathlib import Path
import traceback
import sys
import numpy as np

from hloc import extract_features, match_features, localize_sfm, pairs_from_exhaustive
from hloc.utils import read_write_model

# Read COLMAP model using .bin or .txt extension
def read_colmap_model(model_path: Path):
    try:
        cams, images, points3D = read_write_model.read_model(model_path, ext=".bin")
        return cams, images, points3D
    except Exception:
        cams, images, points3D = read_write_model.read_model(model_path, ext=".txt")
        return cams, images, points3D

# Get the number of 3D points per image from the COLMAP model
def build_points3d_len_by_name(model_path: Path):
    _, images, _ = read_colmap_model(model_path)
    length_by_name = {}
    for img in images.values():
        pts_ids = getattr(img, "point3D_ids", None)
        if pts_ids is None:
            pts_ids = getattr(img, "points3D_ids", None)
        if pts_ids is None:
            continue
        length_by_name[img.name] = int(len(pts_ids))
    return length_by_name

# Attempt to get image dimensions from H5 features file
def get_image_size_from_features(features_h5_path: Path, image_name: str):
    try:
        import h5py
    except Exception:
        return None
    try:
        with h5py.File(features_h5_path, "r") as f:
            if image_name in f and "image_size" in f[image_name]:
                arr = f[image_name]["image_size"][()]
                return int(arr[0]), int(arr[1])
    except Exception:
        pass
    return None

# Attempt to get image dimensions from the raw file
def get_image_size_from_jpeg(image_path: Path):
    try:
        from PIL import Image
        with Image.open(image_path) as im:
            w, h = im.size
            return int(w), int(h)
    except Exception:
        return None

# Generate queries.txt with estimated pinhole intrinsics
def build_queries_file(output_path: Path, query_image: Path, features_path: Path) -> Path:
    queries_txt = output_path / "queries.txt"
    name = query_image.name

    size = get_image_size_from_features(features_path, name)
    if size is None:
        size = get_image_size_from_jpeg(query_image)

    if size is None:
        width, height = 1024, 768
    else:
        width, height = size

    maxdim = max(width, height)
    fx = fy = int(0.9 * maxdim)
    cx = width / 2.0
    cy = height / 2.0

    line = f"{name} PINHOLE {width} {height} {fx} {fy} {cx} {cy}"
    queries_txt.write_text(line + "\n")
    print(f"Created queries file: {queries_txt}")
    return queries_txt

# invalid match indices (>= 3D point count) to -1
def clip_matches_by_points3d(model_path: Path, matches_path: Path, clipped_out: Path):
    import h5py

    length_by_name = build_points3d_len_by_name(model_path)
    print(f"Loaded points3D lengths for {len(length_by_name)} images.")

    with h5py.File(matches_path, "r") as fin, \
         h5py.File(clipped_out, "w") as fout:

        # Copy attributes
        for k, v in fin.attrs.items():
            try:
                fout.attrs[k] = v
            except Exception:
                pass

        for qname in fin.keys():
            qgroup_in = fin[qname]
            qgroup_out = fout.create_group(qname)
            
            for db_name in qgroup_in.keys():
                src_grp = qgroup_in[db_name]
                out_grp = qgroup_out.create_group(db_name)

                if "matches0" not in src_grp:
                    out_grp.create_dataset("matches0", data=np.full((0,), -1, dtype=np.int16))
                    if "matching_scores0" in src_grp:
                        out_grp.create_dataset("matching_scores0", data=np.zeros((0,), dtype=np.float32))
                    continue

                matches_arr = src_grp["matches0"][()]
                dtype0 = matches_arr.dtype
                matches_arr = np.array(matches_arr, copy=True)

                # Clip indices that exceed the valid range in the COLMAP model
                L = length_by_name.get(db_name, None)
                if L is not None:
                    invalid = matches_arr >= L
                    if np.any(invalid):
                        matches_arr[invalid] = -1

                out_grp.create_dataset("matches0", data=matches_arr.astype(dtype0))
                if "matching_scores0" in src_grp:
                    scores_arr = src_grp["matching_scores0"][()]
                    out_grp.create_dataset("matching_scores0", data=scores_arr)

    print(f"Clipped matches written to: {clipped_out}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True, help='Path to query image')
    parser.add_argument('--model', required=True, help='Path to COLMAP sparse/0 folder')
    parser.add_argument('--dataset_images', required=True, help='Path to original training images folder')
    parser.add_argument('--output', default='outputs_superpoint', help='Output folder')
    args = parser.parse_args()

    query_image = Path(args.image)
    model_path = Path(args.model)
    dataset_images = Path(args.dataset_images)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    feature_path = output_path / "features.h5"
    match_path = output_path / "matches.h5"
    pairs_path = output_path / "pairs.txt"
    results_path = output_path / "localization_results.txt"
    clipped_matches_path = output_path / "matches_clipped.h5"

    print(f"Query: {query_image}")
    print(f"Model: {model_path}")
    print(f"Output: {output_path}")

    # 1. Read COLMAP model to get database image names
    try:
        _, images_db, _ = read_colmap_model(model_path)
        db_name_list = [img.name for img in images_db.values()]
        print(f"Model loaded: {len(images_db)} images.")
    except Exception as e:
        print(f"Error reading model: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 2. Extract features for database and query images
    print("Extracting SuperPoint features...")
    conf_extract = extract_features.confs['superpoint_aachen']
    extract_features.main(
        conf_extract,
        dataset_images,
        feature_path=feature_path,
        image_list=db_name_list
    )

    extract_features.main(
        conf_extract,
        query_image.parent,
        feature_path=feature_path,
        image_list=[query_image.name],
        overwrite=False
    )

    # 3. Generate retrieval pairs
    print("Generating pairs...")
    pairs_from_exhaustive.main(
        pairs_path,
        image_list=[query_image.name],
        ref_list=db_name_list
    )

    # 4. Match features using SuperGlue
    print("Matching features...")
    conf_match = match_features.confs['superglue']
    match_features.main(
        conf_match,
        pairs_path,
        features=feature_path,
        matches=match_path
    )

    # 5. Create queries file
    print("Building queries.txt...")
    queries_file = build_queries_file(output_path, query_image, feature_path)

    # 6. Clip matches to ensure validity against 3D model
    print("Clipping matches...")
    clip_matches_by_points3d(model_path, match_path, clipped_matches_path)

    # 7. Run localization
    print("Localizing...")
    try:
        logs = localize_sfm.main(
            model_path,
            queries_file,
            pairs_path,
            feature_path,
            clipped_matches_path,
            results_path,
            covisibility_clustering=False
        )

        pose = None
        image_name = query_image.name

        if isinstance(logs, dict):
            if len(logs) > 0:
                image_name, pose = next(iter(logs.items()))
        elif isinstance(logs, (list, tuple)):
            if len(logs) > 0:
                pose = logs[0]
                if isinstance(pose, dict):
                    image_name = pose.get("name", image_name)

        if pose is None:
            print("Could not obtain pose from localization output.")
            return

        # Extract rotation and translation
        if isinstance(pose, dict):
            qvec = pose.get("qvec", pose.get("rotation", "N/A"))
            tvec = pose.get("tvec", pose.get("translation", "N/A"))
        else:
            qvec = getattr(pose, "qvec", getattr(pose, "rotation", "N/A"))
            tvec = getattr(pose, "tvec", getattr(pose, "translation", "N/A"))

        print(f"Pose for {image_name}:")
        print(f"Qvec: {qvec}")
        print(f"Tvec: {tvec}")

        try:
            with open(results_path, "a") as f:
                f.write(f"{image_name}\nQ: {qvec}\nT: {tvec}\n\n")
            print(f"Result saved to: {results_path}")
        except Exception:
            pass

    except Exception:
        print("Exception during localization:")
        traceback.print_exc()
        return

if __name__ == "__main__":
    main()