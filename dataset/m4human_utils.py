import os
import re
import copy

import lmdb
import msgpack
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from .data_loader_Load_data import load_json_data, load_mat_file, load_transformation_matrices
from .lmdb_utils import (
    extract_frame_index,
    save_to_lmdb,
    pack_ndarray,
    pack_dict_np,
    unpack_ndarray,
    unpack_dict,
)


def get_all_file_pairs(
    root_dir,
    gender_info,
    select_p=None,
):
    if select_p is None:
        select_p = [
            "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10",
            "p11", "p12", "p13", "p14", "p15", "p16", "p17", "p18", "p19", "p20",
        ]

    all_file_pairs = []
    sub_dirs = [
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d)) and re.match(r"p\d+", d) and d in select_p
    ]

    for sub_dir in sorted(sub_dirs, key=lambda x: int(re.findall(r"\d+", x)[0])):
        sub_dir_num = int(re.findall(r"\d+", sub_dir)[0])

        mat_root_dir = os.path.join(root_dir, sub_dir, "actions")
        parameters_dir = os.path.join(root_dir, sub_dir, "parameters")

        print(f"Checking directory {sub_dir}...")

        if not (os.path.exists(mat_root_dir) and os.path.exists(parameters_dir)):
            print(f"Skipping {sub_dir}: missing folders.")
            continue

        for i in range(1, 51):
            json_path = os.path.join(parameters_dir, f"{i}.json")
            if not os.path.exists(json_path):
                continue

            mat_dir = os.path.join(mat_root_dir, str(i), "mmwave")
            rgb_dir = os.path.join(mat_root_dir, str(i), "rgb")

            if not (os.path.exists(mat_dir) and os.path.exists(rgb_dir)):
                continue

            mat_files = {
                extract_frame_index(f): os.path.join(mat_dir, f)
                for f in os.listdir(mat_dir) if f.endswith(".mat")
            }
            png_files = {
                extract_frame_index(f): os.path.join(rgb_dir, f)
                for f in os.listdir(rgb_dir) if f.endswith(".png")
            }

            common_frames = sorted(set(mat_files) & set(png_files), key=int)
            union_frames = sorted(set(mat_files) | set(png_files), key=int)

            for new_frame_idx, frame_key in enumerate(union_frames):
                if frame_key not in common_frames:
                    continue

                indicator = [sub_dir_num, i, new_frame_idx]
                all_file_pairs.append((
                    mat_files[frame_key],
                    png_files[frame_key],
                    json_path,
                    new_frame_idx,
                    indicator,
                    gender_info[sub_dir.capitalize()],
                ))

    all_file_pairs.sort(key=lambda x: x[4])
    return all_file_pairs


class ToTensor(object):
    def __call__(self, sample):
        def convert_to_tensor(value, key=None, sub_key=None):
            try:
                if value is None:
                    return None
                elif isinstance(value, np.ndarray):
                    return torch.tensor(value).float()
                elif isinstance(value, (int, float)):
                    return torch.tensor([value]).float()
                elif isinstance(value, dict):
                    return {k: convert_to_tensor(v, key=key, sub_key=k) for k, v in value.items()}
                elif isinstance(value, (list, tuple)):
                    return [convert_to_tensor(elem, key=f"{key}[{idx}]") for idx, elem in enumerate(value)]
                return value
            except Exception as e:
                name = f"{key}.{sub_key}" if sub_key else key
                print(f"[convert_to_tensor] Failed to convert '{name}'")
                print(f"  Type: {type(value)}, Dtype: {getattr(value, 'dtype', 'N/A')}, Value: {value}")
                raise e

        new_sample = {}
        for key, value in sample.items():
            try:
                if value is None:
                    new_sample[key] = None
                    continue

                if key == "image":
                    new_sample[key] = torch.tensor(value).float().permute(2, 0, 1)

                elif key == "indicator":
                    new_sample[key] = torch.tensor(value).long()

                else:
                    new_sample[key] = convert_to_tensor(value, key)

            except Exception as e:
                print(f"[ToTensor] Error processing key: '{key}'")
                raise e

        return new_sample


def compress_radar(radar_image):
    radar_image = np.nan_to_num(radar_image, nan=0.0)
    indices = np.array(np.nonzero(radar_image)).T
    values = radar_image[tuple(indices.T)]
    compressed = np.concatenate([indices, values[:, None]], axis=1)
    return compressed


def decompress_radar(compressed, radar_shape):
    indices = compressed[:, :3].astype(np.int64)
    values = compressed[:, 3]
    radar_image = np.zeros(radar_shape, dtype=values.dtype)
    radar_image[indices[:, 0], indices[:, 1], indices[:, 2]] = values
    return radar_image


def pad_radar_pc(radar_pc, target_points=1000):
    P, D = radar_pc.shape
    assert D == 4, "Each point must have 4 values (x, y, z, intensity)"

    padded_pc = np.zeros((target_points, D), dtype=radar_pc.dtype)

    if P >= target_points:
        idx = np.random.choice(P, target_points, replace=False)
        padded_pc = radar_pc[idx]
    else:
        padded_pc[:P] = radar_pc

    return padded_pc


def ensure_read_envs(lmdb_envs, env_paths, owner_pid):
    cur_pid = os.getpid()

    if owner_pid != cur_pid or lmdb_envs is None:
        if lmdb_envs:
            for ev in lmdb_envs.values():
                try:
                    ev.close()
                except Exception:
                    pass

        lmdb_envs = {}
        for name, path in env_paths.items():
            subdir_flag = os.path.isdir(path)
            lmdb_envs[name] = lmdb.open(
                path,
                readonly=True,
                lock=False,
                readahead=False,
                max_readers=4096,
                subdir=subdir_flag,
            )
        owner_pid = cur_pid

    return lmdb_envs, owner_pid


def process_and_cache(
    file_pairs,
    split,
    use_image,
    load_save,
    env_paths,
    lmdb_envs,
):
    print(f"Processing and caching {split} dataset...")

    writer_envs = {
        "radar_comp": lmdb.open(env_paths["radar_comp"], map_size=1 << 40, max_readers=4096, subdir=False),
        "radar_pc": lmdb.open(env_paths["radar_pc"], map_size=1 << 40, max_readers=4096, subdir=False),
        "params": lmdb.open(env_paths["params"], map_size=1 << 40, max_readers=4096, subdir=False),
        "calib": lmdb.open(env_paths["calib"], map_size=1 << 40, max_readers=4096, subdir=False),
        "indicator": lmdb.open(env_paths["indicator"], map_size=1 << 40, max_readers=4096, subdir=False),
    }
    if use_image:
        writer_envs["image"] = lmdb.open(env_paths["image"], map_size=1 << 40, max_readers=4096, subdir=False)

    for entry in tqdm(file_pairs, desc=f"Processing {split}"):
        mat_file, png_file, json_path, json_idx, indicator, gender = entry
        radar_data = load_mat_file(mat_file)
        json_data = load_json_data(json_path)
        if radar_data is None or json_data is None:
            print("Radar data or json data is None!")
            continue

        image = None
        if use_image and png_file and os.path.exists(png_file):
            try:
                image = plt.imread(png_file)
            except Exception as e:
                print(f"Error loading image {png_file}: {e}")
                continue

        ins_ext_dir = os.path.join(os.path.dirname(json_path), "../ins_ext")
        cam_ins, vicon_to_cam_R, vicon_to_cam_T, radar_to_cam_R, radar_to_cam_T = load_transformation_matrices(
            ins_ext_dir, indicator
        )
        if any(x is None for x in [cam_ins, vicon_to_cam_R, vicon_to_cam_T, radar_to_cam_R, radar_to_cam_T]):
            print("Calibration has error")
            continue

        parameter = {
            "joints": np.array(json_data["joints"][json_idx]),
            "betas": np.array(json_data["betas"][json_idx])[:10],
            "pose_body": np.array(json_data["pose_body"][json_idx]),
            "trans": np.array(json_data["trans"][json_idx]),
            "root_orient": np.array(json_data["root_orient"][json_idx]),
            "gender": gender,
        }

        calibration = {
            "cam_intrinsic": cam_ins,
            "vicon_to_cam_rotmatrix": vicon_to_cam_R,
            "vicon_to_cam_tvec": vicon_to_cam_T,
            "radar_to_cam_rotmatrix": radar_to_cam_R,
            "radar_to_cam_tvec": radar_to_cam_T,
        }

        compressed = compress_radar(radar_data["rawImage_XYZ"])
        radar_PC = radar_data["RawPoints"]

        if load_save:
            save_to_lmdb(lmdb_envs["radar_comp"], str(indicator), pack_ndarray(compressed))
            save_to_lmdb(lmdb_envs["radar_pc"], str(indicator), pack_ndarray(radar_PC))
            save_to_lmdb(lmdb_envs["params"], str(indicator), pack_dict_np(parameter))
            save_to_lmdb(lmdb_envs["calib"], str(indicator), pack_dict_np(calibration))
            save_to_lmdb(lmdb_envs["indicator"], str(indicator), msgpack.packb(indicator))
            if use_image and image is not None:
                save_to_lmdb(lmdb_envs["image"], str(indicator), pack_ndarray(image))

    for ev in writer_envs.values():
        ev.close()

    print(f"Finished caching {split} dataset.")


def find_temporal_radar_pc_lmdb(indicator_list, lmdb_envs, current_index, T=6):
    indicator = indicator_list[current_index]
    sub_id, action_id, current_frame = indicator
    radar_pc_selected = []
    indicator_selected = []

    last_pc = None
    last_indicator = indicator
    expected_frame = current_frame

    for i in range(current_index, -1, -1):
        s, a, f = indicator_list[i]
        if s == sub_id and a == action_id and f == expected_frame:
            with lmdb_envs["radar_pc"].begin() as txn_pc:
                radar_pc_data = txn_pc.get(str(indicator_list[i]).encode())
            last_pc = unpack_ndarray(radar_pc_data)
            last_indicator = indicator_list[i]

        radar_pc_selected.append(copy.deepcopy(last_pc))
        indicator_selected.append(last_indicator)
        expected_frame -= 1
        if len(radar_pc_selected) == T:
            break
        if s != sub_id or a != action_id or f < current_frame - T + 1:
            break

    while len(radar_pc_selected) < T:
        radar_pc_selected.append(copy.deepcopy(last_pc) if last_pc is not None else None)
        indicator_selected.append(last_indicator)

    return radar_pc_selected[::-1], indicator_selected[::-1]


def find_temporal_radar_image_lmdb(indicator_list, lmdb_envs, current_index, radar_shape, T=6):
    indicator = indicator_list[current_index]
    sub_id, action_id, current_frame = indicator
    radar_image_selected = []
    indicator_selected = []

    last_image = None
    last_indicator = indicator
    expected_frame = current_frame

    for i in range(current_index, -1, -1):
        s, a, f = indicator_list[i]
        if s == sub_id and a == action_id and f == expected_frame:
            with lmdb_envs["radar_comp"].begin() as txn_comp:
                radar_comp_data = txn_comp.get(str(indicator_list[i]).encode())
            last_image = decompress_radar(unpack_ndarray(radar_comp_data), radar_shape)
            last_indicator = indicator_list[i]

        radar_image_selected.append(copy.deepcopy(last_image))
        indicator_selected.append(last_indicator)
        expected_frame -= 1
        if len(radar_image_selected) == T:
            break
        if s != sub_id or a != action_id or f < current_frame - T + 1:
            break

    while len(radar_image_selected) < T:
        radar_image_selected.append(copy.deepcopy(last_image) if last_image is not None else None)
        indicator_selected.append(last_indicator)

    return radar_image_selected[::-1], indicator_selected[::-1]


def remove_nan_entries(
    indicator_list,
    lmdb_envs,
    use_image,
    radar_shape,
    non_valid_indicator,
    rescan=False,
):
    def has_nan_or_inf(obj):
        if isinstance(obj, dict):
            return any(has_nan_or_inf(v) for v in obj.values())
        elif isinstance(obj, (list, tuple)):
            return any(has_nan_or_inf(v) for v in obj)
        elif isinstance(obj, np.ndarray):
            return np.isnan(obj).any() or np.isinf(obj).any()
        elif isinstance(obj, torch.Tensor):
            return torch.isnan(obj).any().item() or torch.isinf(obj).any().item()
        return False

    nonvalid_indicators = []
    total = len(indicator_list)

    if rescan:
        for idx in tqdm(range(total), desc="Filtering NaN/Inf entries"):
            indicator = indicator_list[idx]
            reasons = []
            to_remove = False

            with lmdb_envs["radar_comp"].begin() as txn_comp, \
                lmdb_envs["radar_pc"].begin() as txn_pc, \
                lmdb_envs["params"].begin() as txn_param, \
                lmdb_envs["calib"].begin() as txn_calib:
                radar_comp_data = txn_comp.get(str(indicator).encode())
                radar_pc_data = txn_pc.get(str(indicator).encode())
                param_data = txn_param.get(str(indicator).encode())
                calib_data = txn_calib.get(str(indicator).encode())

            radar_comp = unpack_ndarray(radar_comp_data)
            radar_pc = unpack_ndarray(radar_pc_data)
            parameter = unpack_dict(param_data)
            calibration = unpack_dict(calib_data)

            radar_image = decompress_radar(radar_comp, radar_shape)
            radar_PC = radar_pc

            if np.allclose(radar_image, 0) or np.allclose(radar_PC, 0):
                to_remove = True
                reasons.append("decompressed radar is all zeros")
            if has_nan_or_inf(radar_image) or has_nan_or_inf(radar_PC):
                to_remove = True
                reasons.append("decompressed radar contains NaN or Inf")
            if np.abs(radar_image).max() > 1e3:
                to_remove = True
                reasons.append("decompressed radar has large absolute value (>1e3)")
            if np.abs(radar_PC).max() > 1e3:
                to_remove = True
                reasons.append("radar_PC has large absolute value (>1e3)")

            fields = [
                ("parameter", parameter),
                ("calibration", calibration),
                ("indicator", indicator),
            ]
            if use_image:
                with lmdb_envs["image"].begin() as txn_img:
                    image_data = txn_img.get(str(indicator).encode())
                if image_data is not None:
                    image = unpack_ndarray(image_data)
                    fields.append(("image", image))

            for name, value in fields:
                if has_nan_or_inf(value):
                    to_remove = True
                    reasons.append(f"{name} contains NaN or Inf")

            if to_remove:
                nonvalid_indicators.append(indicator)
                print(f"Removing sample indicator {indicator} due to: {', '.join(reasons)}")
                continue
    else:
        nonvalid_indicators = non_valid_indicator

    invalid_set = {tuple(c1) for c1 in nonvalid_indicators}
    valid_indicators = [c for c in indicator_list if tuple(c) not in invalid_set]
    print(f"Kept {len(valid_indicators)} / {total} samples after filtering.")
    return valid_indicators