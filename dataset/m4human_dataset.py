import os
import numpy as np
import lmdb
import msgpack
import smplx

import torch
from torch.utils.data import Dataset

from .data_loader_camera_calibration import (
    calibrate_param_to_radar,
    aabb_from_smplx_params,
    generate_vertices_in_radar_space,
    compute_joints_from_vertices,
)
from .dataset_config_clean import save_idx_to_file, load_idx_to_file
from .lmdb_utils import unpack_ndarray, unpack_dict_np
from .data_loader_Plotting_projection import plot_frames_for_gif_new, project_3d_to_2d

from .m4human_utils import (
    get_all_file_pairs,
    ToTensor,
    process_and_cache,
    ensure_read_envs,
    find_temporal_radar_pc_lmdb,
    find_temporal_radar_image_lmdb,
    pad_radar_pc,
    remove_nan_entries,
)

# Gender information
gender_info = {
    "P1": 0, "P3": 0, "P4": 0, "P7": 0, "P8": 0, "P10": 0, "P13": 0, "P15": 0,
    "P2": 1, "P5": 1, "P6": 1, "P9": 1, "P11": 1, "P12": 1, "P14": 1, "P16": 1,
    "P17": 1, "P18": 1, "P19": 1, "P20": 1
}

NON_VALID_INDICATOR = [
    [1, 12, 58], [1, 13, 491], [1, 31, 324], [1, 36, 284], [1, 37, 43],
    [5, 43, 551], [5, 43, 552], [5, 47, 136], [5, 47, 140]
]


class RF3DPoseDataset(Dataset):
    def __init__(
        self,
        file_pairs,
        transform=None,
        split="train",
        cache_dir="cached_data_test",
        load_save=True,
        smplx_model_path="models",
        use_image=False,
        radar_shape=(121, 111, 31),
        temporal_window=4,
        normalize_flag=True,
        main_modality="radar_points",
        scale_id="p3",
        split_id="s2",
    ):
        assert split in ["train", "val", "test"]
        self.transform = transform
        self.split = split
        self.use_image = use_image
        self.radar_shape = radar_shape
        self.temporal_window = temporal_window
        self.cache_path = os.path.join(cache_dir, "rf3dpose_all")
        os.makedirs(self.cache_path, exist_ok=True)

        if main_modality == "radar_points":
            self.normalize_flag = normalize_flag
        else:
            self.normalize_flag = False

        self._load_smplx_model(smplx_model_path)

        self.env_paths = {
            "radar_comp": os.path.join(self.cache_path, "radar_comp.lmdb"),
            "radar_pc": os.path.join(self.cache_path, "radar_pc.lmdb"),
            "params": os.path.join(self.cache_path, "params.lmdb"),
            "calib": os.path.join(self.cache_path, "calib.lmdb"),
            "indicator": os.path.join(self.cache_path, "indicator.lmdb"),
        }
        if self.use_image:
            self.env_paths["image"] = os.path.join(self.cache_path, "image.lmdb")

        self.lmdb_envs = None
        self._lmdb_owner_pid = None

        lmdb_ready = True
        for p in self.env_paths.values():
            if not os.path.exists(p):
                lmdb_ready = False
                break
            env = lmdb.open(p, readonly=True, subdir=False, lock=False)
            with env.begin() as txn:
                cursor = txn.cursor()
                if not cursor.first():
                    lmdb_ready = False
                    break
            env.close()

        if lmdb_ready and load_save:
            print(f"Dataset loaded Successfully from {self.cache_path} ...")
        else:
            self.lmdb_envs, self._lmdb_owner_pid = ensure_read_envs(
                self.lmdb_envs, self.env_paths, self._lmdb_owner_pid
            )
            process_and_cache(
                file_pairs=file_pairs,
                split=self.split,
                use_image=self.use_image,
                load_save=load_save,
                env_paths=self.env_paths,
                lmdb_envs=self.lmdb_envs,
            )

        self.lmdb_envs, self._lmdb_owner_pid = ensure_read_envs(
            self.lmdb_envs, self.env_paths, self._lmdb_owner_pid
        )

        with self.lmdb_envs["indicator"].begin() as txn:
            self.indicator_list = []
            cursor = txn.cursor()
            for _, value in cursor:
                indicator = msgpack.unpackb(value, raw=False)
                self.indicator_list.append(indicator)

        self.indices_saved_file = os.path.join(self.cache_path, "indeces.pkl.gz")
        if os.path.exists(self.indices_saved_file) is False:
            save_idx_to_file(self.indicator_list, path=self.indices_saved_file)
            split_indicator_lists = load_idx_to_file(path=self.indices_saved_file)
        else:
            split_indicator_lists = load_idx_to_file(path=self.indices_saved_file)

        self.indicator_list = split_indicator_lists[scale_id][split_id][self.split]

        print(f"Loaded {self.split} Dataset with length {len(self.indicator_list)}.")
        test_subs = sorted({s for s, a, f in self.indicator_list})
        test_acts = sorted({a for s, a, f in self.indicator_list})
        print(f"Unique sub in {self.split}:", test_subs)
        print(f"Unique act in {self.split}:", test_acts)

        self.indicator_list = remove_nan_entries(
            indicator_list=self.indicator_list,
            lmdb_envs=self.lmdb_envs,
            use_image=self.use_image,
            radar_shape=self.radar_shape,
            non_valid_indicator=NON_VALID_INDICATOR,
            rescan=False,
        )

    def _load_smplx_model(self, model_path):
        self.smplx_models = {}
        for gender in ["neutral", "male", "female"]:
            self.smplx_models[gender] = smplx.create(
                model_path=model_path,
                model_type="smplx",
                gender=gender,
            )
        self.faces = self.smplx_models["neutral"].faces

    def __getitem__(self, idx):
        self.lmdb_envs, self._lmdb_owner_pid = ensure_read_envs(
            self.lmdb_envs, self.env_paths, self._lmdb_owner_pid
        )

        indicator = self.indicator_list[idx]

        # Single-frame supervision branch (same behavior as dataset_mmMesh2).
        with self.lmdb_envs["params"].begin() as txn_param, \
            self.lmdb_envs["calib"].begin() as txn_calib:
            parameter = unpack_dict_np(txn_param.get(str(indicator).encode()))
            calibration = unpack_dict_np(txn_calib.get(str(indicator).encode()))

        if self.use_image:
            with self.lmdb_envs["image"].begin() as txn_img:
                image_data = txn_img.get(str(indicator).encode())
                image = unpack_ndarray(image_data)
        else:
            image = None

        radar_PC_selected, indicator_selected = find_temporal_radar_pc_lmdb(
            indicator_list=self.indicator_list,
            lmdb_envs=self.lmdb_envs,
            current_index=idx,
            T=self.temporal_window,
        )
        radar_image_selected, _ = find_temporal_radar_image_lmdb(
            indicator_list=self.indicator_list,
            lmdb_envs=self.lmdb_envs,
            current_index=idx,
            radar_shape=self.radar_shape,
            T=self.temporal_window,
        )
        radar_image_selected = np.stack(radar_image_selected, axis=0)

        for i in range(len(radar_PC_selected)):
            if self.normalize_flag:
                radar_PC_selected[i][..., 2] = radar_PC_selected[i][..., 2] - 1.5
            radar_PC_selected[i] = pad_radar_pc(radar_PC_selected[i], target_points=1000)
        radar_PC_seq = np.stack(radar_PC_selected, axis=0)

        parameter["gender"] = gender_info[f"P{indicator[0]}"]
        parameter_radar = calibrate_param_to_radar(parameter, calibration)
        if self.normalize_flag:
            parameter_radar["trans"][..., 2] = parameter_radar["trans"][..., 2] - 1.5

        radar_coords = generate_vertices_in_radar_space(self.smplx_models, parameter_radar)
        joints_root = compute_joints_from_vertices(self.smplx_models, radar_coords, parameter_radar["gender"])
        bbbox = aabb_from_smplx_params(parameter_radar)

        camera_extrinsics = np.eye(4)
        camera_extrinsics[:3, :3] = calibration["radar_to_cam_rotmatrix"]
        camera_extrinsics[:3, 3] = calibration["radar_to_cam_tvec"]
        projected_vertices = project_3d_to_2d(radar_coords / 1000)

        sample = {
            "rawImage_XYZ": radar_image_selected,
            "vertices": radar_coords,
            "bbbox": bbbox,
            "projected_vertices": projected_vertices,
            "parameter": parameter_radar,
            "calibration": calibration,
            "indicator": indicator,
            "radar_points": radar_PC_seq,
            "joints_root": joints_root,
        }

        if self.use_image:
            sample["image"] = image

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.indicator_list)


def test_data_loader(dataset):
    plot_frames_for_gif_new(dataset)


if __name__ == "__main__":
    root_dir = "/media/jiarui/HDD/Dataset/Edinburgh_mmwave"
    all_pairs = get_all_file_pairs(root_dir=root_dir, gender_info=gender_info)

    indicator_target = [2, 50, 100]
    for i in range(len(all_pairs)):
        if all_pairs[i][-2][0] == indicator_target[0] and all_pairs[i][-2][1] == indicator_target[1] and all_pairs[i][-2][2] == indicator_target[2]:
            s = i
            print(f"Found target indicator at index {s}: {all_pairs[i][-2]}")
            break

    file_pairs = all_pairs
    print(f"Total frames: {len(file_pairs)}")
    dataset = RF3DPoseDataset(file_pairs, transform=ToTensor(), load_save=True, use_image=False)
    print("Dataset loaded.")
    test_data_loader(dataset)