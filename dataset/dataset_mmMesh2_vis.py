import logging
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from .data_loader_camera_calibration import *
from .data_loader_Load_data import load_json_data, load_mat_file, load_transformation_matrices, gender_info
from .data_loader_Plotting_projection import plot_random_samples,project_to_2d, plot_frames_for_gif,project_3d_to_2d1, plot_frames_for_gif_new, normalize_img_patch, plot_frames_for_gif_depth



import smplx
from tqdm import tqdm
from collections import defaultdict
import random
from typing import List, Tuple, Union, Dict, Any
from collections import Counter, defaultdict
from .dataset_config import *
from .lmdb_utils import *
# Load detector
# from detectron2.engine.defaults import DefaultPredictor
# from lib.utils.utils_detectron2 import DefaultPredictor_Lazy
# from detectron2.config import LazyConfig
# import lib
# from lib.utils_det import vitdet_getitem_function, normalize_img_patch
from pathlib import Path
import cv2

# # pip install ultralytics opencv-python torch
# import cv2
# import numpy as np
# from ultralytics import YOLO
# Gender information
gender_info = {
    "P1": 0, "P3": 0, "P4": 0, "P7": 0, "P8": 0, "P10": 0, "P13": 0, "P15": 0,
    "P2": 1, "P5": 1, "P6": 1, "P9": 1, "P11": 1, "P12": 1, "P14": 1, "P16": 1,
    "P17": 1, "P18": 1, "P19": 1, "P20": 1
}

NON_VALID_INDICATOR = [[1, 12, 58], [1, 13, 491], [1, 31, 324], [1, 36, 284], [1, 37, 43], [5, 43, 551], [5, 43, 552], [5, 47, 136], [5, 47, 140]]


def get_all_file_pairs(root_dir, select_p = ["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10", "p11", "p12", "p13", "p14", "p15", "p16", "p17", "p18", "p19", "p20"]):
    all_file_pairs = []
    sub_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d)) and re.match(r'p\d+', d) and d in select_p]

    for sub_dir in sorted(sub_dirs, key=lambda x: int(re.findall(r'\d+', x)[0])):
        sub_dir_num = int(re.findall(r'\d+', sub_dir)[0])

        mat_root_dir = os.path.join(root_dir, sub_dir, 'actions')
        parameters_dir = os.path.join(root_dir, sub_dir, 'parameters')

        print(f"Checking directory {sub_dir}...")

        if not (os.path.exists(mat_root_dir) and os.path.exists(parameters_dir)):
            print(f"Skipping {sub_dir}: missing folders.")
            continue

        for i in range(1, 51):
            json_path = os.path.join(parameters_dir, f'{i}.json')
            if not os.path.exists(json_path):
                continue

            mat_dir = os.path.join(mat_root_dir, str(i), 'mmwave')
            rgb_dir = os.path.join(mat_root_dir, str(i), 'rgb')
            depth_dir = os.path.join(mat_root_dir, str(i), 'depth')

            if not (os.path.exists(mat_dir) and os.path.exists(rgb_dir) and os.path.exists(depth_dir)):
                continue

            mat_files = {extract_frame_index(f): os.path.join(mat_dir, f) for f in os.listdir(mat_dir) if f.endswith('.mat')}
            png_files = {extract_frame_index(f): os.path.join(rgb_dir, f) for f in os.listdir(rgb_dir) if f.endswith('.png')}
            depth_files = {extract_frame_index(f): os.path.join(depth_dir, f) for f in os.listdir(depth_dir) if f.endswith('.png')}
            

            common_frames = sorted(set(mat_files) & set(png_files) & set(depth_files), key=int)
            union_frames = sorted(set(mat_files) | set(png_files) | set(depth_files), key=int)
            
            
            

            for new_frame_idx, frame_key in enumerate(union_frames):
                if frame_key not in common_frames:
                    print(f"Skipping frame {new_frame_idx} in {i} - not present in both mat and png directories.")
                    continue
                indicator = [sub_dir_num, i, new_frame_idx]
                all_file_pairs.append((
                    mat_files[frame_key],
                    png_files[frame_key],
                    depth_files[frame_key],
                    json_path,
                    new_frame_idx,
                    indicator,
                    gender_info[sub_dir.capitalize()],
                ))

    # Sort by numeric indicator
    all_file_pairs.sort(key=lambda x: x[5])  # x[4] is [sub_dir_num, action_id, frame_idx]

    return all_file_pairs



class ToTensor(object):
    def __call__(self, sample):
        def convert_to_tensor(value, key=None, sub_key=None):
            try:
                if value is None:
                    return None
                elif isinstance(value, np.ndarray):
                    return torch.tensor(value).float()
                elif isinstance(value, (list, tuple)):
                    return torch.tensor(value).float()
                elif isinstance(value, (int, float)):
                    return torch.tensor([value]).float()
                return value  # For any non-numeric types like str
            except Exception as e:
                name = f"{key}.{sub_key}" if sub_key else key
                print(f"[convert_to_tensor] ❌ Failed to convert '{name}'")
                print(f"  Type: {type(value)}, Dtype: {getattr(value, 'dtype', 'N/A')}, Value: {value}")
                raise e

        new_sample = {}
        for key, value in sample.items():
            try:
                if value is None:
                    new_sample[key] = None
                    continue

                if key == 'image' or key == 'depth':
                    new_sample[key] = torch.tensor(value).float().permute(2, 0, 1)

                elif key == 'indicator':
                    new_sample[key] = torch.tensor(value).long()

                elif key in ['parameter', 'calibration']:
                    nested_tensor_dict = {}
                    for sub_key, sub_val in value.items():
                        nested_tensor_dict[sub_key] = convert_to_tensor(sub_val, key, sub_key)
                    new_sample[key] = nested_tensor_dict

                else:
                    new_sample[key] = convert_to_tensor(value, key)

            except Exception as e:
                print(f"[ToTensor] ❌ Error processing key: '{key}'")
                raise e

        return new_sample

# Dataset class with lazy loading and LMDB caching
class RF3DPoseDataset(Dataset):
    def __init__(
        self, file_pairs, transform=None, split='train',
        cache_dir='cached_data_test', load_save=True,
        smplx_model_path='models', use_image=False,
        radar_shape=(121, 111, 31), temporal_window = 4, is_demo=False,
        normalize_flag=True, main_modality="radar_points", protocol_id = "p3", split_id = "s2",
    ):
        assert split in ['train', 'val', 'test']
        self.transform = transform
        self.split = split
        self.use_image = use_image
        self.radar_shape = radar_shape
        self.temporal_window = temporal_window
        self.is_demo = is_demo
        self.cache_path = os.path.join(cache_dir, f'rf3dpose_all')
        os.makedirs(self.cache_path, exist_ok=True)
        
        if main_modality == "radar_points":
            self.normalize_flag = normalize_flag
        else:
            self.normalize_flag = False
        


        # Load SMPLX model
        self._load_smplx_model(smplx_model_path)

        # ✅ 只保存路径；按 worker 懒打开
        self.env_paths = {
            'radar_comp': os.path.join(self.cache_path, 'radar_comp.lmdb'),
            'radar_pc':   os.path.join(self.cache_path, 'radar_pc.lmdb'),
            'params':     os.path.join(self.cache_path, 'params.lmdb'),
            'calib':      os.path.join(self.cache_path, 'calib.lmdb'),
            'indicator':  os.path.join(self.cache_path, 'indicator.lmdb'),
        }
        if self.use_image:
            self.env_paths['image'] = os.path.join(self.cache_path, 'image.lmdb')
            self.env_paths['depth'] = os.path.join(self.cache_path, 'depth.lmdb')
            self.env_paths["item_image"] = os.path.join(self.cache_path, 'item_image.lmdb')
            self.env_paths["item_depth"] = os.path.join(self.cache_path, 'item_depth.lmdb')

        self.lmdb_envs = None
                
        # Check if LMDB paths exist and are non-empty
        lmdb_ready = True
        for p in self.env_paths.values():
            if not os.path.exists(p):
                lmdb_ready = False
                break
            # Check if LMDB contains any data (not just exists)
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
            print("error loading the lmdb")
            raise ValueError("LMDB not ready. Please set load_save=False to process and cache the dataset.")
            # self._process_and_cache(file_pairs, load_save)
        
        # After caching, load indicator list
        self._ensure_read_envs()   # ✅ 新增
        with self.lmdb_envs['indicator'].begin() as txn:
            self.indicator_list = []
            cursor = txn.cursor()
            for key, value in cursor:
                indicator = msgpack.unpackb(value, raw=False)
                self.indicator_list.append(indicator)
        if not self.is_demo:
            self.indices_saved_file = os.path.join(self.cache_path, "indeces.pkl.gz")
            if os.path.exists(self.indices_saved_file) == False:
                save_idx_to_file(self.indicator_list,path=self.indices_saved_file)
                split_indicator_lists = load_idx_to_file(path=self.indices_saved_file)
            else: split_indicator_lists = load_idx_to_file(path=self.indices_saved_file)
            
            self.indicator_list = split_indicator_lists[protocol_id][split_id][self.split]
            
            print(f"Loaded {self.split} Dataset with length {len(self.indicator_list)}.")
            test_subs = sorted({s for s, a, f in self.indicator_list})
            test_acts = sorted({a for s, a, f in self.indicator_list})
            print(f"Unique sub in {self.split}:", test_subs)
            print(f"Unique act in {self.split}:", test_acts)
                
        self._remove_nan_entries()

    def _load_smplx_model(self, model_path):
        self.smplx_models = {}
        for gender in ['neutral', 'male', 'female']:
            self.smplx_models[gender] = smplx.create(model_path=model_path, model_type='smplx', gender=gender)
        self.faces = self.smplx_models['neutral'].faces


    def find_temporal_radar_pc_lmdb(self, current_index, T=6):
        self._ensure_read_envs()   # ✅ 新增
        indicator = self.indicator_list[current_index]
        sub_id, action_id, current_frame = indicator
        radar_pc_selected = []
        indicator_selected = []

        last_pc = None
        last_indicator = indicator
        expected_frame = current_frame

        # Scan backwards in indicator_list
        for i in range(current_index, -1, -1):
            s, a, f = self.indicator_list[i]
            if s == sub_id and a == action_id and f == expected_frame:
                with self.lmdb_envs['radar_pc'].begin() as txn_pc:
                    radar_pc_data = txn_pc.get(str(self.indicator_list[i]).encode())
                last_pc = unpack_ndarray(radar_pc_data)
                last_indicator = self.indicator_list[i]

            radar_pc_selected.append(copy.deepcopy(last_pc))
            indicator_selected.append(last_indicator)
            expected_frame -= 1
            if len(radar_pc_selected) == T:
                break
            if s != sub_id or a != action_id or f < current_frame - T + 1:
                break

        # If not enough, repeat last_pc
        while len(radar_pc_selected) < T:
            radar_pc_selected.append(copy.deepcopy(last_pc) if last_pc is not None else None)
            indicator_selected.append(last_indicator)

        return radar_pc_selected[::-1], indicator_selected[::-1]

    def find_temporal_radar_image_lmdb(self, current_index, T=6):
        self._ensure_read_envs()   # ✅ 新增
        indicator = self.indicator_list[current_index]
        sub_id, action_id, current_frame = indicator
        radar_image_selected = []
        indicator_selected = []

        last_image = None
        last_indicator = indicator
        expected_frame = current_frame

        for i in range(current_index, -1, -1):
            s, a, f = self.indicator_list[i]
            if s == sub_id and a == action_id and f == expected_frame:
                with self.lmdb_envs['radar_comp'].begin() as txn_comp:
                    radar_comp_data = txn_comp.get(str(self.indicator_list[i]).encode())
                last_image = self._decompress_radar(unpack_ndarray(radar_comp_data))
                last_indicator = self.indicator_list[i]

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
    
    def pad_radar_pc(self, radar_pc, target_points=1000):
        self._ensure_read_envs()   # ✅ 新增
        P, D = radar_pc.shape
        assert D == 4, "Each point must have 4 values (x, y, z, intensity)"

        padded_pc = np.zeros((target_points, D), dtype=radar_pc.dtype)

        if P >= target_points:
            idx = np.random.choice(P, target_points, replace=False)
            padded_pc = radar_pc[idx]
        else:
            padded_pc[:P] = radar_pc

        return padded_pc

    def _compress_radar(self, radar_image):
        radar_image = np.nan_to_num(radar_image, nan=0.0)  # Replace NaN with 0
        indices = np.array(np.nonzero(radar_image)).T
        values = radar_image[tuple(indices.T)]
        compressed = np.concatenate([indices, values[:, None]], axis=1)
        return compressed

    def _decompress_radar(self, compressed):
        indices = compressed[:, :3].astype(np.int64)
        values = compressed[:, 3]
        radar_image = np.zeros(self.radar_shape, dtype=values.dtype)
        radar_image[indices[:, 0], indices[:, 1], indices[:, 2]] = values
        return radar_image

    def _ensure_read_envs(self):
        cur_pid = os.getpid()
        # 第一次：还没有任何 env
        if getattr(self, "_lmdb_owner_pid", None) != cur_pid or self.lmdb_envs is None:
            # 先把旧的（若有）关掉，防止句柄泄漏
            if getattr(self, "lmdb_envs", None):
                for ev in self.lmdb_envs.values():
                    try:
                        ev.close()
                    except Exception:
                        pass
            self.lmdb_envs = {}
            for name, path in self.env_paths.items():
                subdir_flag = os.path.isdir(path)
                self.lmdb_envs[name] = lmdb.open(
                    path,
                    readonly=True,
                    lock=False,
                    readahead=False,
                    max_readers=4096,
                    subdir=subdir_flag,
                )
            self._lmdb_owner_pid = cur_pid  # 记录归属 PID
    
    
    '''important preprocessing'''
    def _remove_nan_entries(self, rescan = False):
        self._ensure_read_envs()   # ✅ 新增
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

        valid_indicators = []
        nonvalid_indicators = []
        total = len(self.indicator_list)

        if rescan:
            for idx in tqdm(range(total), desc="Filtering NaN/Inf entries"):
                indicator = self.indicator_list[idx]
                reasons = []
                to_remove = False

                # Load from LMDB
                with self.lmdb_envs['params'].begin() as txn_param, \
                    self.lmdb_envs['radar_comp'].begin() as txn_comp, \
                    self.lmdb_envs['radar_pc'].begin() as txn_pc, \
                    self.lmdb_envs['calib'].begin() as txn_calib: #  
                    

                    radar_comp_data = txn_comp.get(str(indicator).encode())
                    radar_pc_data = txn_pc.get(str(indicator).encode())
                    param_data = txn_param.get(str(indicator).encode())
                    calib_data = txn_calib.get(str(indicator).encode())

                radar_comp = unpack_ndarray(radar_comp_data)
                radar_pc = unpack_ndarray(radar_pc_data)
                parameter = unpack_dict(param_data)
                calibration = unpack_dict(calib_data)

                radar_image = self._decompress_radar(radar_comp)
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
                    ('parameter', parameter),
                    ('calibration', calibration),
                    ('indicator', indicator)
                ]
                if self.use_image:
                    with self.lmdb_envs['image'].begin() as txn_img:
                        image_data = txn_img.get(str(indicator).encode())
                    if image_data is not None:
                        image = unpack_ndarray(image_data)
                        fields.append(('image', image))
                        
                    with self.lmdb_envs['depth'].begin() as txn_depth:
                        depth_data = txn_depth.get(str(indicator).encode())
                    if depth_data is not None:
                        depth = unpack_ndarray(depth_data)
                        fields.append(('depth', depth))
                        
                    with self.lmdb_envs['item_image'].begin() as txn_item_img:
                        item_image_data = txn_item_img.get(str(indicator).encode())
                    if item_image_data is not None:
                        item_image = unpack_dict_np(item_image_data)
                        fields.append(('item_image', item_image))
                    
                for name, value in fields:
                    if has_nan_or_inf(value):
                        to_remove = True
                        reasons.append(f"{name} contains NaN or Inf")

                if to_remove:
                    nonvalid_indicators.append(indicator)
                    print(f"Removing sample indicator {indicator} due to: {', '.join(reasons)}")
                    continue

                valid_indicators.append(indicator)
        else:
            nonvalid_indicators = NON_VALID_INDICATOR
        
        
        valid_indicators = [c for c in self.indicator_list if tuple(c) not in {tuple(c1) for c1 in nonvalid_indicators}]
        self.indicator_list = valid_indicators
        print(f"Kept {len(valid_indicators)} / {total} samples after filtering.")


    def __getitem__(self, idx):
            indicator = self.indicator_list[idx]

            # Load param and image if exists from LMDB
            with self.lmdb_envs['params'].begin() as txn_param, \
                 self.lmdb_envs['calib'].begin() as txn_calib:

                parameter = unpack_dict_np(txn_param.get(str(indicator).encode()))
                calibration = unpack_dict_np(txn_calib.get(str(indicator).encode()))

            image = None
            depth_image = None
            item_image = None
            item_depth = None
            if self.use_image:
                with self.lmdb_envs['image'].begin() as txn_img:
                    image_data = txn_img.get(str(indicator).encode())
                    image = unpack_ndarray(image_data)
                with self.lmdb_envs['depth'].begin() as txn_depth:
                    depth_data = txn_depth.get(str(indicator).encode())
                    depth_image = unpack_ndarray(depth_data)
                with self.lmdb_envs['item_depth'].begin() as txn_item_depth:
                    item_depth_data = txn_item_depth.get(str(indicator).encode())
                    item_depth = unpack_dict_np(item_depth_data)
                    item_depth['img'] = item_depth['img'].astype(np.float32)
                with self.lmdb_envs['item_image'].begin() as txn_item_img:
                    item_image_data = txn_item_img.get(str(indicator).encode())
                    item_image = unpack_dict_np(item_image_data)
                    item_image['img'] = normalize_img_patch(item_image['img'].astype(np.float32))
                    
                    
                box_center = item_image["box_center"]
                box_size = item_image["box_size"]
                img_size = item_image["img_size"]

                

            # Temporal radar point cloud and radar image
            radar_PC_selected, indicator_selected = self.find_temporal_radar_pc_lmdb(idx, T=self.temporal_window)
            radar_image_selected, _ = self.find_temporal_radar_image_lmdb(idx, T=self.temporal_window)
            radar_image_selected = np.stack(radar_image_selected, axis=0)  # shape: (T, 121, 111, 31)

            for i in range(len(radar_PC_selected)):
                if self.normalize_flag:
                    radar_PC_selected[i][..., 2] = radar_PC_selected[i][..., 2] - 1.5
                radar_PC_selected[i] = self.pad_radar_pc(radar_PC_selected[i], target_points=1000)
            radar_PC_seq = np.stack(radar_PC_selected, axis=0)  # shape: (T, 1000, self.temporal_window=4)

            parameter["gender"] = gender_info[f"P{indicator[0]}"]
            parameter_radar = calibrate_param_to_radar(parameter, calibration)
            parameter_cam = calibrate_param_to_rgb(parameter, calibration)
            
            if self.normalize_flag:
                parameter_radar["trans"][...,2] = parameter_radar["trans"][...,2] - 1.5
            radar_coords = generate_vertices_in_radar_space(self.smplx_models, parameter_radar)
            joints_root = compute_joints_from_vertices(self.smplx_models, radar_coords, parameter_radar["gender"])
            bbbox = aabb_from_smplx_params(parameter_radar)

            camera_extrinsics = np.eye(4)
            camera_extrinsics[:3, :3] = calibration['radar_to_cam_rotmatrix']
            camera_extrinsics[:3, 3] = calibration['radar_to_cam_tvec']
            
            rgb_coords = generate_vertices_in_radar_space(self.smplx_models, parameter_cam)
            projected_vertices = project_3d_to_2d1(rgb_coords/1000)
            # projected_vertices = project_3d_to_2d(radar_coords / 1000)
            
            sample = {
                'rawImage_XYZ': radar_image_selected,
                'img_vertices': rgb_coords,
                'vertices': radar_coords,
                'bbbox': bbbox,
                'projected_vertices': projected_vertices,
                'parameter': parameter_radar,
                'calibration': calibration,
                'indicator': indicator,
                'radar_points': radar_PC_seq,
                "joints_root": joints_root,
            }

            if self.use_image:
                sample['image'] = image
                sample["depth"] = depth_image
                sample['item_image'] = item_image
                sample["item_depth"] = item_depth


            if self.transform:
                sample = self.transform(sample)

            return sample

    def __len__(self):
        return len(self.indicator_list)



if __name__ == "__main__":
    # root_dir = "/media/jiarui/HDD/Dataset/Edinburgh_mmwave"
    # all_pairs = get_all_file_pairs(root_dir)
    # indicator_target = [2,50, 100]
    # for i in range(len(all_pairs)):
    #     if all_pairs[i][-2][0] == indicator_target[0] and all_pairs[i][-2][1] == indicator_target[1] and all_pairs[i][-2][2] == indicator_target[2]:
    #         s = i
    #         print(f"Found target indicator at index {s}: {all_pairs[i][-2]}")
    #         break
    
    # file_pairs = all_pairs#[s:s+100]
    # print(f"Total frames: {len(file_pairs)}")
    dataset = RF3DPoseDataset([], transform=ToTensor(), load_save=True, use_image=True, is_demo=True, cache_dir="../cached_data_test_vis/")
    print("Dataset loaded.")
    for i in range(50):
        plot_frames_for_gif_depth(dataset, combined_gif_path=f'vis_depth/combined_output_depth_{i+1}.gif', act_id=i+1)
