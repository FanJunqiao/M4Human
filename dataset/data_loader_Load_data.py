import logging
import scipy.io as sio
import numpy as np
import os
import json


# gender information from P1-p20
gender_info = {
    "P1": 0, "P3": 0, "P4": 0, "P7": 0, "P8": 0, "P10": 0, "P13": 0, "P15": 0,
    "P2": 1, "P5": 1, "P6": 1, "P9": 1, "P11": 1, "P12": 1, "P14": 1, "P16": 1,
    "P17": 1, "P18": 1, "P19": 1, "P20": 1
}

# Loading mat files (input data)
def load_mat_file(mat_path):
    try:
        radar_data = {}
        mat_data = sio.loadmat(mat_path)
        rawImage_XYZ = mat_data.get('rawImage_XYZ')
        if rawImage_XYZ is None:
            raise ValueError(f"'rawImage_XYZ' not found in {mat_path}")
        rawImage_XYZ = rawImage_XYZ.astype(np.float32)
        radar_data["rawImage_XYZ"] = rawImage_XYZ

        RawPoints = mat_data.get('RawPoints')
        if RawPoints is None:
            raise ValueError(f"'RawPoints' not found in {mat_path}")
        RawPoints = RawPoints.astype(np.float32)
        radar_data["RawPoints"] = RawPoints
        return radar_data
    except Exception as e:
        print(f"Error loading MAT file {mat_path}: {e}")
        return None

# Loading obj files (Ground truth)
def load_obj_file(obj_path):
    try:
        vertices = []
        faces = []
        with open(obj_path, 'r') as obj_file:
            for line in obj_file:
                if line.startswith('v '):
                    vertices.append(list(map(float, line.strip().split()[1:])))
                elif line.startswith('f '):
                    faces.append([int(i.split('/')[0]) - 1 for i in line.strip().split()[1:]])
        vertices = np.array(vertices, dtype=np.float32)
        faces = np.array(faces, dtype=np.int32).reshape(-1, 3)
        return vertices, faces
    except Exception as e:
        print(f"Error loading OBJ file {obj_path}: {e}")
        return None, None
    
# Loading Json files (SMPL-X parameters data)
def load_json_data(json_path):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading JSON file {json_path}: {e}")
        return None

# Loading camera calibration data
def load_transformation_matrices(ins_ext_dir, indicator):
    
    try:
        if indicator[0] in [14,19]:
            act = indicator[1]
            if indicator[0] == 14:
                if act <= 15:
                    sub_dir = "action1-15"
                elif act <= 25:
                    sub_dir = "action16-25"
                elif act <= 29:
                    sub_dir = "action26-29"
                elif act <= 37:
                    sub_dir = "action30-37"
                elif act <= 42:
                    sub_dir = "action38-42"
                else:
                    sub_dir = "action43-50"
            else:  # indicator[0] == 19
                if act <= 14:
                    sub_dir = "action1-14"
                elif act <= 20:
                    sub_dir = "action15-20"
                elif act <= 24:
                    sub_dir = "action21-24"
                elif act <= 50:
                    sub_dir = "action25-50"
                        
            cam_ins = np.loadtxt(os.path.join(ins_ext_dir, sub_dir, 'cam_ins.txt'))
            vicon_to_cam = np.loadtxt(os.path.join(ins_ext_dir, sub_dir, 'vicon_to_cam.txt'))
            
            radar_to_cam_rotmatrix = np.loadtxt(os.path.join(ins_ext_dir, sub_dir, 'radar2rgb_rotmatrix.txt'))
            radar_to_cam_tvec = np.loadtxt(os.path.join(ins_ext_dir, sub_dir, 'radar2rgb_tvec.txt'))
            
            vicon_to_cam_rotmatrix = vicon_to_cam[:3, :3]
            vicon_to_cam_tvec = vicon_to_cam[:3, 3]
            
            return cam_ins, vicon_to_cam_rotmatrix, vicon_to_cam_tvec, radar_to_cam_rotmatrix, radar_to_cam_tvec
        else:
            cam_ins = np.loadtxt(os.path.join(ins_ext_dir, 'cam_ins.txt'))
            vicon_to_cam = np.loadtxt(os.path.join(ins_ext_dir, 'vicon_to_cam.txt'))
            
            radar_to_cam_rotmatrix = np.loadtxt(os.path.join(ins_ext_dir, 'radar2rgb_rotmatrix.txt'))
            radar_to_cam_tvec = np.loadtxt(os.path.join(ins_ext_dir, 'radar2rgb_tvec.txt'))
            
            vicon_to_cam_rotmatrix = vicon_to_cam[:3, :3]
            vicon_to_cam_tvec = vicon_to_cam[:3, 3]
            
            return cam_ins, vicon_to_cam_rotmatrix, vicon_to_cam_tvec, radar_to_cam_rotmatrix, radar_to_cam_tvec
    except Exception as e:
        print(f"Error loading transformation matrices from {ins_ext_dir}: {e}")
        return None, None, None, None, None
