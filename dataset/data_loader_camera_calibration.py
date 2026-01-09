import logging
import os
import numpy as np
# set log information path
import torch
import copy
log_file_path = 'data11_log_cali.txt'

if log_file_path:
    directory = os.path.dirname(log_file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(message)s',
                    handlers=[logging.FileHandler(log_file_path)])
logger = logging.getLogger('data_loader')
logger.setLevel(logging.INFO)


def normalize_axis_angle_flat(vec):
    """
    Normalize a flattened axis-angle vector or a single one.
    
    Args:
        vec (np.ndarray): shape (3,) or (N*3,), flattened axis-angle(s)
        
    Returns:
        np.ndarray: same shape as input, with each axis-angle normalized
    """
    vec = np.asarray(vec)
    assert vec.ndim == 1, "Input must be 1D array of shape (3,) or (N*3,)"
    assert vec.shape[0] % 3 == 0, "Length must be multiple of 3"

    N = vec.shape[0] // 3
    vec_reshaped = vec.reshape(N, 3)

    angles = np.linalg.norm(vec_reshaped, axis=1, keepdims=True)
    small_angle_mask = angles < 1e-8

    axis = np.zeros_like(vec_reshaped)
    axis[~small_angle_mask[:, 0]] = vec_reshaped[~small_angle_mask[:, 0]] / angles[~small_angle_mask[:, 0]]

    # Wrap angle to [0, 2π]
    angles = np.mod(angles, 2 * np.pi)

    # If angle > π, flip direction and reduce angle
    flip_mask = angles[:, 0] > np.pi
    angles[flip_mask] = 2 * np.pi - angles[flip_mask]
    axis[flip_mask] = -axis[flip_mask]

    normalized = axis * angles
    return normalized.reshape(-1)  # flatten back to (N*3,)


def rodrigues(r):
    """Convert axis-angle to rotation matrix."""
    theta = np.linalg.norm(r)
    if theta < 1e-8:
        return np.eye(3)
    k = r / theta
    K = np.array([
        [0, -k[2], k[1]],
        [k[2], 0, -k[0]],
        [-k[1], k[0], 0]
    ])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)

def inv_rodrigues(R):
    """Convert rotation matrix back to axis-angle."""
    theta = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    if theta < 1e-8:
        return np.zeros(3)
    r = np.array([
        R[2,1] - R[1,2],
        R[0,2] - R[2,0],
        R[1,0] - R[0,1]
    ]) / (2 * np.sin(theta))
    return r * theta

def calibrate_param_to_radar(param, calib):
    R_root = rodrigues(param['root_orient'])
    R_cam = calib['vicon_to_cam_rotmatrix'] @ R_root
    R_radar = np.linalg.inv(calib['radar_to_cam_rotmatrix']) @ R_cam

    T_cam = calib['vicon_to_cam_rotmatrix'] @ (param['joints'][0])  + calib['vicon_to_cam_tvec'] / 1000 # param['trans']
    T_radar = np.linalg.inv(calib['radar_to_cam_rotmatrix']) @ (T_cam - calib['radar_to_cam_tvec']) 
    
    # print("calib", calib['vicon_to_cam_rotmatrix'], calib['vicon_to_cam_tvec'], calib['radar_to_cam_rotmatrix'], calib['radar_to_cam_tvec'], param["trans"], param['joints'][0])


    # some = calib['vicon_to_cam_rotmatrix'] @ (-param['trans']+param['joints'][0])  + calib['vicon_to_cam_tvec'] / 1000 # 
    # some = np.linalg.inv(calib['radar_to_cam_rotmatrix']) @ some - calib['radar_to_cam_tvec']/ 1000  #
    
    param_radar = copy.deepcopy(param)
    param_radar['root_orient'] = inv_rodrigues(R_radar)
    param_radar['trans'] = T_radar + (param['trans']-param['joints'][0])

    return param_radar

def calibrate_param_to_rgb(param, calib):
    R_root = rodrigues(param['root_orient'])
    R_cam = calib['vicon_to_cam_rotmatrix'] @ R_root

    T_cam = calib['vicon_to_cam_rotmatrix'] @ (param['joints'][0])  + calib['vicon_to_cam_tvec'] / 1000 # param['trans']

    
    param_rgb = copy.deepcopy(param)
    param_rgb['root_orient'] = inv_rodrigues(R_cam)
    param_rgb['trans'] = T_cam + (param['trans']-param['joints'][0])

    return param_rgb


def pc3d_to_tensor_idx(points_xyz: np.ndarray, clip: bool = True,
                       bounds = [[-3.0, 3.0], [0.5, 6.0], [0.0, 3.0]],
                       tensor_shape = [121, 111, 31]) -> np.ndarray:
    """
    Map (N,3) xyz in meters -> (N,3) integer tensor indices [ix,iy,iz],
    using default bounds & grid_shape.
    """
    pts = np.asarray(points_xyz, dtype=np.float64)
    Nx, Ny, Nz = tensor_shape
    (xmin,xmax), (ymin,ymax), (zmin,zmax) = bounds

    dx = (xmax - xmin) or 1e-9
    dy = (ymax - ymin) or 1e-9
    dz = (zmax - zmin) or 1e-9

    ixf = (pts[:, 0] - xmin) / dx * (Nx - 1)
    iyf = (pts[:, 1] - ymin) / dy * (Ny - 1)
    izf = (pts[:, 2] - zmin) / dz * (Nz - 1)

    idx = np.floor(np.stack([ixf, iyf, izf], axis=1)).astype(np.int64)
    if clip:
        idx[:, 0] = np.clip(idx[:, 0], 0, Nx - 1)
        idx[:, 1] = np.clip(idx[:, 1], 0, Ny - 1)
        idx[:, 2] = np.clip(idx[:, 2], 0, Nz - 1)
    return idx

def aabb_from_vertices_asym_with_tensor(
    verts: np.ndarray,
    pad_min=(0.0, 0.0, 0.0),
    pad_max=(0.0, 0.0, 0.0),
):
    """
    Compute AABB for verts with asymmetric padding, return results in
    both world (meters) and tensor (indices).
    """
    v = np.asarray(verts, dtype=np.float32)
    pad_min = np.asarray(pad_min, dtype=np.float32)
    pad_max = np.asarray(pad_max, dtype=np.float32)

    # ---- pc coords (meters) ----
    vmin = v.min(axis=0) - pad_min
    vmax = v.max(axis=0) + pad_max
    center = (vmin + vmax) / 2.0
    size   = (vmax - vmin)

    x0, y0, z0 = vmin
    x1, y1, z1 = vmax
    corners_metric = np.array([
        [x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0],
        [x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1]
    ], dtype=np.float32)

    # ---- Tensor coords (indices) ----
    key_pts   = np.stack([vmin, vmax, center], axis=0)
    key_idx   = pc3d_to_tensor_idx(key_pts, clip=True)
    min_idx, max_idx, center_idx = key_idx[0], key_idx[1], key_idx[2]
    size_idx = (max_idx - min_idx).astype(np.int64)
    corners_idx = pc3d_to_tensor_idx(corners_metric, clip=True)

    return {
        "pc": {
            "min": vmin, "max": vmax,
            "center": center, "size": size,
            "corners": corners_metric,
        },
        "tensor": {
            "min": min_idx, "max": max_idx,
            "center": center_idx, "size": size_idx,
            "corners": corners_idx,
        }
    }


def aabb_from_smplx_params(
    smplx_params: dict,
    beta_min=(0.6, 0.9, 1.2),
    beta_max = (0.6, 0.3, 1.2)
):
    """
    Build an axis-aligned bounding box (AABB) from SMPL-X params.

    smplx_params: dict containing at least 'trans' (3,)
    beta_size: scalar or (3,) array, bbox size in meters along x,y,z

    Returns bbox in both world (meters) and tensor (indices).
    """
    # Center from SMPL-X translation
    root_trans = np.asarray(smplx_params["trans"], dtype=np.float32).reshape(3)

    # Fixed size from beta
    beta_min = np.asarray(beta_min, dtype=np.float32)
    beta_max = np.asarray(beta_max, dtype=np.float32)

    # ---- World bbox ----
    vmin = root_trans -  beta_min
    vmax = root_trans + beta_max
    center = root_trans
    size   = beta_min + beta_max

    x0, y0, z0 = vmin
    x1, y1, z1 = vmax
    corners_metric = np.array([
        [x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0],
        [x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1]
    ], dtype=np.float32)

    # ---- Tensor bbox ----
    key_pts   = np.stack([vmin, vmax, center], axis=0)
    key_idx   = pc3d_to_tensor_idx(key_pts, clip=True)
    min_idx, max_idx, center_idx = key_idx[0], key_idx[1], key_idx[2]

    # Enforce non-degenerate tensor bbox
    for d in range(3):
        if max_idx[d] <= min_idx[d]:
            max_idx[d] = min_idx[d] + 1

    size_idx = (max_idx - min_idx).astype(np.int64)
    corners_idx = pc3d_to_tensor_idx(corners_metric, clip=True)

    return {
        "pc": {
            "min": vmin, "max": vmax,
            "center": center, "size": size,
            "corners": corners_metric,
        },
        "tensor": {
            "min": min_idx, "max": max_idx,
            "center": center_idx, "size": size_idx,
            "corners": corners_idx,
        }
    }



def compute_joints_from_vertices(smplx_models, vertices, gender):
    vertices = torch.tensor(vertices)
    # unwrap to access layers under DDP/DataParallel
    if gender > 0.5:
        gender = 'male'
    else:
       gender = 'female'
    smplx_layer = smplx_models[gender]
    J_regressors = smplx_layer.J_regressor[:22]
    joints = torch.einsum('ij,jc->ic', J_regressors, vertices)/1000
    return joints.numpy()

def generate_vertices_in_radar_space(smplx_models, param, calibration=None):
    """
    Generate mesh vertices and transform them into radar coordinate space.

    Args:
        smplx_models (dict): Dict of gender to SMPLX models, e.g., {'male': ..., 'female': ..., 'neutral': ...}
        param (dict): SMPLX parameters with keys: 'betas', 'pose_body', 'root_orient', 'trans', 'gender'
        calibration (dict): Contains vicon-to-cam and cam-to-radar extrinsics

    Returns:
        radar_coords (np.ndarray): Vertices in radar coordinate system, shape (N, 3)
    """
    gender_value = param.get('gender', 'neutral')
    if gender_value > 0.5:
        gender = 'male'
    else:
       gender = 'female'
    smplx_model = smplx_models[gender]

    output = smplx_model(
        betas=torch.tensor(param['betas'][None], dtype=torch.float32),
        body_pose=torch.tensor(param['pose_body'][None], dtype=torch.float32),
        global_orient=torch.tensor(param['root_orient'][None], dtype=torch.float32),
        transl=torch.tensor(param['trans'][None], dtype=torch.float32)
    )
    vertices = output.vertices.detach().cpu().numpy().squeeze() * 1000  # mm
    if calibration is not None:

        vertices = vicon_to_cam_transform(
            vertices,
            calibration['vicon_to_cam_rotmatrix'],
            calibration['vicon_to_cam_tvec']
        )
        vertices = cam_to_radar_transform(
            vertices,
            calibration['radar_to_cam_rotmatrix'],
            calibration['radar_to_cam_tvec']
        )

    return vertices

# comvert vicon frame to camera frame
def vicon_to_cam_transform(vicon_coords, vicon_to_cam_rotmatrix, vicon_to_cam_tvec):
    try:
        cam_coords = np.dot(vicon_to_cam_rotmatrix, vicon_coords.T).T + vicon_to_cam_tvec
        return cam_coords
    except Exception as e:
        logger.error(f"Error during Vicon to Camera transformation: {e}")
        return None

#convert camera frame to radar frame
def cam_to_radar_transform(cam_coords, radar_to_cam_rotmatrix, radar_to_cam_tvec):
    try:
        # 逆变换
        radar_coords = np.dot(np.linalg.inv(radar_to_cam_rotmatrix), (cam_coords - radar_to_cam_tvec).T).T
        return radar_coords
    except Exception as e:
        logger.error(f"Error during Camera to Radar transformation: {e}")
        return None
    
x_base = 3.236
def radar_grid_to_pointcloud(rawImage_XYZ, xlim=(-x_base, x_base), ylim=(0., x_base*2/60*55), zlim=(-x_base/60*15, x_base/60*15), threshold_k=2.0):
    """
    Convert radar volume into point cloud using CFAR-like thresholding.
    Args:
        rawImage_XYZ: 3D numpy array of radar power values (shape: [X, Y, Z])
        xlim, ylim, zlim: real-world limits of the grid axes
        threshold_k: scale factor for thresholding (CFAR approximation)
    Returns:
        points: (N, 3) array of 3D coordinates
    """
    X, Y, Z = rawImage_XYZ.shape
    x = np.linspace(xlim[0], xlim[1], X)
    y = np.linspace(ylim[0], ylim[1], Y)
    z = np.linspace(zlim[0], zlim[1], Z)
    
    threshold = rawImage_XYZ.mean() + threshold_k * rawImage_XYZ.std()
    mask = rawImage_XYZ > threshold
    
    indices = np.argwhere(mask)
    if len(indices) == 0:
        return np.zeros((0, 3))  # no valid points

    points = np.array([
        [x[i], y[j], z[k]]
        for i, j, k in indices
    ])
    return points


def radar_pc_to_image(points, radar_to_cam_rotmatrix, radar_to_cam_tvec, cam_intrinsics):
    """
    Project radar point cloud to image coordinates.
    Args:
        points: (N, 3) radar-space 3D points
        radar_to_cam_rotmatrix: (3, 3)
        radar_to_cam_tvec: (3,)
        cam_intrinsics: (3, 3)
    Returns:
        projected_2d: (N, 2) image coordinates
    """
    cam_coords = np.dot(radar_to_cam_rotmatrix, points.T).T + radar_to_cam_tvec
    in_front_mask = cam_coords[:, 2] > 0
    cam_coords = cam_coords[in_front_mask]

    projected = np.dot(cam_intrinsics, cam_coords.T).T
    projected_2d = projected[:, :2] / projected[:, 2:3]
    return projected_2d