import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import sys 
import os
import time
import copy
from smplx import SMPLXLayer




# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# ROOT_DIR = os.path.dirname(BASE_DIR)
# sys.path.append(ROOT_DIR)
# sys.path.append(os.path.join(ROOT_DIR, 'modules'))

from .p4trans.point_4d_convolution import *
from .p4trans.transformer import * # alias for models' transformer


def batch_rodrigues(theta):
    """
    Convert axis-angle vectors to rotation matrices using Rodrigues' formula.
    
    Args:
        theta: [B, 3] axis-angle vectors (rotation vector)
    Returns:
        R: [B, 3, 3] rotation matrices
    """
    angle = torch.norm(theta, dim=1, keepdim=True)  # [B, 1]
    axis = theta / (angle + 1e-8)                   # [B, 3], unit rotation axis

    # Create skew-symmetric cross-product matrix K for each axis
    x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]
    zeros = torch.zeros_like(x)
    K = torch.stack([
        zeros, -z, y,
        z, zeros, -x,
        -y, x, zeros
    ], dim=1).reshape(-1, 3, 3)  # [B, 3, 3]

    I = torch.eye(3, device=theta.device).unsqueeze(0)  # [1, 3, 3]
    angle = angle.unsqueeze(-1)  # [B, 1, 1]
    sin = torch.sin(angle)
    cos = torch.cos(angle)

    R = I + sin * K + (1 - cos) * torch.bmm(K, K)  # [B, 3, 3]
    return R

def rotation_6d_to_matrix(x):
    x = x.view(-1, 3, 2)
    b1 = torch.nn.functional.normalize(x[:, :, 0], dim=1)
    dot = torch.sum(b1 * x[:, :, 1], dim=1, keepdim=True)
    b2 = torch.nn.functional.normalize(x[:, :, 1] - dot * b1, dim=1)
    b3 = torch.cross(b1, b2, dim=1)
    return torch.stack((b1, b2, b3), dim=2)  # [B, 3, 3]

def inv_rodrigues_batch(R):
    """
    R: [B, 3, 3] rotation matrix
    return: [B, 3] axis-angle vector (rotation vector)
    """
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    cos_theta = (trace - 1) / 2
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    theta = torch.acos(cos_theta)  # [B]
    sin_theta = torch.sqrt(1 - cos_theta**2 + 1e-6)

    r = torch.stack([
        R[:, 2, 1] - R[:, 1, 2],
        R[:, 0, 2] - R[:, 2, 0],
        R[:, 1, 0] - R[:, 0, 1]
    ], dim=1) / (2 * sin_theta.unsqueeze(1))  # [B, 3]

    axis_angle = r * theta.unsqueeze(1)  # [B, 3]
    axis_angle[theta < 1e-5] = 0.0
    return axis_angle

def quat_to_rotmat(quat):
    norm_quat = quat
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]

    B = quat.size(0)

    rotmat = torch.stack(
        [1 - 2 * (y ** 2 + z ** 2), 2 * (x * y - z * w), 2 * (x * z + y * w),
         2 * (x * y + z * w), 1 - 2 * (x ** 2 + z ** 2), 2 * (y * z - x * w),
         2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x ** 2 + y ** 2)],
        dim=1).view(B, 3, 3)
    return rotmat



class P4Transformer(nn.Module):
    def __init__(self, smplx_model_paths,
                 radius, nsamples, spatial_stride,                                # P4DConv: spatial
                 temporal_kernel_size, temporal_stride,                                 # P4DConv: temporal
                 emb_relu,                                                              # embedding: relu
                 dim, depth, heads, dim_head,                                           # transformer
                 mlp_dim, num_classes,                                                  # output
                 dropout1=0.0, dropout2=0.0,                                            # dropout
                ):                                           
        super().__init__()




        self.tube_embedding = P4DConv(in_planes=1, mlp_planes=[dim], mlp_batch_norm=[False], mlp_activation=[False],
                                  spatial_kernel_size=[radius, nsamples], spatial_stride=spatial_stride,
                                  temporal_kernel_size=temporal_kernel_size, temporal_stride=temporal_stride, temporal_padding=[1, 0],
                                  operator='+', spatial_pooling='max', temporal_pooling='max')

        self.pos_embedding = nn.Conv1d(in_channels=4, out_channels=dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.emb_relu = nn.ReLU() if emb_relu else False

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout=dropout1)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.ReLU(),
            # nn.Dropout(dropout2),
            nn.Linear(mlp_dim, num_classes),
        )

        # point Prediction Head
        self.fc2_betas = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim//4),
            nn.ReLU(),
            nn.Dropout(dropout2),
            nn.Linear(dim//4, 10)
        )
        self.fc2_pose_body = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim//4),
            nn.ReLU(),
            nn.Dropout(dropout2),
            nn.Linear(dim//4, 63)
        )
        self.fc2_root_orient = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout2),
            nn.Linear(128, 3)
        )
        self.fc2_trans = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout2),
            nn.Linear(128, 3)
        )
        self.fc2_gender = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout2),
            nn.Linear(64, 1)
        )
        self.smplx_layer_male = SMPLXLayer(model_path=smplx_model_paths['male'], gender='male', use_pca=False)
        self.smplx_layer_female = SMPLXLayer(model_path=smplx_model_paths['female'], gender='female', use_pca=False)
        self.smplx_layer_neutral = SMPLXLayer(model_path=smplx_model_paths['neutral'], gender='neutral', use_pca=False)
        
        self.faces_male = self.smplx_layer_male.faces
        self.faces_female = self.smplx_layer_female.faces
        self.faces_neutral = self.smplx_layer_neutral.faces
        
        self.loss_weights = {
            'betas': 2.0,
            'pose_body': 5.0,
            'root_orient': 1.0,
            'trans': 1.0,
            'vertices': 0.001,
            'gender': 1.0
        }



    def forward(self, radar):    #, print_depth=None, print_radar=None
        temp_start_time = time.time()
        point_cloud = radar[:, :, :, :3]
        point_fea = radar[:, :, :, 3:].permute(0, 1, 3, 2)                                                                                                           # [B, L, N, 3]
        device = radar.get_device()
        
        xyzs, features = self.tube_embedding(point_cloud, point_fea)                                                                                         # [B, L, n, 3], [B, L, C, n] 

        xyzts = []
        xyzs = torch.split(tensor=xyzs, split_size_or_sections=1, dim=1)
        xyzs = [torch.squeeze(input=xyz, dim=1).contiguous() for xyz in xyzs]
        for t, xyz in enumerate(xyzs):
            t = torch.ones((xyz.size()[0], xyz.size()[1], 1), dtype=torch.float32, device=device) * (t+1)
            xyzt = torch.cat(tensors=(xyz, t), dim=2)
            xyzts.append(xyzt)
        xyzts = torch.stack(tensors=xyzts, dim=1)

        xyzts = torch.reshape(input=xyzts, shape=(xyzts.shape[0], xyzts.shape[1]*xyzts.shape[2], xyzts.shape[3]))                           # [B, L*n, 4]

        features = features.permute(0, 1, 3, 2)                                                                                             # [B, L,   n, C]
        features = torch.reshape(input=features, shape=(features.shape[0], features.shape[1]*features.shape[2], features.shape[3]))         # [B, L*n, C]
        
        xyzts_embd = self.pos_embedding(xyzts.permute(0, 2, 1)).permute(0, 2, 1)

        embedding = xyzts_embd + features

        if self.emb_relu:
            embedding = self.emb_relu(embedding)


        # open for template embedding
        output = self.transformer(embedding)
        output = torch.max(input=output, dim=1, keepdim=False, out=None)[0]
        
        x = output
        pred_betas = self.fc2_betas(x)
        pred_pose_body = self.fc2_pose_body(x)
        pred_root_orient = self.fc2_root_orient(x)

        pred_trans = self.fc2_trans(x)
        gender_pred = torch.sigmoid(self.fc2_gender(x))

        return pred_betas, pred_pose_body, pred_root_orient, pred_trans, gender_pred, None
    
    def get_positional_embeddings1(self, sequence_length, d):
        result = np.ones([1, sequence_length, d])
        for i in range(sequence_length):
            for j in range(d):
                result[0][i][j] = np.sin(i / (10000 ** (j / d))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / d)))
        return result
    
    def get_smplx_output(self, pred_betas, pred_pose_body, pred_root_orient, pred_trans, gender):
        vertices_list = []
        for i in range(gender.shape[0]):
            gender_value = gender[i].item()
            if gender_value > 0.5:
                smplx_layer = self.smplx_layer_male
            else:
                smplx_layer = self.smplx_layer_female
            
            pred_root_orient_matrix = batch_rodrigues(pred_root_orient[i].unsqueeze(0)).unsqueeze(0)
            pred_pose_body_matrix = batch_rodrigues(pred_pose_body[i].view(-1, 3)).view(-1, 21, 3, 3)
            
            output = smplx_layer(
                betas=pred_betas[i].unsqueeze(0), 
                body_pose=pred_pose_body_matrix, 
                global_orient=pred_root_orient_matrix, 
                transl=pred_trans[i].unsqueeze(0)
            )
            vertices_list.append(output.vertices)
        vertices = torch.cat(vertices_list, dim=0)
        return vertices*1000
    
def print_anchors(anchor, points):
    from matplotlib import pyplot as plt
    # anchor (16, 312, 4)
    # points (16, 4, 5000, 6)
    
    idx = 0
    b, t, N, _ = points.shape
    anchor_select = anchor[idx].cpu().numpy()
    points_select = points[idx].reshape(t*N, 6).cpu().numpy()

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.scatter(points_select[:,0]+0.0, points_select[:,2]-0.0, marker="o", s=1) # -0.22
    ax.scatter(anchor_select[:,0]+0.0, anchor_select[:,2]-0.0, marker="x", s=1.5, c="r") # -0.22
    ax.set_xlim([-2,2])
    ax.set_ylim([-1,1])
    plt.axis('off')
    plt.savefig(os.path.join("dataloader_vis", f"attn.png"), format="png", bbox_inches='tight', pad_inches=0)


    return
