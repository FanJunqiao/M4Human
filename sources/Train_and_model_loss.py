import torch
import numpy as np
import torch.nn as nn  # minimal add
from torch.nn.parallel import DistributedDataParallel as DDP  # [DDP-safe]

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



def rodrigues_2_rot_mat(rvecs):
    batch_size = rvecs.shape[0]
    r_vecs = rvecs.reshape(-1, 3)
    total_size = r_vecs.shape[0]
    thetas = torch.norm(r_vecs, dim=1, keepdim=True)
    is_zero = torch.eq(torch.squeeze(thetas), torch.tensor(0.0))
    u = r_vecs / thetas

    # Each K is the cross product matrix of unit axis vectors
    # pyformat: disable
    zero = torch.autograd.Variable(torch.zeros([total_size], device="cuda"))  # for broadcasting
    Ks_1 = torch.stack([  zero   , -u[:, 2],  u[:, 1] ], axis=1)  # row 1
    Ks_2 = torch.stack([  u[:, 2],  zero   , -u[:, 0] ], axis=1)  # row 2
    Ks_3 = torch.stack([ -u[:, 1],  u[:, 0],  zero    ], axis=1)  # row 3
    # pyformat: enable
    Ks = torch.stack([Ks_1, Ks_2, Ks_3], axis=1)                  # stack rows

    identity_mat = torch.autograd.Variable(torch.eye(3, device="cuda").repeat(total_size,1,1))
    Rs = identity_mat + torch.sin(thetas).unsqueeze(-1) * Ks + \
         (1 - torch.cos(thetas).unsqueeze(-1)) * torch.matmul(Ks, Ks)
    # Avoid returning NaNs where division by zero happened
    R = torch.where(is_zero[:,None,None], identity_mat, Rs)

    return R.reshape(batch_size, -1)

def unwrap_model(m):
    return m.module if isinstance(m, (nn.DataParallel, DDP)) else m

def compute_joints_from_vertices(model, vertices, gender):
    # unwrap to access layers under DDP/DataParallel
    core = unwrap_model(model)
    J_regressors = []
    for g in gender:
        if g.item() > 0.5:
            J_regressors.append(core.smplx_layer_male.J_regressor[:22].to(vertices.device))
        else:
            J_regressors.append(core.smplx_layer_female.J_regressor[:22].to(vertices.device))
    J_regressors = torch.stack(J_regressors, dim=0)  # (B, 22, 10475)
    joints = torch.einsum('bij,bjc->bic', J_regressors, vertices)  # (B, 22, 3)
    return joints



def pos_loss(output, target):
   """ LOSS FUNCTION FOR POSITION ANGLE BASED ON MINIMUM ARC SEPARATION """
   loss = torch.mean(torch.stack([ (output-target)**2,
                                  (1-torch.abs(output-target))**2] ).min(dim=0)[0])
   return loss

def geodesic_loss_from_axis_angle(pred, target):
    # pred, target: [B, N*3]
    B, N3 = pred.shape
    pred_rotmat = batch_rodrigues(pred.view(B,-1, 3).view(-1, 3))         # [B*N, 3, 3]
    target_rotmat = batch_rodrigues(target.view(B,-1, 3).view(-1, 3))     # [B*N, 3, 3]
    
    R_diff = torch.matmul(pred_rotmat, target_rotmat.transpose(1, 2))  # [B*N, 3, 3]
    trace = R_diff.diagonal(offset=0, dim1=-2, dim2=-1).sum(-1)        # [B*N]
    
    theta = torch.acos(torch.clamp((trace - 1) / 2, -1.0+1e-6, 1.0-1e-6))         # [B*N]
    return theta.view(B, -1).mean()  # Mean over joints and batch

def rot_loss(pred, target):
    B, N3 = pred.shape
    rot_pred = batch_rodrigues(pred.view(-1, 3))   # [B*N, 3, 3]
    rot_target = batch_rodrigues(target.view(-1, 3))  # [B*N, 3, 3]
    return ((rot_pred - rot_target)**2).mean()


def _unwrap_model(m):
    if isinstance(m, (nn.DataParallel, DDP)):
        return m.module
    return m

# loss function
def combined_loss(pred_betas, pred_pose_body, pred_root_orient, pred_trans, pred_vertices, 
                  gt_betas, gt_pose_body, gt_root_orient, gt_trans, gt_vertices, 
                  pred_genders, gt_genders, gt_center, criterion1, criterion2, gender_criterion, model, pred_center=None):
    
    core = _unwrap_model(model)  # <-- DP/DDP-safe

    betas_loss = criterion1(pred_betas, gt_betas)
    pose_body_loss = criterion2(pred_pose_body, gt_pose_body)
    root_orient_loss = rot_loss(pred_root_orient, gt_root_orient)
    trans_loss = criterion2(pred_trans, gt_trans)
    # vertices_loss = criterion2(pred_vertices / 1000, gt_vertices / 1000)
    vertices_loss = torch.zeros_like(trans_loss)
    gender_loss = gender_criterion(pred_genders, gt_genders)
    
    if (pred_center is not None):
        center_loss = criterion2(pred_center, gt_center)
        
    losses = {
        'betas': betas_loss,
        'pose_body': pose_body_loss,
        'root_orient': root_orient_loss,
        'trans': trans_loss,
        'vertices': vertices_loss,
        'gender': gender_loss
    }

    # use weights from the unwrapped core
    weights = getattr(core, 'loss_weights', {})

    # total weighted loss
    total_loss = sum(weights.get(k, 1.0) * v for k, v in losses.items())
    if (pred_center is not None):
        total_loss = total_loss + weights.get('trans', 1.0) * center_loss
        losses['center'] = center_loss

    # also scale each individual loss by its weight
    for k in losses:
        losses[k] = losses[k] * weights.get(k, 1.0)

    return total_loss, losses