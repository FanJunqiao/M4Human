import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from smplx import SMPLXLayer
import numpy as np


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




# set-up a multihead attention model
class MultiHeadAttentionModule(nn.Module):
    def __init__(self, in_channels, num_heads=8):
        super(MultiHeadAttentionModule, self).__init__()
        assert in_channels % num_heads == 0, "in_channels should be divisible by num_heads"
        
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        
        self.query_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.key_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.value_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, depth, height, width = x.size()

        # Apply the convolutional layers and split into heads
        query = self.query_conv(x).view(batch_size, self.num_heads, self.head_dim, -1).permute(0, 1, 3, 2)  # [batch_size, num_heads, depth*height*width, head_dim]
        key = self.key_conv(x).view(batch_size, self.num_heads, self.head_dim, -1)  # [batch_size, num_heads, head_dim, depth*height*width]
        value = self.value_conv(x).view(batch_size, self.num_heads, self.head_dim, -1).permute(0, 1, 3, 2)  # [batch_size, num_heads, depth*height*width, head_dim]

        # Attention score calculation
        energy = torch.matmul(query, key) / (self.head_dim ** 0.5)  # [batch_size, num_heads, depth*height*width, depth*height*width]
        attention = F.softmax(energy, dim=-1)  # [batch_size, num_heads, depth*height*width, depth*height*width]

        # Attention application
        out = torch.matmul(attention, value)  # [batch_size, num_heads, depth*height*width, head_dim]
        out = out.permute(0, 1, 3, 2).contiguous()  # [batch_size, num_heads, head_dim, depth*height*width]
        out = out.view(batch_size, C, depth, height, width)  # [batch_size, C, depth, height, width]
        
        # Output
        out = self.gamma * out + x
        return out

# setting FPN with attention
class FPN3DWithMultiHeadAttention(nn.Module):
    def __init__(self, in_channels_list, out_channels, num_heads=8, stride_perform = False):
        super(FPN3DWithMultiHeadAttention, self).__init__()
        self.lateral_convs = nn.ModuleList()
        self.output_convs = nn.ModuleList()
        self.attention_modules = nn.ModuleList()
        self.attention_pos = []

        stride = (1,3,3) if stride_perform else (1,1,1)

        for in_channels in in_channels_list:
            lateral_conv = nn.Conv3d(in_channels[0], out_channels, kernel_size=3, padding=1, stride=stride)
            lateral_shape = [in_channels[1], math.ceil(in_channels[2]/stride[1]), math.ceil(in_channels[3]/stride[2])]
            output_stride = (0, 2-3*lateral_shape[1]+in_channels[2], 2-3*lateral_shape[2]+in_channels[3]) if stride_perform else (0, 0, 0)
            output_conv = nn.ConvTranspose3d(out_channels, out_channels, kernel_size=3, padding=1, stride=stride, output_padding=output_stride)
            attention_module = MultiHeadAttentionModule(out_channels, num_heads)
            pos = nn.Parameter(torch.zeros(1, out_channels, lateral_shape[0], lateral_shape[1], lateral_shape[2])).cuda()
            self.lateral_convs.append(lateral_conv)
            self.output_convs.append(output_conv)
            self.attention_modules.append(attention_module)
            self.attention_pos.append(pos)

    def forward(self, inputs):
        # laterals = [lateral_conv(x) for lateral_conv, x in zip(self.lateral_convs, inputs)]

        # for i in range(len(laterals) - 1, -1, -1):
        #     if i != len(laterals) - 1:
        #         upsampled = F.interpolate(laterals[i + 1], size=laterals[i].shape[2:], mode="trilinear", align_corners=False)
        #         laterals[i] += upsampled
        #     laterals[i] = self.attention_modules[i](laterals[i] + self.attention_pos[i])
        laterals = self.lateral_convs[-1](inputs[-1]) + self.attention_pos[-1]
        for i in range(len(inputs)):
            laterals = self.attention_modules[i](laterals)

        laterals = self.output_convs[-1](laterals)
        return [laterals]

# setting the final model for training
class Simple3DConvModelWithTripleCNNFPNAndAttention(nn.Module):
    def __init__(self, smplx_model_paths, input_channels=(4,121,111,31), fpn_out_channels=256, reduced_channels=256, dropout_rate=0.0):
        super(Simple3DConvModelWithTripleCNNFPNAndAttention, self).__init__()

        '''Detection part of model'''
        self.conv1 = nn.Conv3d(input_channels[3], 64, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1)

        # Calculate input sizes for FPN layers
        first_layer_size = [math.ceil(input_channels[0] / 2), math.ceil(input_channels[1] / 2), math.ceil(input_channels[2] / 2)]
        second_layer_size = [math.ceil(dim / 2) for dim in first_layer_size]
        third_layer_size = [math.ceil(dim / 2) for dim in second_layer_size]

        self.fpn = FPN3DWithMultiHeadAttention([
            [64, first_layer_size[0], first_layer_size[1], first_layer_size[2]],
            [128, second_layer_size[0], second_layer_size[1], second_layer_size[2]],
            [256, third_layer_size[0], third_layer_size[1], third_layer_size[2]]
        ], fpn_out_channels, stride_perform=False)

        # 重新计算展平后的尺寸：128 * 1 * 16 * 14
        self.conv1x1 = nn.Conv3d(fpn_out_channels, reduced_channels, kernel_size=1)
        self.maxpool_trans = nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,3,3))
        flattened_size = reduced_channels * 1 * (16//3) * (14//3)

        dim = 1024
        self.fc1 = nn.Linear(flattened_size, 1024)
        # self.dropout = nn.Dropout(0.3)
        
        # detection cropping
        self.register_buffer("trans_offset", torch.tensor([-3., 0.5, 0.0]))
        self.register_buffer("trans_scale", torch.tensor([6., 5.5, 3.0]))
        self.register_buffer("tensor_trans_scale", torch.tensor([121, 111, 31]))
        self.register_buffer("tensor_offset_min", torch.tensor([20, 16, 12]))
        self.register_buffer("tensor_offset_max", torch.tensor([20, 16, 12]))
        self.cropsize = self.tensor_offset_min + self.tensor_offset_max  # (3,)

        
        '''Mesh part of model'''
        self.conv1_m = nn.Conv3d(4, 64, kernel_size=3, stride=2, padding=1)
        self.conv2_m = nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv3_m = nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1)

        # Calculate input sizes for FPN layers
        first_layer_size = [math.ceil(self.cropsize[0] / 2), math.ceil(self.cropsize[1] / 2), math.ceil(self.cropsize[2] / 2)]
        second_layer_size = [math.ceil(dim / 2) for dim in first_layer_size]
        third_layer_size = [math.ceil(dim / 2) for dim in second_layer_size]
        self.fpn_m = FPN3DWithMultiHeadAttention([
                            [64, first_layer_size[0], first_layer_size[1], first_layer_size[2]],
                            [128, second_layer_size[0], second_layer_size[1], second_layer_size[2]],
                            [256, third_layer_size[0], third_layer_size[1], third_layer_size[2]]
                            ], fpn_out_channels*2, stride_perform=False)
        
        self.conv1x1_m = nn.Conv3d(fpn_out_channels*2, reduced_channels, kernel_size=1)
        flattened_size_m = reduced_channels * (self.cropsize[0]//8 * self.cropsize[1]//8 * self.cropsize[2]//8)
        
        dim = 1024
        self.fc1_m = nn.Linear(flattened_size_m, 1024)
        # self.dropout = nn.Dropout(dropout_rate)
        
        
        '''Output Head part of model'''
        dropout2 = dropout_rate
        
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
        self.fc2_trans1 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 128),
            nn.ReLU(),
            # nn.Dropout(dropout2),
            nn.Linear(128, 3),
            nn.Sigmoid()
        )

        self.fc2_trans2 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 128),
            nn.ReLU(),
            # nn.Dropout(dropout2),
            nn.Linear(128, 3),
            nn.Sigmoid()
        )


        self.fc2_gender = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout2),
            nn.Linear(64, 1),
            nn.Sigmoid()
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

    def forward(self, in_tensor):
        # assert not (torch.isnan(in_tensor) | torch.isinf(in_tensor)).any(), "NaN or Inf in x before fc2_gender"

        x = in_tensor.permute(0,4,1,2,3)
        # if torch.isnan(x).any():
        #     print("NaN detected in x after permute")
        # print(x.shape)
        # # print((x[0]-x[1]).sum())
        c1 = F.relu(self.conv1(x))
        # if torch.isnan(c1).any():
        #     print("NaN detected in conv1")
        c2 = F.relu(self.conv2(c1))
        # if torch.isnan(c2).any():
        #     print("NaN detected in conv2")
        c3 = F.relu(self.conv3(c2))
        
        # if torch.isnan(c3).any():
        #     print("NaN detected in c3")

        fpn_outs = self.fpn([c2, c3])
        # fpn_outs = [c3]
        
        # if torch.isnan(fpn_outs[-1]).any():
        #     print("NaN detected in fpn_outs[-1]")
        # print(fpn_outs[-1].shape)
        
        x = F.relu(self.conv1x1(fpn_outs[-1]))
        
        # if torch.isnan(x).any():
        #     print("NaN detected in x after conv1x1")
        # print(x.shape)
        # x = x.view(x.size(0), x.size(1), -1)
        # x = F.max_pool1d(x, kernel_size=x.size(-1)).squeeze(-1)
        x = self.maxpool_trans(x)
        x = x.view(x.size(0), -1)
        # print(x.shape)
    
        x = self.fc1(x)
        # if torch.isnan(x).any():
        #     print("NaN detected in x after fc1")
        pred_trans_norm1 = self.fc2_trans1(x)
        
        
        
        
        
        ''' Mesh part of model'''
        # mesh pose estimation input
        pred_trans_crop = pred_trans_norm1.clone()
        pred_trans_crop[:,-1] = 12/31
        x_crop, crop_start = self.crop_5d_asym_fixed(in_tensor, pred_trans_crop)
        # if torch.isnan(pred_trans_norm).any():
        #     print("NaN in pred_trans_norm")
        # if torch.isnan(x_crop).any():
        #     print("NaN in crop output")
        # print(x_crop.shape)
        
        c_crop1 = F.relu(self.conv1_m(x_crop))
        # print(c_crop1.shape)
        c_crop2 = F.relu(self.conv2_m(c_crop1))
        # print(c_crop2.shape)
        c_crop3 = F.relu(self.conv3_m(c_crop2))
        # print(c_crop3.shape)
        
        fpn_outs_m = self.fpn_m([c_crop2, c_crop3])
        
        x_m = F.relu(self.conv1x1_m(fpn_outs_m[-1]))
        x_m = x_m.view(x_m.size(0), -1)
        x_m = self.fc1_m(x_m)
        x = x + x_m  # combine detection and mesh features
        

        # output head
        gender_pred = self.fc2_gender(x)
        pred_betas = self.fc2_betas(x)
        pred_pose_body = self.fc2_pose_body(x)
        pred_root_orient = self.fc2_root_orient(x)
        pred_trans_norm2 = self.fc2_trans2(x)

        pred_trans1 = pred_trans_norm1 * self.trans_scale + self.trans_offset
        pred_trans2 = pred_trans_norm2 * self.trans_scale + self.trans_offset
        

        
        
        
                
        return pred_betas, pred_pose_body, pred_root_orient, pred_trans2, gender_pred, pred_trans1

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
    
    def crop_5d_asym_fixed(
        self,
        x: torch.Tensor,                 # (B, T, D, H, W)
        pred_trans_norm: torch.Tensor,   # (B, 3) in [0,1], order (D,H,W)
        mode: str = "nearest",
        align_corners: bool = True,
    ):
        assert x.dim() == 5, "x must be (B,T,D,H,W)"
        B, T, D, H, W = x.shape
        device = x.device

        # Offsets as long on correct device
        off_min = self.tensor_offset_min.to(device=device, dtype=torch.long).view(1, 3)  # (1,3)
        off_max = self.tensor_offset_max.to(device=device, dtype=torch.long).view(1, 3)  # (1,3)
        win_sz  = off_min + off_max                                                       # (1,3)
        Dz, Hy, Wx = int(win_sz[0,0].item()), int(win_sz[0,1].item()), int(win_sz[0,2].item())


        # Asymmetric start = center - off_min
        centers = pred_trans_norm * self.tensor_trans_scale.view(1, 3)                     # (B,3)
        centers = torch.round(centers).to(torch.long)                                      # nearest voxel
        start = centers - off_min                                                          # (B,3)

        # Clamp starts so full window fits
        max_start = torch.tensor([D - Dz, H - Hy, W - Wx], device=device, dtype=torch.long).view(1, 3)
        start = torch.minimum(torch.maximum(start, torch.zeros_like(start)), max_start)     # (B,3)

        # Normalize to [-1,1] for grid_sample; last dim = (x,y,z)
        z_base = torch.arange(Dz, device=device, dtype=torch.float32).view(1, Dz, 1, 1)
        y_base = torch.arange(Hy, device=device, dtype=torch.float32).view(1, 1, Hy, 1)
        x_base = torch.arange(Wx, device=device, dtype=torch.float32).view(1, 1, 1, Wx)

        z_idx = (start[:, 0].view(B, 1, 1, 1).float() + z_base).expand(B, Dz, Hy, Wx)
        y_idx = (start[:, 1].view(B, 1, 1, 1).float() + y_base).expand(B, Dz, Hy, Wx)
        x_idx = (start[:, 2].view(B, 1, 1, 1).float() + x_base).expand(B, Dz, Hy, Wx)

        z_norm = 2.0 * z_idx / max(D - 1, 1) - 1.0
        y_norm = 2.0 * y_idx / max(H - 1, 1) - 1.0
        x_norm = 2.0 * x_idx / max(W - 1, 1) - 1.0
        grid   = torch.stack([x_norm, y_norm, z_norm], dim=-1)  # (B, Dz, Hy, Wx, 3)

        # Sample: treat T as channels (C)
        out = F.grid_sample(
            x, grid, mode=mode, padding_mode="zeros", align_corners=align_corners
        )  # (B, T, Dz, Hy, Wx)

        return out, start  # integer (D0,H0,W0) per sample


if __name__ == "__main__":
    model = Simple3DConvModelWithTripleCNNFPNAndAttention(
        smplx_model_paths={
            'male': 'models/smplx/SMPLX_MALE.npz',
            'female': 'models/smplx/SMPLX_FEMALE.npz',
            'neutral': 'models/smplx/SMPLX_NEUTRAL.npz'
        }
    ).cuda()
    input_tensor = torch.randn(2, 4, 121, 111, 31).cuda()
    betas, pose_body, root_orient, trans, gender = model(input_tensor)
    print(betas.shape, pose_body.shape, root_orient.shape)
