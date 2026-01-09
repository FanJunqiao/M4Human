import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import sys 
import os
import time
import copy





BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'modules'))

from .p4trans.point_4d_convolution import *
from .p4trans.transformer import * # alias for models' transformer


class P4Transformer(nn.Module):
    def __init__(self, radius, nsamples, spatial_stride,                                # P4DConv: spatial
                 temporal_kernel_size, temporal_stride,                                 # P4DConv: temporal
                 emb_relu,                                                              # embedding: relu
                 dim, depth, heads, dim_head,                                           # transformer
                 mlp_dim, num_classes,                                                  # output
                 dropout1=0.0, dropout2=0.0,                                            # dropout
                ):                                           
        super().__init__()




        self.tube_embedding = P4DConv(in_planes=3, mlp_planes=[dim], mlp_batch_norm=[False], mlp_activation=[False],
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
        self.joint_num = 17

        self.joint_template = nn.Parameter(torch.rand(size = (self.joint_num, 1024)))
        self.joint_posembeds_vector = torch.tensor(self.get_positional_embeddings1(self.joint_num, 1024)).float().cuda()
        # point Prediction Head
        input_dim = dim
        mid_dim = 64
        

        self.dim_reduce_head = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim//2),
            nn.ReLU(),
            nn.Linear(input_dim//2, mid_dim)
        )
        
        input_dim = mid_dim
        self.point_prediction_heads = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim//2),
            nn.ReLU(),
            nn.Linear(input_dim//2, 3)
        )

        self.var_prediction_heads = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim//2),
            nn.ReLU(),
            nn.Linear(input_dim//2, 3),
        )



    def forward(self, radar):    #, print_depth=None, print_radar=None
        print(radar.shape)
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
        print(xyzts.shape, features.shape)

        xyzts = torch.reshape(input=xyzts, shape=(xyzts.shape[0], xyzts.shape[1]*xyzts.shape[2], xyzts.shape[3]))                           # [B, L*n, 4]

        features = features.permute(0, 1, 3, 2)                                                                                             # [B, L,   n, C]
        features = torch.reshape(input=features, shape=(features.shape[0], features.shape[1]*features.shape[2], features.shape[3]))         # [B, L*n, C]

        print(xyzts.shape, features.shape)
        
        xyzts_embd = self.pos_embedding(xyzts.permute(0, 2, 1)).permute(0, 2, 1)

        embedding = xyzts_embd + features
        radar_feature = features
        radar_position = xyzts

        mid_time1 = time.time()
        # open for template embedding
        # joint_embedding = self.joint_template.expand(radar.shape[0], -1, -1) + self.joint_posembeds_vector
        joint_embedding = torch.rand(size = (radar.shape[0], self.joint_num, 1024)).cuda() + self.joint_template.expand(radar.shape[0], -1, -1) + self.joint_posembeds_vector
        embedding = torch.cat([joint_embedding, embedding], dim=1)

        # print_anchors(radar_position, radar)
        

        if self.emb_relu:
            embedding = self.emb_relu(embedding)


        # open for template embedding
        output = self.transformer(embedding)
        joint_embedding = output[:, :self.joint_num, :]
        joint_embedding = self.dim_reduce_head(joint_embedding)
        mid_time2 = time.time()
        joint_embed_out = joint_embedding
        output = self.point_prediction_heads(joint_embedding)
        # output_var = torch.exp(self.var_prediction_heads(joint_embedding))
        final_time = time.time()

        # print("time:", (mid_time1 - temp_start_time, mid_time2 - mid_time1, final_time - temp_start_time))




        # # For max-pooling algorithm
        # # output = torch.mean(input=features, dim=1, keepdim=False, out=None)
        # output = self.transformer(embedding)
        # print(sum(param.numel() for param in self.transformer.parameters()))

        # output = torch.max(input=output, dim=1, keepdim=False, out=None)[0]
        # joint_embed_out = output
        
        # output = self.mlp_head(output).view(-1, 17, 3).cuda()
        # output_var = torch.rand(output.shape[0], output.shape[1], output.shape[2])

        return output, None, joint_embed_out, radar_feature, radar_position
    def get_positional_embeddings1(self, sequence_length, d):
        result = np.ones([1, sequence_length, d])
        for i in range(sequence_length):
            for j in range(d):
                result[0][i][j] = np.sin(i / (10000 ** (j / d))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / d)))
        return result
    
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
