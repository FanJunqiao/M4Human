import torch
from models.mamba_simple_jointscan import *
from models.MotionTransformer import *
import copy
from einops import rearrange, repeat, einsum

def compute_limb_lengths(pose: torch.Tensor):
    """
    Compute limb lengths from a pose tensor of shape [B, C], where C = (J-1) * 3.
    Assumes joint 0 (root) is added as zeros and determines parent list based on J.

    Args:
        pose (torch.Tensor): Tensor of shape [B, C] with (J-1)*3 values. Root joint (index 0) is assumed missing.

    Returns:
        torch.Tensor: Limb lengths of shape [B, num_limbs]
    """
    B, C = pose.shape
    J = C // 3 + 1  # total number of joints (including root)

    # Add root joint (joint 0) as zeros
    pose = pose.view(B, J - 1, 3)
    root = torch.zeros(B, 1, 3, device=pose.device, dtype=pose.dtype)
    full_pose = torch.cat([root, pose], dim=1)  # [B, J, 3]

    # Select parent list
    if J == 15:
        parent = [-1, 0, 1, 2, 3, 1, 5, 6, 0, 8, 9, 0, 11, 12, 1]
    elif J == 17:
        parent = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]
    elif J == 22:
        parent = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
    else:
        raise ValueError(f"Unsupported joint count: {J}. Provide parent list manually.")

    # Compute limb lengths
    limb_lengths = []
    for j, p in enumerate(parent):
        if p == -1:
            continue
        limb = torch.norm(full_pose[:, j] - full_pose[:, p], dim=-1)  # [B]
        limb_lengths.append(limb.unsqueeze(-1))  # [B, 1]

    return torch.cat(limb_lengths, dim=-1)  # [B, num_limbs]


def reorder_mamba_input(
    all_pose_tensor: torch.Tensor,
    v_order=None,
    reverse=False,
    first="T"
):
    reverse_order = None
    B,T,V,C = all_pose_tensor.shape
    if v_order != None:
        order = [0]*len(v_order)
        reverse_order = [0]*V
        for i in range(len(v_order)):
            order[i] = v_order[i]
            reverse_order[v_order[i]] = i
        all_pose_tensor = all_pose_tensor[:,:,order,:]

    if first == "None":
        return all_pose_tensor, reverse_order
    if first == "T" and reverse == False:
        all_pose_tensor =  all_pose_tensor.permute(0, 2, 1, 3).reshape(B,T*V,C)
    elif first == "V" and reverse == False:
        all_pose_tensor = all_pose_tensor.reshape(B,T*V,C)
    elif first == "T" and reverse == True:
        all_pose_tensor = torch.flip(all_pose_tensor, dims=[1,2]).permute(0, 2, 1, 3).reshape(B,T*V,C)
    elif first == "V" and reverse == True:
        all_pose_tensor = torch.flip(all_pose_tensor, dims=[1,2]).reshape(B,T*V,C)
    return all_pose_tensor, reverse_order
    
    
def reorder_mamba_output(
    all_pose_tensor: torch.Tensor,
    original_shape,
    reverse_order=None,
    reverse=False,
    first="T",
):
    B,T,V,C = original_shape
    if first == "None":
        if reverse_order != None:
            all_pose_tensor = all_pose_tensor[:,:,reverse_order,:]
        return all_pose_tensor
    if first == "T" and reverse == False:
        all_pose_tensor =  all_pose_tensor.reshape(B,V,T,C).permute(0, 2, 1, 3)
    elif first == "V" and reverse == False:
        all_pose_tensor = all_pose_tensor.reshape(B,T,V,C)
    elif first == "T" and reverse == True:
        all_pose_tensor = torch.flip(all_pose_tensor.reshape(B,V,T,C).permute(0, 2, 1, 3), dims=[1,2])
    elif first == "V" and reverse == True:
        all_pose_tensor = torch.flip(all_pose_tensor.reshape(B,T,V,C), dims=[1,2])

    if reverse_order != None:
        all_pose_tensor = all_pose_tensor[:,:,reverse_order,:]
    return all_pose_tensor

def circular_permute_v(x: torch.Tensor, reverse_prob=0.0):
    """
    Batch-wise circular permutation on the V dimension of a [B, V, C] tensor,
    with optional per-sample reversal.

    Args:
        x (Tensor): Input tensor of shape [B, V, C]
        reverse_prob (float): Probability to reverse V dimension per sample

    Returns:
        permuted_x (Tensor): Tensor after batch-wise circular permutation and optional reverse
        vs (Tensor): Start index used for circular permutation per sample (shape [B])
        reverse_flags (Tensor): Bool tensor indicating which samples were reversed (shape [B])
    """
    B, V, C = x.shape
    device = x.device

    # Random start index for each sample
    vs = torch.randint(0, V, (B,), device=device)  # shape: (B,)
    idx = (torch.arange(V, device=device).unsqueeze(0) + V - vs.unsqueeze(1)) % V  # shape: (B, V)
    idx_expanded = idx.unsqueeze(-1).expand(-1, -1, C)  # shape: (B, V, C)
    permuted_x = torch.gather(x, dim=1, index=idx_expanded)

    # Per-sample random reverse flag
    reverse_flags = torch.rand(B, device=device) < reverse_prob  # shape: (B,)
    if reverse_flags.any():
        print("reverse")
        permuted_x[reverse_flags] = permuted_x[reverse_flags].flip(dims=[1])

    return permuted_x, vs, reverse_flags

def reverse_circular_permute_v(permuted_x: torch.Tensor, vs: torch.Tensor, reverse_flags: torch.Tensor):
    """
    Reverses batch-wise circular permutation and optional reversal on the V dimension.

    Args:
        permuted_x (Tensor): Tensor after circular permutation [B, V, C]
        vs (Tensor): Start index used for circular permutation per sample [B]
        reverse_flags (Tensor): Bool tensor indicating which samples were reversed [B]

    Returns:
        x_restored (Tensor): Tensor with original V order restored [B, V, C]
    """
    B, V, C = permuted_x.shape
    device = permuted_x.device

    # Reverse the reverse
    if reverse_flags.any():
        permuted_x[reverse_flags] = permuted_x[reverse_flags].flip(dims=[1])

    # Undo circular permutation
    idx = (torch.arange(V, device=device).unsqueeze(0) + vs.unsqueeze(1)) % V  # shape: (B, V)
    idx_expanded = idx.unsqueeze(-1).expand(-1, -1, C)  # shape: (B, V, C)
    x_restored = torch.gather(permuted_x, dim=1, index=idx_expanded)

    return x_restored

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    if len(timesteps.shape) == 1:
        
        args = timesteps[:, None].float() * freqs[None, :]
    elif len(timesteps.shape) == 2:
        args = timesteps[:, :, None].float() * freqs[None, None, :]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[..., :1])], dim=-1)
    
    return embedding

class PositionalEmbedding(torch.nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.num_channels//2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x
    
class FourierEmbedding(torch.nn.Module):
    def __init__(self, num_channels, scale=16):
        super().__init__()
        self.register_buffer('freqs', torch.randn(num_channels // 2) * scale)

    def forward(self, x):
        x = x.ger((2 * np.pi * self.freqs).to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x

class Mamba(nn.Module):
    def __init__(self, args: ModelArgs):
        """My Mamba model."""
        super().__init__()
        self.args = args
        self.args.d_model = 520
        self.args.temp_emb_dim = 520
        print(args)
        self.vorder = [13,0,1,2,3,2,1,0,7,8,9,8,7,0,10,11,12,11,10,0,4,5,6,5,4,0]
        # self.vorder = list(range(16))
        self.repeat_joint_num = len(self.vorder)
        self.joint_num = 14
        self.T_num = 10
        self.joint_latent_dim = self.args.d_model // self.repeat_joint_num
        
        

        self.trans_latent = self.joint_latent_dim * self.repeat_joint_num
        self.mamb_latent = self.joint_latent_dim * self.T_num
        print(self.args.d_model,self.args.temp_emb_dim,self.trans_latent, self.mamb_latent)


        self.positional_embedding = PositionalEmbedding(self.args.d_model)
        self.sequence_embedding = nn.Parameter(torch.zeros(1, self.T_num, self.joint_num, self.joint_latent_dim))
        
        self.embedding = nn.Linear(3, self.joint_latent_dim)
        self.cond_embed2 = nn.Linear(self.args.vocab_size * self.args.cond_length, self.args.temp_emb_dim)
        self.time_embed = nn.Sequential(
            nn.Linear(self.args.d_model, self.args.temp_emb_dim),
            nn.SiLU(),
            nn.Linear(self.args.temp_emb_dim, self.args.temp_emb_dim),
        )

        

        # self.args.d_model = self.mamb_latent
        self.layers_M = nn.ModuleList([ResidualBlock(args, self.mamb_latent) for _ in range(args.n_layer)])
        # self.layer_T = nn.ModuleList([ResidualBlock(args, self.trans_latent) for _ in range(args.n_layer)])
        self.layer_T = nn.ModuleList()
        for i in range(self.args.n_layer):
            self.layer_T.append(
                TemporalDiffusionTransformerDecoderLayer(
                    latent_dim=self.trans_latent,
                    time_embed_dim=self.args.temp_emb_dim,
                    ffn_dim=2*512,
                    num_head=self.args.num_T_in_M_head,
                    dropout=self.args.dropout,
                    causal_mask = self.args.cfg.causal_mask,
                )
            )
        print(self.trans_latent,self.args.temp_emb_dim,2*512,self.args.num_T_in_M_head)

        self.norm_f = RMSNorm(self.trans_latent)

        self.lm_head = nn.Linear(self.joint_latent_dim, 3)
        


    def forward(self, x, timesteps, mod=None, root=None, sqrt_alpha_hat = None):
        """
        Args:
            input_ids (long tensor): shape (b, l)    (See Glossary at top for definitions of b, l, d_in, n...)
    
        Returns:
            logits: shape (b, l, vocab_size)

        Official Implementation:
            class MambaLMHeadModel, https://github.com/state-spaces/mamba/blob/main/mamba_ssm/models/mixer_seq_simple.py#L173

        """
        B, T, C = x.shape
        x = x.reshape(B,T,C//3,3)
        # print("input", x.shape)



        emb = self.time_embed(timestep_embedding(timesteps, self.args.d_model))
        # print("emb", emb.shape)
        

        if mod is not None:
            mod_proj = self.cond_embed2(mod.reshape(B, -1))#

            emb = (emb + mod_proj).unsqueeze(dim=1)

        h = self.embedding(x)
        h = h + self.sequence_embedding



        h, reverse_order = reorder_mamba_input(all_pose_tensor=h, v_order=self.vorder, reverse=False, first="None")
        b,t,v,c = h.shape

        i = 0
        prelist = []
        for layer in self.layers_M:
            if i < (self.args.n_layer // 2):
                # # mamba joint scan
                prelist.append(h)
                h = h.permute(0, 2, 1, 3).reshape(b,v,t*c)
                if self.training:
                    h, vs, reverse_flags = circular_permute_v(h,reverse_prob=0.0)
                h = layer(h, emb)
                if self.training:
                    h = reverse_circular_permute_v(h, vs, reverse_flags=reverse_flags)
                h = h.view(b,v,t,c).permute(0,2,1,3)

                # # transformer freq scan
                h = h.reshape(b,t,v*c)
                h = self.layer_T[i](h, emb)
                h = h.view(b,t,v,c)
                # print("h in", i, h.shape)
            elif i >= (self.args.n_layer // 2):
                h = h.permute(0, 2, 1, 3).reshape(b,v,t*c)
                if self.training:
                    h, vs, reverse_flags = circular_permute_v(h,reverse_prob=0.0)
                h = layer(h, emb)
                if self.training:
                    h = reverse_circular_permute_v(h, vs, reverse_flags=reverse_flags)
                h = h.view(b,v,t,c).permute(0,2,1,3)
                h = h.reshape(b,t,v*c)
                h = self.layer_T[i](h, emb)
                h = h.view(b,t,v,c)
                h += prelist[-1]
                prelist.pop()
                # print("h out", i, h.shape)
            i += 1
    


        # h = h.view(b,t,v*c)
        # h = self.norm_f(h)
        # logits = self.lm_head(h)
        # logits = logits.view(b,t,v,3)
        # logits = reorder_mamba_output(logits, logits.shape, reverse_order=reverse_order, reverse=False, first="None")
        # logits = logits.reshape(B,T,C).contiguous()
        

        # output head
        h = h.view(b,t,v*c)
        h = self.norm_f(h)
        h = h.view(b,t,v,c)
        h = reorder_mamba_output(h, h.shape, reverse_order=reverse_order, reverse=False, first="None")

        logits = self.lm_head(h)
        logits = logits.reshape(B,T,C).contiguous()

        return logits