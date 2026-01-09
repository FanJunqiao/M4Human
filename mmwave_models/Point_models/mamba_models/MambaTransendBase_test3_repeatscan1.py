import torch
from models.mamba_simple_jointscan import *
from models.MotionTransformer import *
import copy
from einops import rearrange, repeat, einsum

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

def for_h_mamba(h, shape_list, flat_order, use_reverse=False, use_parts=False):
    """
    Args:
        h:           Tensor of shape [B, V, C]
        flat_order:  List[int], flattened reordered joints
        shape_list:  Tuple (B, V, C, N, P)
        use_reverse: Whether to include reversed h in h_forrev
        use_parts:   Whether to extract h_parts

    Returns:
        h_forrev: [B or 2B, V, C]
        h_parts:  [N*B, P, C] if use_parts else None
    """
    B, V, C, N, P = shape_list
    assert V == N * P, "Mismatch: V must equal N * P"

    h_for = h                                 # always included
    h_rev = h.flip(dims=[1]) if use_reverse else None

    if use_reverse:
        h_forrev = torch.cat([h_for, h_rev], dim=0)  # [2B, V, C]
    else:
        h_forrev = h_for.clone()                    # [B, V, C]

    h_parts = None
    if use_parts:
        h_parts = h[:, flat_order, :].reshape(N * B, P, C)

    return h_forrev, h_parts


def reverse_h_mamba(h_forrev_out, h_parts_out, shape_list, reverse_order, use_reverse=False, use_parts=False):
    """
    Args:
        h_forrev_out: [B or 2B, V, C]
        h_parts_out:  [N*B, P, C] or None
        shape_list:   Tuple (B, V, C, N, P)
        reverse_order: List[int], inverse of flat_order
        use_reverse:  Whether h_rev was included
        use_parts:    Whether h_parts was included

    Returns:
        h_avg: [B, V, C]
    """
    B, V, C, N, P = shape_list
    assert V == N * P, "Mismatch: V must equal N * P"

    terms = []
    count = 1  # forward is always on

    h_for = h_forrev_out[:B]
    terms.append(h_for)

    if use_reverse:
        h_rev = h_forrev_out[B:2*B].flip(dims=[1])
        terms.append(h_rev)
        count += 1

    if use_parts and h_parts_out is not None:
        h_parts = h_parts_out.reshape(B, V, C)
        h_parts = h_parts[:, reverse_order, :]
        terms.append(h_parts)
        count += 1

    h_avg = sum(terms) / count
    return h_avg

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
        print(args)
        self.vorder = [9,8,7,10,11,12,11,10,7,13,14,15,14,13,7,6,0,1,2,1,0,6,3,4,5,4,3,6,7,8]
        # self.vorder = list(range(16))
        self.repeat_joint_num = len(self.vorder)
        self.joint_num = 16
        self.T_num = 20
        self.joint_latent_dim = self.args.d_model // self.joint_num
        
        

        self.trans_latent = self.joint_latent_dim * self.repeat_joint_num
        self.mamb_latent = self.joint_latent_dim * self.T_num
        self.args.d_model = self.mamb_latent

        self.vorder = [9,8,7,10,11,12,11,10,7,13,14,15,14,13,7,6,0,1,2,1,0,6,3,4,5,4,3,6,7,8]
        self.vorder_parts = [[0,1,14,27,28,29],[2,3,4,5,6,7],[8,9,10,11,12,13],[15,16,17,18,19,20],[21,22,23,24,25,26]]
        self.flat_part_indices = [0, 1, 14, 27, 28, 29, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
        self.reverse_flat_part_indices = [0, 1, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 2, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 3, 4, 5]

        self.num_parts = len(self.vorder_parts)
        self.part_len = len(self.vorder_parts[0])  # assume all equal
        

        self.positional_embedding = PositionalEmbedding(self.args.d_model)
        self.sequence_embedding = nn.Parameter(torch.zeros(1, self.T_num, self.joint_num, self.joint_latent_dim))
        
        self.embedding = nn.Linear(3, self.joint_latent_dim)
        self.cond_embed2 = nn.Linear(self.args.vocab_size * self.args.cond_length, self.args.temp_emb_dim)
        self.time_embed = nn.Sequential(
            nn.Linear(self.args.d_model, self.args.temp_emb_dim),
            nn.SiLU(),
            nn.Linear(self.args.temp_emb_dim, self.args.temp_emb_dim),
        )

        


        self.layers_M = nn.ModuleList([ResidualBlock(args, self.mamb_latent) for _ in range(args.n_layer)])
        # self.layer_T = nn.ModuleList([ResidualBlock(args, self.trans_latent) for _ in range(args.n_layer)])
        self.layer_T = nn.ModuleList()
        for i in range(self.args.n_layer):
            self.layer_T.append(
                TemporalDiffusionTransformerDecoderLayer(
                    latent_dim=self.trans_latent,
                    time_embed_dim=self.args.temp_emb_dim,
                    ffn_dim=2*self.args.d_model,
                    num_head=self.args.num_T_in_M_head,
                    dropout=self.args.dropout,
                    causal_mask = self.args.cfg.causal_mask,
                )
            )

        self.norm_f = RMSNorm(self.trans_latent)

        self.lm_head = nn.Linear(self.joint_latent_dim, 3)
        


    def forward(self, x, timesteps, mod=None, sqrt_alpha_hat = None):
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



        emb = self.time_embed(timestep_embedding(timesteps, self.args.d_model)).unsqueeze(dim=1)
        # print("emb", emb.shape)
        

        if mod is not None:
            mod_proj = self.cond_embed2(mod.reshape(B, -1)).unsqueeze(dim=1)
            emb = (emb + mod_proj)

        h = self.embedding(x)
        h = h# + self.sequence_embedding


        h, reverse_order = reorder_mamba_input(all_pose_tensor=h, v_order=self.vorder, reverse=False, first="None")
        b,t,v,c = h.shape
        N, P = self.num_parts, self.part_len
        reverse_flag = False
        parts_flag = False
        for_rev_num = 1
        if reverse_flag: for_rev_num += 1




        i = 0
        prelist = []
        for layer in self.layers_M:
            if i < (self.args.n_layer // 2):
                # mamba joint scan
                prelist.append(h)
                h = h.permute(0, 2, 1, 3).reshape(b,v,t*c)
                h_for_rev, h_parts = for_h_mamba(h, (b,v,t*c,N,P), self.flat_part_indices, use_parts=parts_flag, use_reverse=reverse_flag)
                if self.training:
                    h_for_rev, vs, reverse_flags = circular_permute_v(h_for_rev,reverse_prob=0.0)
                h_for_rev = h_for_rev+layer(h_for_rev, emb.repeat(for_rev_num,1,1))
                if h_parts != None:
                    h_parts = h_parts+layer(h_parts, emb.repeat(N,1,1))
                if self.training:
                    h_for_rev = reverse_circular_permute_v(h_for_rev, vs, reverse_flags=reverse_flags)
                h = reverse_h_mamba(h_for_rev, h_parts, (b,v,t*c,N,P), self.reverse_flat_part_indices, use_parts=parts_flag, use_reverse=reverse_flag)

                # transformer freq scan
                h = h.view(b,v,t,c).permute(0,2,1,3).reshape(b,t,v*c)
                h = h+self.layer_T[i](h, emb)
                h = h.view(b,t,v,c)
                # print("h in", i, h.shape)
            elif i >= (self.args.n_layer // 2):

                # mamba joint scan
                h = h.permute(0, 2, 1, 3).reshape(b,v,t*c)
                h_for_rev, h_parts = for_h_mamba(h, (b,v,t*c,N,P), self.flat_part_indices, use_parts=parts_flag, use_reverse=reverse_flag)
                if self.training:
                    h_for_rev, vs, reverse_flags = circular_permute_v(h_for_rev,reverse_prob=0.0)
                h_for_rev = h_for_rev+layer(h_for_rev, emb.repeat(for_rev_num,1,1))
                if h_parts != None:
                    h_parts = h_parts+layer(h_parts, emb.repeat(N,1,1))
                if self.training:
                    h_for_rev = reverse_circular_permute_v(h_for_rev, vs, reverse_flags=reverse_flags)
                h = reverse_h_mamba(h_for_rev, h_parts, (b,v,t*c,N,P), self.reverse_flat_part_indices, use_parts=parts_flag, use_reverse=reverse_flag)

                 # transformer freq scan
                h = h.view(b,v,t,c).permute(0,2,1,3).reshape(b,t,v*c)
                h = h+self.layer_T[i](h, emb)
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