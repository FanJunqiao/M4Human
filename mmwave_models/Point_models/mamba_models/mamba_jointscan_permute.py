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
    
    if first == "None":
        if reverse_order != None:
            all_pose_tensor = all_pose_tensor[:,:,reverse_order,:]
        return all_pose_tensor
    B,T,V,C = original_shape
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


def circular_permute_v(x: torch.Tensor, reverse_prob = 0.0):
    """
    Circularly permutes the V dimension of a [B, V, C] tensor starting from a random index vs.

    Args:
        x (Tensor): Input tensor of shape [B, V, C]
        return_vs (bool): Whether to return the start index vs

    Returns:
        permuted_x (Tensor): Permuted tensor
        vs (int, optional): The start index used for permutation
    """
    B, V, C = x.shape
    vs = torch.randint(0, V, (1,)).item()  # Random start index

    # Perform circular permutation on V dimension
    reverse_flag = np.random.rand()<reverse_prob
    permuted_x = torch.cat([x[:, vs:, :], x[:, :vs, :]], dim=1)

    if reverse_flag:
        permuted_x = torch.flip(permuted_x, dims = [1])


    return permuted_x, vs, reverse_flag


def reverse_circular_permute_v(permuted_x: torch.Tensor, vs: int, reverse_flag: bool):
    """
    Reverses the circular permutation on the V dimension of a [B, V, C] tensor.

    Args:
        permuted_x (Tensor): Tensor after circular permutation [B, V, C]
        vs (int): The start index used in the original permutation

    Returns:
        original_x (Tensor): Tensor with original V order restored
    """
    B, V, C = permuted_x.shape
    if reverse_flag:
        torch.flip(permuted_x, dims=[1])
    return torch.cat([permuted_x[:, -vs:, :], permuted_x[:, :-vs, :]], dim=1) if vs > 0 else permuted_x

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



        emb = self.time_embed(timestep_embedding(timesteps, self.args.d_model))
        # print("emb", emb.shape)
        

        if mod is not None:
            mod_proj = self.cond_embed2(mod.reshape(B, -1))#
            emb = (emb + mod_proj).unsqueeze(dim=1)

        h = self.embedding(x)
        h = h + self.sequence_embedding


        h, reverse_order = reorder_mamba_input(all_pose_tensor=h, v_order=self.vorder, reverse=False, first="None")
        b,t,v,c = h.shape
        # b,t,v,_ = x.shape
        # if self.training:
        #     mask_ratio = 0.8
        #     binary_mask = torch.rand(B, v, device=x.device) < mask_ratio  # bool mask
        #     # 2. Expand mask to shape (B, T, V, 1)
        #     binary_mask = binary_mask.unsqueeze(1).unsqueeze(-1).expand(B, T, v, 3).float()  # shape (B, T, V, C)
        #     x_after = x * binary_mask + torch.randn_like(x) * (1-binary_mask)



        
        

        

        i = 0
        prelist = []
        for layer in self.layers_M:
            if i < (self.args.n_layer // 2):
                # mamba joint scan
                prelist.append(h)
                h = h.permute(0, 2, 1, 3).reshape(b,v,t*c)
                if self.training:
                    h, vs, reverse_flag = circular_permute_v(h,reverse_prob=0.0)
                h = layer(h, emb)
                if self.training:
                    h = reverse_circular_permute_v(h, vs, reverse_flag=reverse_flag)

                # transformer freq scan
                h = h.view(b,v,t,c).permute(0,2,1,3).reshape(b,t,v*c)
                h = self.layer_T[i](h, emb)
                h = h.view(b,t,v,c)
                # print("h in", i, h.shape)
            elif i >= (self.args.n_layer // 2):
                h = h.permute(0, 2, 1, 3).reshape(b,v,t*c)
                if self.training:
                    h, vs, reverse_flag = circular_permute_v(h,reverse_prob=0.0)
                h = layer(h, emb)
                if self.training:
                    h = reverse_circular_permute_v(h, vs, reverse_flag=reverse_flag)
                h = h.view(b,v,t,c).permute(0,2,1,3).reshape(b,t,v*c)
                h = self.layer_T[i](h, emb)
                h = h.view(b,t,v,c)
                h += prelist[-1]
                prelist.pop()
                # print("h out", i, h.shape)
            i += 1
    
        # for trans_module in self.layer_T:
        #     attn_h = trans_module(h, emb)
        #     h = attn_h + h


        h = h.view(b,t,v*c)
        h = self.norm_f(h)
        h = h.view(b,t,v,c)
        h = reorder_mamba_output(h, h.shape, reverse_order=reverse_order, reverse=False, first="None")
        logits = self.lm_head(h)


        return logits.reshape(B,T,C).contiguous()