import torch
import torch.nn.functional as F
from torch import layer_norm, nn
import numpy as np

import math

from utils import *
from einops import rearrange, repeat, einsum



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
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad for all the networks.

    Args:
        nets (nn.Module | list[nn.Module]): A list of networks or a single
            network.
        requires_grad (bool): Whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class StylizationBlock(nn.Module):

    def __init__(self, latent_dim, time_embed_dim, dropout):
        super().__init__()
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, 2 * latent_dim),
        )
        self.norm = nn.LayerNorm(latent_dim)
        self.out_layers = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(nn.Linear(latent_dim, latent_dim)),
        )

    def forward(self, h, emb):
        """
        h: B, T, D
        emb: B, D
        """
        # B, 1, 2D
        if len(emb.shape) == 2:
            emb_out = self.emb_layers(emb).unsqueeze(1)
        else: 
            emb_out = self.emb_layers(emb)
        # scale: B, 1, D / shift: B, 1, D
        scale, shift = torch.chunk(emb_out, 2, dim=2)
        h = self.norm(h) * (1 + scale) + shift
        h = self.out_layers(h)
        return h


class FFN(nn.Module):

    def __init__(self, latent_dim, ffn_dim, dropout, time_embed_dim):
        super().__init__()
        self.linear1 = nn.Linear(latent_dim, ffn_dim)
        self.linear2 = zero_module(nn.Linear(ffn_dim, latent_dim))
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.proj_out = StylizationBlock(latent_dim, time_embed_dim, dropout)

    def forward(self, x, emb):
        y = self.linear2(self.dropout(self.activation(self.linear1(x))))
        y = x + self.proj_out(y, emb)
        return y


class TemporalSelfAttention(nn.Module):

    def __init__(self, latent_dim, num_head, dropout, time_embed_dim, causal_mask = False):
        super().__init__()
        self.num_head = num_head
        self.norm = nn.LayerNorm(latent_dim)
        self.query = nn.Linear(latent_dim, latent_dim, bias=False)
        self.key = nn.Linear(latent_dim, latent_dim, bias=False)
        self.value = nn.Linear(latent_dim, latent_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.proj_out = StylizationBlock(latent_dim, time_embed_dim, dropout)
        self.causal_mask = causal_mask

    def forward(self, x, emb):
        """
        x: B, T, D
        """
        B, T, D = x.shape
        H = self.num_head
        # B, T, 1, D
        query = self.query(self.norm(x)).unsqueeze(2)
        # B, 1, T, D
        key = self.key(self.norm(x)).unsqueeze(1)
        query = query.view(B, T, H, -1)
        key = key.view(B, T, H, -1)
        # B, T, T, H
        attention = torch.einsum('bnhd,bmhd->bnmh', query, key) / math.sqrt(D // H)
        # if self.causal_mask: # and self.training == False:
        #     mask = torch.ones(T, T).tril().bool().cuda()
        #     mask = mask[None,:,:,None].repeat(B,1,1,H)
        #     attention.masked_fill_(~mask, float('-inf'))
        weight = self.dropout(F.softmax(attention, dim=2))
        value = self.value(self.norm(x)).view(B, T, H, -1)
        if self.causal_mask:# and self.training == True:
            mask = torch.ones(T, T).tril().cuda()
            mask = mask[None,:,:,None].repeat(B,1,1,H)
            weight *= mask
        y = torch.einsum('bnmh,bmhd->bnhd', weight, value).reshape(B, T, D)
        y = x + self.proj_out(y, emb)
        return y


class TemporalCrossAttention(nn.Module):

    def __init__(self, latent_dim, mod_dim, num_head, dropout, time_embed_dim):
        super().__init__()
        self.num_head = num_head
        self.norm = nn.LayerNorm(latent_dim)
        self.text_norm = nn.LayerNorm(mod_dim)
        self.query = nn.Linear(latent_dim, latent_dim, bias=False)
        self.key = nn.Linear(mod_dim, latent_dim, bias=False)
        self.value = nn.Linear(mod_dim, latent_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.proj_out = StylizationBlock(latent_dim, time_embed_dim, dropout)
    
    def forward(self, x, xf, emb):
        """
        x: B, T, D
        xf: B, N, L
        """
        B, T, D = x.shape
        N = xf.shape[1]
        H = self.num_head
        # B, T, 1, D
        query = self.query(self.norm(x)).unsqueeze(2)
        # B, 1, N, D
        key = self.key(self.text_norm(xf)).unsqueeze(1)
        query = query.view(B, T, H, -1)
        key = key.view(B, N, H, -1)
        # B, T, N, H
        attention = torch.einsum('bnhd,bmhd->bnmh', query, key) / math.sqrt(D // H)
        weight = self.dropout(F.softmax(attention, dim=2))
        value = self.value(self.text_norm(xf)).view(B, N, H, -1)
        y = torch.einsum('bnmh,bmhd->bnhd', weight, value).reshape(B, T, D)
        y = x + self.proj_out(y, emb)
        return y

    
class TemporalDiffusionTransformerDecoderLayer(nn.Module):

    def __init__(self,
                 latent_dim=32,
                 time_embed_dim=128,
                 ffn_dim=256,
                 num_head=4,
                 dropout=0.5,
                 causal_mask = False
                 ):
        super().__init__()
        self.sa_block = TemporalSelfAttention(
            latent_dim, num_head, dropout, time_embed_dim, causal_mask=causal_mask)
        self.ffn = FFN(latent_dim, ffn_dim, dropout, time_embed_dim)

    def forward(self, x, emb):
        x = self.sa_block(x, emb)
        x = self.ffn(x, emb)
        return x


class MotionTransformer(nn.Module):
    def __init__(self,
                 input_feats,
                 num_frames=240,
                 latent_dim=512,
                 ff_size=1024,
                 num_layers=8,
                 num_heads=8,
                 dropout=0.2,
                 activation="gelu",
                 args = None,
                 **kargs):
        super().__init__()

        self.num_frames = num_frames
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ff_size = ff_size
        self.dropout = dropout
        self.activation = activation
        self.input_feats = input_feats
        self.time_embed_dim = latent_dim
        self.sequence_embedding = nn.Parameter(torch.randn(num_frames, latent_dim))
        self.args = args

        # Input Embedding
        self.joint_embed = nn.Linear(self.input_feats, self.latent_dim)
        
        if self.args.cond_length == self.args.vocab_length:
            self.cond_embed = nn.Linear(self.args.vocab_size , self.latent_dim) # * self.args.cond_length
        else:
            self.cond_embed = nn.Linear(self.args.vocab_size * self.args.cond_length, self.latent_dim)
        
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
        )

        self.alpha_embed = nn.Sequential(
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
        )
        # self.alpha_embed = nn.Sequential(
        #     nn.Linear(1, self.args.temp_emb_dim),
        #     nn.SiLU(),
        #     nn.Linear(self.args.temp_emb_dim, self.args.temp_emb_dim),
        # )
        self.t_embedding_type = self.args.cfg.t_embedding
        self.conv_size = 1
        self.alpha_conv =nn.Conv1d(
            in_channels=self.time_embed_dim,
            out_channels=self.time_embed_dim,
            kernel_size=self.conv_size,
            padding = self.conv_size - 1,
            padding_mode = "replicate",
        )

        self.temporal_decoder_blocks = nn.ModuleList()
        for i in range(num_layers):
            self.temporal_decoder_blocks.append(
                TemporalDiffusionTransformerDecoderLayer(
                    latent_dim=latent_dim,
                    time_embed_dim=self.time_embed_dim,
                    ffn_dim=ff_size,
                    num_head=num_heads,
                    dropout=dropout,
                )
            )

        # Output Module
        self.out = zero_module(nn.Linear(self.latent_dim, self.input_feats))

    def forward(self, x, timesteps, mod=None,  sqrt_alpha_hat = None):
        """
        x: B, T, D
        """
        B, T = x.shape[0], x.shape[1]
        # print("input", x.shape)

        # import pdb
        # pdb.set_trace()

        if self.t_embedding_type == "alpha":
            t_shape = list(sqrt_alpha_hat.shape)
            t_shape.append(self.latent_dim)
            sqrt_alpha_hat = timestep_embedding(sqrt_alpha_hat.view(-1)*1000, self.latent_dim).view(t_shape)
            emb = self.alpha_embed(sqrt_alpha_hat)
            (b, l, d) = emb.shape
            emb = rearrange(emb, 'b l d_in -> b d_in l')
            emb = self.alpha_conv(emb)
            emb = emb[:, :, :l]
            emb = rearrange(emb, 'b d_in l -> b l d_in')
        elif self.t_embedding_type == "t":
            emb = self.time_embed(timestep_embedding(timesteps, self.latent_dim))

        if mod is not None:
            # print("mod", mod.shape)
            mod_proj = self.cond_embed(mod)#.reshape(B, -1)
            emb = emb + mod_proj

        
        

        # B, T, latent_dim
        h = self.joint_embed(x)
        h = h + self.sequence_embedding.unsqueeze(0)[:, :T, :]

        # print("h", h.shape)
        
        i = 0
        prelist = []
        for module in self.temporal_decoder_blocks:
            if i < (self.num_layers // 2):
                prelist.append(h)
                h = module(h, emb)
                # print("h in", i, h.shape)
            elif i >= (self.num_layers // 2):
                h = module(h, emb)
                h += prelist[-1]
                prelist.pop()
                # print("h out", i, h.shape)
            i += 1

        output = self.out(h)
        # print("output", output.shape)
        output = output.view(B, T, -1).contiguous()
        return output
