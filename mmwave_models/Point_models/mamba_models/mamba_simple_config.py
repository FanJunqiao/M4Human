"""Simple, minimal implementation of Mamba in one file of PyTorch.

Suggest reading the following before/while reading the code:
    [1] Mamba: Linear-Time Sequence Modeling with Selective State Spaces (Albert Gu and Tri Dao)
        https://arxiv.org/abs/2312.00752
    [2] The Annotated S4 (Sasha Rush and Sidd Karamcheti)
        https://srush.github.io/annotated-s4

Glossary:
    b: batch size                       (`B` in Mamba paper [1] Algorithm 2)
    l: sequence length                  (`L` in [1] Algorithm 2)
    d or d_model: hidden dim
    n or d_state: latent state dim      (`N` in [1] Algorithm 2)
    expand: expansion factor            (`E` in [1] Section 3.4)
    d_in or d_inner: d * expand         (`D` in [1] Algorithm 2)
    A, B, C, D: state space parameters  (See any state space representation formula)
                                        (B, C are input-dependent (aka selective, a key innovation in Mamba); A, D are not)
    Δ or delta: input-dependent step size
    dt_rank: rank of Δ                  (See [1] Section 3.6 "Parameterization of ∆")

"""
from __future__ import annotations
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass



def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

@dataclass
class ModelArgs:
    d_model: int
    n_layer: int
    num_T_in_M_layers: int
    num_T_in_M_head: int

    ## extra param for model design
    vocab_length: int
    vocab_size: int
    cond_length: int
    temp_emb_dim: int
    
    dropout: int = 0.0
    d_state: int = 16
    expand: int = 2
    dt_rank: Union[int, str] = 'auto'
    d_conv: int = 4 
    pad_vocab_size_multiple: int = 8
    conv_bias: bool = True
    bias: bool = False

    alpha_kalman: int = 5
    alpha_kalman_enable: bool = False
    cfg: object = None



    ## extra param for mamba robust loss
    return_loss: bool = True

    ## extra param for mamba parallel
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random" # "random" or "constant"
    dt_scale: float = 1.0
    dt_init_floor = 1e-4

    rms_norm_eps: float = 1e-5
    base_std: float = 0.02

    inner_layernorms: bool = False # apply layernorms to internal activations
    mup: bool = False
    mup_base_width: float = 128 # width=d_model
    pscan: bool = True # use parallel scan mode or sequential mode when training
    use_cuda: bool = False # use official CUDA implementation when training (not compatible with (b)float16)
    
    def __post_init__(self):
        self.d_inner = int(self.expand * self.d_model)
        
        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)
            
        if self.vocab_size % self.pad_vocab_size_multiple != 0:
            self.vocab_size += (self.pad_vocab_size_multiple
                                - self.vocab_size % self.pad_vocab_size_multiple)

        ## extra param for GCN
        self.adj = [[0, 1], [1, 2],
                    [3, 4], [4, 5],
                    [6, 7], [7, 8], [8,9],
                    [7, 10], [10, 11], [11, 12],
                    [7, 13], [13, 14], [14, 15],
                    [0, 3], [3, 6], [0, 6]]
        # self.adj = [[0,1], [1,2], [2,3],
        #             [0,4], [4,5], [5, 6],
        #             [0,7], [7, 8], [8, 9], [9, 10],
        #             [8,11], [11,12], [12, 13],
        #             [8,14], [14,15], [15,16]]


        ## extra param for inner dct
        self.inner_dct_enable: bool = False
        self.dct_m = self.cfg.dct_m_all
        self.idct_m = self.cfg.idct_m_all
        self.n_pre = self.cfg.n_pre
        self.n_pre_con = self.cfg.n_pre_cond