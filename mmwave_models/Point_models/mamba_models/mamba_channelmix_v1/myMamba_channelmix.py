import torch
from .mamba_simple import *
from models.MotionTransformer import *
from .ChebConv import ChebConv, _GraphConv, _ResChebGC, adj_mx_from_edges


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


class Mamba(nn.Module):
    def __init__(self, args: ModelArgs):
        """My Mamba model."""
        super().__init__()
        self.args = args
        print(args)

        self.joint_num = self.args.vocab_size//3
        self.d_joint = self.args.d_model//self.joint_num
        self.d_model = self.d_joint * self.joint_num
        self.args.d_model = self.d_model
        
        # self.embedding = nn.Linear(3, self.d_joint)
        self.embedding = ChebConv(in_c=3, out_c=self.d_joint, K=2)
        if self.args.cond_length == self.args.vocab_length:
            self.cond_embed = nn.Linear(self.args.vocab_size , self.args.temp_emb_dim) # * self.args.cond_length
        else:
            self.cond_embed = nn.Linear(self.args.vocab_size * self.args.cond_length, self.args.temp_emb_dim)
        self.time_embed = nn.Sequential(
            nn.Linear(self.d_model, self.args.temp_emb_dim),
            nn.SiLU(),
            nn.Linear(self.args.temp_emb_dim, self.args.temp_emb_dim),
        )
        self.sequence_embedding = nn.Parameter(torch.randn(self.args.vocab_length, self.d_model))


        self.layers_M = nn.ModuleList([ResidualBlock(args) for _ in range(args.n_layer)])


        # channel attention implementation
        self.layer_channel_T = nn.ModuleList()
        for _ in range(args.n_layer):
            self.layer_channel_T.append(
                TChannelWiseSelfAttention(
                    latent_dim=self.d_joint,
                    num_head=1,
                    dropout=0.2,
                )
            )


        # Graconv implementation
        self.layer_channel_G = nn.ModuleList()
        self.edges = torch.tensor(self.args.adj, dtype=torch.long)
        self.adj = adj_mx_from_edges(num_pts=self.joint_num, edges=self.edges, sparse=False).cuda()

        for _ in range(args.n_layer):
            # self.layer_channel_G.append(ChebConv(in_c=self.d_joint, out_c=self.d_joint, K=2))
            self.layer_channel_G.append(_GraphConv(self.d_joint, self.d_joint, self.args.dropout))

        



        self.layer_T = nn.ModuleList()

        for i in range(self.args.num_T_in_M_layers):
            self.layer_T.append(
                TemporalDiffusionTransformerDecoderLayer(
                    latent_dim=self.d_model,
                    time_embed_dim=self.args.temp_emb_dim,
                    ffn_dim=2*self.d_model,
                    num_head=self.args.num_T_in_M_head,
                    dropout=self.args.dropout,
                )
            )

        self.norm_f = RMSNorm(args.d_model)

        self.lm_head = nn.Linear(self.d_joint, 3)
        


    def forward(self, x, timesteps, mod=None):
        """
        Args:
            input_ids (long tensor): shape (b, l)    (See Glossary at top for definitions of b, l, d_in, n...)
    
        Returns:
            logits: shape (b, l, vocab_size)

        Official Implementation:
            class MambaLMHeadModel, https://github.com/state-spaces/mamba/blob/main/mamba_ssm/models/mixer_seq_simple.py#L173

        """

        B, T = x.shape[0], x.shape[1]
        # print("input", x.shape)

        emb = self.time_embed(timestep_embedding(timesteps, self.d_model))
        # print("emb", emb.shape)

        if mod is not None:
            # print("mod", mod.shape)
            mod_proj = self.cond_embed(mod)#.reshape(B, -1)
            # print("mod_proj", mod_proj.shape)
            if len(mod_proj.shape) == 3:
                emb = emb.unsqueeze(dim=1) + mod_proj
            else:
                emb = emb + mod_proj
        h = x
        h = h.reshape(B*T, self.joint_num, 3)
        # print("h reshape for emb: ", h.shape)
        h = self.embedding(h, self.adj)
        h = h.reshape(B, T, self.joint_num*self.d_joint)
        # h = h + self.sequence_embedding.unsqueeze(0)[:, :T, :] # whether this is necessary
        # print("h: ", h.shape)



        i = 0
        prelist = []
        for layer in self.layers_M:
            if i < (self.args.n_layer // 2):

                prelist.append(h)

                #for channel attention
                h = h.view(B, T, self.joint_num, self.d_joint).reshape(B*T, self.joint_num, self.d_joint)
                # print("h for channel attention", h.shape)
                h = h + self.layer_channel_G[i](h, self.adj)
                h = h.view(B, T, self.joint_num, self.d_joint).reshape(B, T, self.joint_num * self.d_joint)
                # print("h after channel attention", h.shape)


                
                h = layer(h, emb)
                # print("h in", i, h.shape)

                

            elif i >= (self.args.n_layer // 2):

                #for channel attention
                h = h.view(B, T, self.joint_num, self.d_joint).reshape(B*T, self.joint_num, self.d_joint)
                # print("h for channel attention", h.shape)
                h = h + self.layer_channel_G[i](h, self.adj)
                h = h.view(B, T, self.joint_num, self.d_joint).reshape(B, T, self.joint_num * self.d_joint)
                # print("h after channel attention", h.shape)


                h = layer(h, emb)
                h += prelist[-1]
                prelist.pop()

                

                # print("h out", i, h.shape)
            i += 1

        
        for trans_module in self.layer_T:
            attn_h = trans_module(h, emb)
            h = attn_h + h


            
        h = self.norm_f(h)
        # print("h: ",h.shape)

        h = h.view(B, T, self.joint_num, self.d_joint)
        # print("h reshape for emb: ", h.shape)
        logits = self.lm_head(h)
        logits = logits.view(B, T, self.joint_num*3)
        

        
        # print("logits: ",logits.shape)

        return logits
    




class TChannelWiseSelfAttention(nn.Module):

    def __init__(self, latent_dim, num_head, dropout):
        super().__init__()
        self.num_head = num_head
        self.norm = nn.LayerNorm(latent_dim)
        self.query = nn.Linear(latent_dim, latent_dim, bias=False)
        self.key = nn.Linear(latent_dim, latent_dim, bias=False)
        self.value = nn.Linear(latent_dim, latent_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
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
        weight = self.dropout(F.softmax(attention, dim=2))
        value = self.value(self.norm(x)).view(B, T, H, -1)
        y = torch.einsum('bnmh,bmhd->bnhd', weight, value).reshape(B, T, D)
        y = x + y
        return y