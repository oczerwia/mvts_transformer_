"""Everything needed for the cross former.


A few considerations:

- We want to take into account only data of fixed length (1700)


"""

############# Imports ###################

import torch
import torch.nn as nn
import torch.nn.functional as F
import math



############# Embeddings ###############

class DSW_embedding(nn.Module):
    def __init__(self, seg_len, d_model):
        super(DSW_embedding, self).__init__()
        self.seg_len = seg_len
        self.d_model = d_model

        self.linear = nn.Linear(seg_len, d_model)

    def forward(self, x):  
        # Reshape and permute dimensions
        segment_length = self.seg_len
        x_segment = torch.reshape(x, (x.shape[0], -1, segment_length, x.shape[2]))
        x_segment = x_segment.permute(0, 2, 1, 3)  # (b, seg_len, seg_num, d)

        # Apply linear layer
        x_embed = self.linear(x_segment)

        # Reshape and permute dimensions (assuming d is the feature dimension)
        x_embed = x_embed.view(x.shape[0], -1, x_segment.shape[2], self.d_model)  # view for flexibility
        x_embed = x_embed.permute(0, 2, 1, 3)  # (b, seg_num, d, d_model)

        return x_embed
    

############## Attention Layer ############


class FullAttention(nn.Module):
    '''
    The Attention operation
    '''
    def __init__(self, scale=None, attention_dropout=0.1):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.dropout = nn.Dropout(attention_dropout)
        
    def forward(self, queries, keys, values):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1./torch.sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        
        return V.contiguous()

class AttentionLayer(nn.Module):
    '''
    The Multi-head Self-Attention (MSA) Layer
    '''
    def __init__(self, d_model, n_heads, d_keys=None, d_values=None, dropout = 0.1):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = FullAttention(scale=None, attention_dropout = dropout)
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out = self.inner_attention(
            queries,
            keys,
            values,
        )

        out = out.view(B, L, -1)

        return self.out_projection(out)


class TwoStageAttentionLayer(nn.Module):
    '''
    The Two Stage Attention (TSA) Layer
    input/output shape: [batch_size, Data_dim(D), Seg_num(L), d_model]
    '''
    def __init__(self, seg_num, factor, d_model, n_heads, d_ff = None, dropout=0.1):
        super(TwoStageAttentionLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.time_attention = AttentionLayer(d_model, n_heads, dropout = dropout)
        self.dim_sender = AttentionLayer(d_model, n_heads, dropout = dropout)
        self.dim_receiver = AttentionLayer(d_model, n_heads, dropout = dropout)
        self.router = nn.Parameter(torch.randn(seg_num, factor, d_model))
        
        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)

        self.MLP1 = nn.Sequential(nn.Linear(d_model, d_ff),
                                nn.GELU(),
                                nn.Linear(d_ff, d_model))
        self.MLP2 = nn.Sequential(nn.Linear(d_model, d_ff),
                                nn.GELU(),
                                nn.Linear(d_ff, d_model))

    def forward(self, x):
        #Cross Time Stage: Directly apply MSA to each dimension
        batch = x.shape[0]
        time_in = x.view(x.shape[0], x.shape[1], -1, x.shape[3])
        time_enc = self.time_attention(
            time_in, time_in, time_in
        )
        dim_in = time_in + self.dropout(time_enc)
        dim_in = self.norm1(dim_in)
        dim_in = dim_in + self.dropout(self.MLP1(dim_in))
        dim_in = self.norm2(dim_in)

        #Cross Dimension Stage: use a small set of learnable vectors to aggregate and distribute messages to build the D-to-D connection
        dim_send = dim_in.view(dim_in.shape[0], -1, dim_in.shape[1], dim_in.shape[3])
        batch_router = self.router.repeat(batch, 1, 1) 
        dim_buffer = self.dim_sender(batch_router, dim_send, dim_send)
        dim_receive = self.dim_receiver(dim_send, dim_buffer, dim_buffer)
        dim_enc = dim_send + self.dropout(dim_receive)
        dim_enc = self.norm3(dim_enc)
        dim_enc = dim_enc + self.dropout(self.MLP2(dim_enc))
        dim_enc = self.norm4(dim_enc)

        final_out = dim_enc.view(batch, -1, dim_enc.shape[2], dim_enc.shape[3])

        return final_out
    


############## Encoder Layer ############

class SegMerging(nn.Module):
    '''
    Segment Merging Layer.
    The adjacent `win_size' segments in each dimension will be merged into one segment to
    get representation of a coarser scale
    we set win_size = 2 in our paper
    '''
    def __init__(self, d_model, win_size, norm_layer=nn.LayerNorm):
        super().__init__()
        self.d_model = d_model
        self.win_size = win_size
        self.linear_trans = nn.Linear(win_size * d_model, d_model)
        self.norm = norm_layer(win_size * d_model)

    def forward(self, x):
        """
        x: B, ts_d, L, d_model
        """
        batch_size, ts_d, seg_num, d_model = x.shape
        pad_num = seg_num % self.win_size
        if pad_num != 0: 
            pad_num = self.win_size - pad_num
            x = torch.cat((x, x[:, :, -pad_num:, :]), dim = -2)

        seg_to_merge = []
        for i in range(self.win_size):
            seg_to_merge.append(x[:, :, i::self.win_size, :])
        x = torch.cat(seg_to_merge, -1)  # [B, ts_d, seg_num/win_size, win_size*d_model]

        x = self.norm(x)
        x = self.linear_trans(x)

        return x

class scale_block(nn.Module):
    '''
    We can use one segment merging layer followed by multiple TSA layers in each scale
    the parameter `depth' determines the number of TSA layers used in each scale
    We set depth = 1 in the paper
    '''
    def __init__(self, win_size, d_model, n_heads, d_ff, depth, dropout, \
                    seg_num = 10, factor=10):
        super(scale_block, self).__init__()

        if (win_size > 1):
            self.merge_layer = SegMerging(d_model, win_size, nn.LayerNorm)
        else:
            self.merge_layer = None
        
        self.encode_layers = nn.ModuleList()

        for i in range(depth):
            self.encode_layers.append(TwoStageAttentionLayer(seg_num, factor, d_model, n_heads, \
                                                        d_ff, dropout))
    
    def forward(self, x):
        _, ts_dim, _, _ = x.shape

        if self.merge_layer is not None:
            x = self.merge_layer(x)
        
        for layer in self.encode_layers:
            x = layer(x)        
        
        return x

class Encoder(nn.Module):
    '''
    The Encoder of Crossformer.
    '''
    def __init__(self, e_blocks, win_size, d_model, n_heads, d_ff, block_depth, dropout,
                in_seg_num = 10, factor=10):
        super(Encoder, self).__init__()
        self.encode_blocks = nn.ModuleList()

        self.encode_blocks.append(scale_block(1, d_model, n_heads, d_ff, block_depth, dropout,\
                                            in_seg_num, factor))
        for i in range(1, e_blocks):
            self.encode_blocks.append(scale_block(win_size, d_model, n_heads, d_ff, block_depth, dropout,\
                                            math.ceil(in_seg_num/win_size**i), factor))

    def forward(self, x):
        encode_x = []
        encode_x.append(x)
        
        for block in self.encode_blocks:
            x = block(x)
            encode_x.append(x)

        return encode_x
    


################ Complete architecture #############


class Crossformer(nn.Module):
    def __init__(self, data_dim, seg_len, win_size = 4,
                factor=10, d_model=512, d_ff = 1024, n_heads=8, e_layers=3, 
                dropout=0.0):
        super(Crossformer, self).__init__()
        self.data_dim = data_dim
        self.seg_len = seg_len
        self.merge_win = win_size


        # Embedding
        self.enc_value_embedding = DSW_embedding(seg_len, d_model)
        self.enc_pos_embedding = nn.Parameter(torch.randn(1, data_dim, seg_len, d_model))
        self.pre_norm = nn.LayerNorm(d_model)

        # Encoder
        self.encoder = Encoder(e_layers, win_size, d_model, n_heads, d_ff, block_depth = 1, \
                                    dropout = dropout,in_seg_num = seg_len, factor = factor)
        

    def forward(self, x_seq):
        batch_size = x_seq.shape[0]
        x_seq = self.enc_value_embedding(x_seq)
        x_seq += self.enc_pos_embedding
        x_seq = self.pre_norm(x_seq)
        
        enc_out = self.encoder(x_seq)
        predict_y = None # TODO: Need to implement whatever comes in here

        predict_y[:, :self.out_len, :]