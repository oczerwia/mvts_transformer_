"""Everything needed for the cross former."""

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