import math
from typing import Any, Optional

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.modules import (BatchNorm1d, Dropout, Linear, MultiheadAttention,
                              TransformerEncoderLayer)


class TimeSeriesImputerCNN(nn.Module):
    def __init__(
        self,
        feat_dim,
        max_len,
        activation_fn="relu",
        kernel_sizes=[3, 5],
        num_filters=[16, 32],
        dropout=0.2,
    ):
        super(TimeSeriesImputerCNN, self).__init__()
        self.feat_dim = feat_dim
        self.max_len = max_len
        self.conv_layers = nn.ModuleList([])
        self.pool_layers = nn.ModuleList([])
        i = 0
        # Define multiple convolutional layers with different kernel sizes
        for kernel_size, num_filter in zip(kernel_sizes, num_filters):
            self.conv_layers.append(
                nn.Conv1d(
                    in_channels=feat_dim,
                    out_channels=num_filter,
                    kernel_size=kernel_size,
                    padding=1,  # Changed padding mode
                )
            )
            self.pool_layers.append(nn.MaxPool1d(kernel_size=kernel_size, stride=1, padding=1))  # Removed ceil_mode (optional)
            i = i + 1

        # Flatten the output of convolutional layers (similar to transformer flattening)
        self.flatten = nn.Flatten()

        # Final output layer to match input dimensions (similar to transformer encoder output)
        self.output_layer = nn.Linear(
            self.compute_output_dim(kernel_sizes, num_filters), feat_dim
        )

        self.act = _get_activation_fn(activation_fn)
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        self.kernel_sizes = kernel_sizes

    def compute_output_dim(self, kernel_sizes, num_filters):
        # Calculate the output dimension based on kernel sizes and number of filters
        output_dim = 0
        for kernel_size, num_filter in zip(kernel_sizes, num_filters):
            # Assuming stride = 1 for all convolutions
            output_dim += (self.max_len - (kernel_size - 1)) * num_filter
        return output_dim

    def forward(self, X, mask):
        """
        Args:
            X: (batch_size, seq_length, feat_dim) torch tensor of features (potentially containing missing values)
            mask: (batch_size, seq_length) boolean tensor where 1 indicates valid data and 0 indicates missing values
        Returns:
            output: (batch_size, seq_length, feat_dim) torch tensor with imputed values
        """

        # Apply masking to data before feeding into convolutions (set masked values to 0)
        X = X.permute(0, 2, 1)

        pad_len = 5  # Calculate padding length based on largest kernel size
        X = nn.functional.pad(X, (pad_len, pad_len))
        print(f"Shape after padding: {X.size()}")


        # Pass through convolutional layers
        skip_connections = []
        for conv, pool in zip(self.conv_layers, self.pool_layers):
            out = conv(X)    
            print(f"Output shape after conv layer: {out.size()}")
            out = self.act(out)  # ReLU activation
            out = pool(out)
            print(f"Output shape after pool layer: {out.size()}")
            skip_connections.append(out)

        # Concatenate skip connections for preserving local features
        conv_out = torch.cat(skip_connections, dim=1)

        # Flatten the output (similar to transformer)
        flat_out = self.flatten(conv_out)

        # Dropout for regularization
        out = self.dropout(flat_out)

        # Final output layer
        output = self.output_layer(out)

        # Reshape to match input dimensions
        output = output.view(X.size())

        # Mask the imputed values back into the original data
        output = output * mask.unsqueeze(2)

        return output, None  # Since we have no embedding to return
