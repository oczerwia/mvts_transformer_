import math
from typing import Any, Optional

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.modules import (BatchNorm1d, Dropout, Linear, MultiheadAttention,
                              TransformerEncoderLayer)

class SelfSupervisedLSTMImputer(nn.Module):
    def __init__(
        self,
        feat_dim,
        max_len,
        num_layers=1,
        hidden_size=128,
        dropout=0.1,
        forget_gate_bias=1.0,
        activation="tanh",
    ):
        super(SelfSupervisedLSTMImputer, self).__init__()
        self.feat_dim = feat_dim
        self.max_len = max_len
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.forget_gate_bias = forget_gate_bias
        self.activation_function = activation
        self.lstm = nn.LSTM(
            feat_dim,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=True,
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lstm = self.lstm.to(self.device)

        self.fc = nn.Linear(
            hidden_size * 2, feat_dim
        )  # Reconstruct using both directions
        self.dropout = nn.Dropout(dropout)

        # Activation function
        self.act = _get_activation_fn
        if activation == "tanh":
            self.act = nn.Tanh()
        elif activation == "relu":
            self.act = nn.ReLU()
        elif activation == "gelu":
            self.act = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

    def forward(self, X, mask):  # Assuming mask is not used here
        # Permute for LSTM input format (seq_len, batch_size, feature_dim)
        X = X.permute(1, 0, 2)

        # Set forget gate bias (if custom value provided)
        if self.forget_gate_bias != 1.0:
            self.lstm.bias_ih_l0 = torch.nn.Parameter(
                torch.tensor(
                    self.forget_gate_bias * np.ones(self.lstm.bias_ih_l0.size()),
                    dtype=torch.float,
                    device=self.device,
                )
            )
            self.lstm.bias_hh_l0 = torch.nn.Parameter(
                torch.tensor(
                    self.forget_gate_bias * np.ones(self.lstm.bias_hh_l0.size()),
                    dtype=torch.float,
                    device=self.device,
                )
            )

        # Pass through LSTM
        X = X.to(self.device)
        output, (hidden, cell) = self.lstm(X)

        # Apply activation function
        output = self.act(output)

        # Concatenate outputs from both directions
        output = torch.cat(
            (output[:, :, : self.hidden_size], output[:, :, self.hidden_size :]), dim=2
        )

        # Reconstruct the original features with dropout
        reconstructed_X = self.fc(self.dropout(output))
        reconstructed_X = reconstructed_X.permute(
            1, 0, 2
        )  # Back to (batch_size, seq_len, feature_dim)

        # Mimic Transformer output format: return reconstructed features and hidden state (embedding)
        embedding = output.permute(1, 0, 2)  # (batch_size, seq_length, hidden_size * 2)

        return reconstructed_X, embedding
