import torch
import torch.nn as nn


class melspectogram_encoder:
    def __init__(
        self,
        n_convs=3,
        *,
        in_channels,
        out_channels,
        kernel_size,
        padding,
        stride,
        activation,
        dropout,
        batch_norm,
        hidden_size
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.activation = activation
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.hidden_size = hidden_size
        self.n_convs = n_convs

        self.convs = [ nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            activation=activation,
            dropout=dropout,
            batch_norm=batch_norm,
        ) for  _ in range(n_convs) ]
        
        self.gru = nn.GRU(hidden_size=hidden_size, dropout=dropout)

    def forward(self, x):
        for conv_layer in self.convs:
            x = conv_layer(x)
        x = self.gru(x)
        return x
