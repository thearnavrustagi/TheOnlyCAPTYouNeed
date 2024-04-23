import torch
import torch.nn as nn
from .hyperparameters import (
    IN_CHANNELS,
    OUT_CHANNELS,
    KERNEL_SIZE,
    PADDING_TYPE,
    STRIDE,
    ACTIVATION,
    DROPOUT_CNN,
    BATCH_NORM,
    MS_ENCODER_GRU_HIDDEN_SIZE,
    MS_MAX_LEN,
    MS_NUM_FEATURES,
    MS_GRU_INPUT_SIZE
)


class MelSpectrogramEncoder(torch.nn.Module):
    def __init__(
        self,
        n_convs=3,
        *,
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        kernel_size=KERNEL_SIZE,
        padding=PADDING_TYPE,
        stride=STRIDE,
        activation=ACTIVATION,
        dropout=DROPOUT_CNN,
        batch_norm=BATCH_NORM,
        hidden_size=MS_ENCODER_GRU_HIDDEN_SIZE,

        ms_max_len=MS_MAX_LEN,
        ms_num_features=MS_NUM_FEATURES,
        ms_gru_input_size=MS_GRU_INPUT_SIZE,
    ):
        super().__init__()
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

        self.conv1 = nn.Conv1d(
            in_channels=ms_num_features,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
        )
        
        self.conv2 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
        )
        
        self.conv3 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
        )
        
        self.dropout = torch.nn.Dropout(p=dropout)
        self.batch_norm = torch.nn.LayerNorm((out_channels, ms_max_len))
        
        self.convs = [self.conv1, self.conv2, self.conv3]
        self.gru = nn.GRU(ms_max_len, hidden_size=hidden_size)

    def forward(self, x):
        for conv_layer in self.convs:
            x = conv_layer(x)
            x = self.dropout(x)
            x = self.batch_norm(x)
            x = self.activation(x)
        (x, _) = self.gru(x)
        x = self.dropout(x)
        return x
