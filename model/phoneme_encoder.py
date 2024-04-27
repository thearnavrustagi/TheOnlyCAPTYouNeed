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
    P_ENCODER_GRU_HIDDEN_SIZE,
    P_MAX_LEN,
    P_NUM_FEATURES,
    P_GRU_INPUT_SIZE,
    EMBEDDING_DIM,
    NUM_EMBEDDINGS
)


class PhonemeEncoder(torch.nn.Module):
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
        hidden_size=P_ENCODER_GRU_HIDDEN_SIZE,
        p_max_len=P_MAX_LEN,
        p_num_features=P_NUM_FEATURES,
        p_gru_input_size=P_GRU_INPUT_SIZE,
        embedding_dim=EMBEDDING_DIM
        num_embeddings=NUM_EMBEDDINGS
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
            in_channels=p_num_features,
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
        self.batch_norm = torch.nn.LayerNorm((out_channels, p_max_len))

        self.convs = [self.conv1, self.conv2, self.conv3]
        self.gru = nn.GRU(p_max_len, hidden_size=hidden_size)

        self.embedding_table = torch.nn.Embedding(
            num_embeddings = NUM_EMBEDDINGS,
            embedding_dim = embedding_dim,
            padding_idx=None,
            max_norm=None,
            norm_type=2.0,
            scale_grad_by_freq=False,
            sparse=False,
            _weight=None,
            _freeze=False,
            device=None,
            dtype=None,
        )

    def forward(self, x):
        # Perform embedding lookup
        x = self.embedding_table(x)

        # Apply convolutional layers
        for conv_layer in self.convs:
            x = conv_layer(x)
            x = self.dropout(x)
            x = self.batch_norm(x)
            x = self.activation(x)

        # Apply GRU
        x, _ = self.gru(x)
        x = self.dropout(x)

        return x