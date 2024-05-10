import torch
from torch import nn
from .hyperparameters import (
    P_DECODER_ATTN_EMBED_DIM,
    P_DECODER_ATTN_N_HEADS,
    P_DECODER_GRU_HIDDEN_SIZE,
    P_DECODER_GRU_INPUT_SIZE,
    P_DECODER_DROPOUT,
    P_DECODER_GRU_N_LAYERS,
    MS_MAX_LEN
)


class PhonemeDecoder(nn.Module):
    def __init__(
        self,
        *,
        embed_dim=P_DECODER_ATTN_EMBED_DIM,
        num_heads=P_DECODER_ATTN_N_HEADS,
        gru_input_size=P_DECODER_GRU_INPUT_SIZE,
        gru_hidden_size=P_DECODER_GRU_HIDDEN_SIZE,
        gru_n_layers=P_DECODER_GRU_N_LAYERS,
        dropout=P_DECODER_DROPOUT,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.gru_hidden_size = gru_hidden_size
        self.gru_input_size = gru_input_size
        self.gru_n_layers = gru_n_layers
        self.dropout = dropout

        self.self_attn = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
        )

        self.gru = nn.GRU(
            self.gru_input_size,
            hidden_size=self.gru_hidden_size,
            num_layers=self.gru_n_layers,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        mask = torch.squeeze(torch.sum((x != 0).float(), axis=-2))
        mask = torch.sum((mask != 0).float(), axis=-1)
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        mask = mask.to("cpu")

        attn_output, _ = self.self_attn(x, x, x)
        x = x + attn_output

        x = torch.nn.utils.rnn.pack_padded_sequence(x, mask, batch_first=True, enforce_sorted=False)
        # Apply GRU
        x, _ = self.gru(x)
        x, mask = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True, total_length=MS_MAX_LEN)

        x = self.dropout(x)

        return x
