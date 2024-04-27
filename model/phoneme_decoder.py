from torch import nn
from .hyperparameters import (
    P_DECODER_ATTN_EMBED_DIM,
    P_DECODER_ATTN_N_HEADS,
    P_DECODER_GRU_HIDDEN_SIZE,
    P_DECODER_GRU_INPUT_SIZE,
    P_DECODER_DROPOUT,
)


class PhonemeDecoder(nn.Module):
    def __init__(
        self,
        *,
        embed_dim=P_DECODER_ATTN_EMBED_DIM,
        num_heads=P_DECODER_ATTN_N_HEADS,
        gru_input_size=P_DECODER_GRU_INPUT_SIZE,
        gru_hidden_size=P_DECODER_GRU_HIDDEN_SIZE,
        dropout=P_DECODER_DROPOUT,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.gru_hidden_size = gru_hidden_size
        self.gru_input_size = gru_input_size
        self.dropout = dropout

        self.self_attn = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
        )

        self.gru = nn.GRU(
            self.gru_input_size, hidden_size=self.gru_hidden_size
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        attn_output, _ = self.self_attn(x, x, x)
        x = x + attn_output
        x, _ = self.gru(x)
        y = self.dropout(x)

        return y
