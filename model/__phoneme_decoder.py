import torch
import torch.nn as nn


# location sensitive attention model based on attention-based speech recognition as referenced in the paper:
class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain="linear"):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        # torch.nn.init.xavier_uniform_(self.linear_layer.weight,
        #     gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)


class ConvNorm(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=None,
        dilation=1,
        bias=True,
        w_init_gain="linear",
    ):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert kernel_size % 2 == 1
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

        # torch.nn.init.xavier_uniform_(self.conv.weight,
        #     gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        return self.conv(signal)


class LocationLayer(nn.Module):
    def __init__(self, attention_n_filters, attention_kernel_size, attention_dim):
        super(LocationLayer, self).__init__()
        self.location_conv = ConvNorm(
            1,
            attention_n_filters,
            kernel_size=attention_kernel_size,
            padding=int((attention_kernel_size - 1) / 2),
            stride=1,
            dilation=1,
        )
        self.location_dense = LinearNorm(
            attention_n_filters, attention_dim, bias=False, w_init_gain="tanh"
        )

    def forward(self, attention_weights_cum):
        processed_attention_weights = self.location_conv(attention_weights_cum)
        processed_attention_weights = processed_attention_weights.transpose(1, 2)
        processed_attention_weights = self.location_dense(processed_attention_weights)
        return processed_attention_weights


class Attention(nn.Module):
    def __init__(
        self,
        query_dim,
        memory_dim,
        attention_dim,
        attention_location_n_filters,
        attention_location_kernel_size,
    ):
        super(Attention, self).__init__()
        self.query_layer = LinearNorm(query_dim, attention_dim, w_init_gain="tanh")
        self.memory_layer = LinearNorm(memory_dim, attention_dim, w_init_gain="tanh")
        self.v = LinearNorm(attention_dim, 1)
        self.location_layer = LocationLayer(
            attention_location_n_filters, attention_location_kernel_size, attention_dim
        )
        self.score_mask_value = -float("inf")

    def get_alignment_energies(self, query, memory, attention_weights_cum):
        """
        PARAMS
        ------
        query: decoder output (B, decoder_dim)
        memory: encoder outputs (B, T_in, embed_dim)
        attention_weights_cum: cumulative attention weights (B, 1, max_time)

        RETURNS
        -------
        alignment (batch, max_time)
        """

        # [B, T_in, attn_dim]
        key = self.memory_layer(memory)
        # [B, 1, attn_dim]
        query = self.query_layer(query.unsqueeze(1))
        # [B, T_in, attn_dim]
        location_sensitive_weights = self.location_layer(attention_weights_cum)
        # score function
        energies = self.v(torch.tanh(query + location_sensitive_weights + key))
        # [B, T_in]
        energies = energies.squeeze(-1)

        return energies

    def forward(self, query, memory, attention_weights_cum, mask=None):
        """
        PARAMS
        ------
        query: attention rnn last output [B, decoder_dim]
        memory: encoder outputs [B, T_in, embed_dim]
        attention_weights_cum: cummulative attention weights
        mask: binary mask for padded data
        """
        alignment = self.get_alignment_energies(query, memory, attention_weights_cum)

        if mask is not None:
            alignment.masked_fill_(mask, self.score_mask_value)

        # [B, T_in]
        attention_weights = F.softmax(alignment, dim=1)
        # [B, 1, T_in] * [B, T_in, embbed_dim]
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        # [B, embbed_dim]
        attention_context = attention_context.squeeze(1)

        return attention_context, attention_weights


# phoneme decoder model
class PhonemeDecoder(nn.Module):
    def __init__(
        self,
        *,
        query_dim=ATTN_QUERY_DIM,
        memory_dim=ATTN_MEMORY_DIM,
        attention_dim=ATTN_DIM,
        attention_location_n_filters=ATTN_FILTERS,
        attention_location_kernel_size=ATTN_KERNEL_SIZE,
        query,
        memory,
        attention_weights_cum,
        hidden_size,
        dropout,
    ):
        ## params
        # params for linearnorm and convnorm
        self.query_dim = (query_dim,)
        self.memory_dim = (memory_dim,)
        self.attention_dim = (attention_dim,)

        # params for location layer
        self.attention_location_n_filters = (attention_location_n_filters,)
        self.attention_location_kernel_size = (attention_location_kernel_size,)

        # params for attention
        self.query, self.memory, self.attention_weights_cum = (
            query,
            memory,
            attention_weights_cum,
        )

        # params for gru
        self.hidden_size = (hidden_size,)
        self.dropout = (dropout,)

        ## layers
        self.attention = Attention(
            query_dim,
            memory_dim,
            attention_dim,
            attention_location_n_filters,
            attention_location_kernel_size,
        )
        self.gru = nn.GRU(hidden_size=hidden_size, dropout=dropout)

    def forward(self, query, memory, attention_weights_cum, mask=None):
        attention_context, attention_weights = self.attention(
            query, memory, attention_weights_cum, mask
        )
        output, hidden = self.gru(attention_context)
        return output, hidden
