from .phoneme_decoder import PhonemeDecoder
from .hyperparameters import W_DECODER_ATTN_EMBED_DIM, W_DECODER_GRU_INPUT_SIZE


def WordDecoder(
    decoder_embed_dim=W_DECODER_ATTN_EMBED_DIM, gru_input_size=W_DECODER_GRU_INPUT_SIZE
):
    return PhonemeDecoder(embed_dim=decoder_embed_dim, gru_input_size=gru_input_size)
