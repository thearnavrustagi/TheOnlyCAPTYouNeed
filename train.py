from dataloaders import train_dataloader
from model import MelSpectrogramEncoder, PhonemeEncoder, PhonemeDecoder, WordDecoder


if __name__ == "__main__":
    ms, e_p, tokens = next(iter(train_dataloader))
    print(ms.shape)
    print("MelSpectrogram Encoder")
    layer = MelSpectrogramEncoder()
    mse_out = layer(ms)
    print(mse_out.shape)

    print("Phoneme Encoder")
    layer = PhonemeEncoder()
    pe_out = layer(tokens)
    print(pe_out.shape)

    print("Phoneme Decoder")
    layer = PhonemeDecoder()
    pd_out = layer(pe_out)
    print(pd_out.shape)

    print("Word Decoder")
    layer = WordDecoder()
    wd_out = layer(pd_out)
    print(wd_out.shape)
