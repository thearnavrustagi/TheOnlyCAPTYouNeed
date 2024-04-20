import torch


class MispronunciationDetectionNetwork(torch.nn.Module):
    def __init__(
        self,
        *,
        phoneme_vocabulary: int,
        phoneme_encoder: torch.nn.Module,
        phoneme_decoder: torch.nn.Module,
        word_decoder: torch.nn.Module,
        classification_head: torch.nn.Module
    ):
        self.phoneme_vocabulary = phoneme_vocabulary
        self.phoneme_encoder = phoneme_encoder
        self.phoneme_decoder = phoneme_decoder
        self.word_decoder = word_decoder
        self.classification_head = classification_head

    def forward(self, i):
        x = self.phoneme_encoder(i)
        x = self.phoneme_decoder(x)
        x = self.word_decoder(x)
        x = self.classification_head(x)

        return x


class PhonemeRecognitionNetwork(torch.nn.Module):
    def __init__(
        self,
        *,
        phoneme_vocabulary,
        mel_spectrogram_encoder,
        phoneme_decoder,
        classification_head
    ):
        self.phoneme_vocabulary = phoneme_vocabulary
        self.mel_spectrogram_encoder = mel_spectrogram_encoder
        self.phoneme_decoder = phoneme_decoder
        self.classification_head = classification_head

    def forward(self, i):
        x = self.mel_spectrogram_encoder(i)
        x = self.phoneme_decoder(x)
        x = self.classification_head(x)

        return x
