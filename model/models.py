import torch
from torch import nn

from .hyperparameters import (
    MDN_CLF_IN_DIM,
    MDN_CLF_OUT_DIM,
    PRN_CLF_IN_DIM,
    PRN_CLF_OUT_DIM,
)


class MDNClassificationHead(torch.nn.Module):
    def __init__(self, *, in_dim=MDN_CLF_IN_DIM, out_dim=MDN_CLF_OUT_DIM):
        super(MDNClassificationHead, self).__init__()
        self.clf = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.clf(x)


class PRNClassificationHead(torch.nn.Module):
    def __init__(self, *, in_dim=PRN_CLF_IN_DIM, out_dim=PRN_CLF_OUT_DIM):
        super(PRNClassificationHead, self).__init__()
        self.clf = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.clf(x)


class MispronunciationDetectionNetwork(torch.nn.Module):
    def __init__(
        self,
        *,
        phoneme_encoder: torch.nn.Module,
        phoneme_decoder: torch.nn.Module,
        word_decoder: torch.nn.Module,
        classification_head: torch.nn.Module = MDNClassificationHead()
    ):
        super(MispronunciationDetectionNetwork, self).__init__()

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
        mel_spectrogram_encoder,
        phoneme_decoder,
        classification_head: torch.nn.Module = PRNClassificationHead()
    ):
        super(PhonemeRecognitionNetwork, self).__init__()

        self.mel_spectrogram_encoder = mel_spectrogram_encoder
        self.phoneme_decoder = phoneme_decoder
        self.classification_head = classification_head

    def forward(self, i):
        x = self.mel_spectrogram_encoder(i)
        x = self.phoneme_decoder(x)
        x = self.classification_head(x)

        return x
