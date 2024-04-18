import torch
from utils import train_one_epoch


class Dhvani(torch.nn.Module):
    def __init__(
        self,
        *,
        phoneme_encoder:torch.nn.Module,
        mel_spectrogram_encoder:torch.nn.Module,
        phoneme_decoder:torch.nn.Module,
        word_decoder:torch.nn.Module,
        phoneme_vocabulary:int,
        mdn_classification_head: torch.nn.Module,
        prn_classification_head: torch.nn.Module
    ):
        super(Dhvani, self).__init__()

        self.phoneme_recognition_network = PhonemeRecognitionNetwork(
            phoneme_vocabulary=phoneme_vocabulary,
            mel_spectrogram_encoder=mel_spectrogram_encoder,
            phoneme_decoder=phoneme_decoder,
            classification_head=prn_classification_head
        )
        self.mispronunciation_detection_network = MispronunciationDetectionNetwork(
            phoneme_encoder=phoneme_encoder,
            phoneme_decoder=phoneme_decoder,
            word_decoder=word_decoder,
            classification_head=mdn_classification_head
        )

    def train(self, *, dataloader):
        prn_optimizer = torch.optim.Adam(self.phoneme_recognition_network.parameters())
        mdn_optimizer = torch.optim.Adam(self.mispronunciation_detection_network.parameters())A

        loss_fn = torch.nn.CrossEntropyLoss()

        train_one_epoch(self.phoneme_recognition_network, dataloader, prn_optimizer, loss_fn, xidx=0, yidx=0)
        train_one_epoch(self.mispronunciation_detection_network, dataloader, mdn_optimizer, loss_fn, xidx=2, yidx=1)


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
        


