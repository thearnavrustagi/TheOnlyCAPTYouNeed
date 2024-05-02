import torch
import random
import numpy as np
from .utils import train_one_epoch
from .models import MispronunciationDetectionNetwork, PhonemeRecognitionNetwork
from .models import MDNClassificationHead, PRNClassificationHead
from .hyperparameters import N_EPOCHS, PRN_CLF_OUT_DIM, LR, SEED
from .melspectogram_encoder import MelSpectrogramEncoder
from .phoneme_encoder import PhonemeEncoder
from .phoneme_decoder import PhonemeDecoder
from .word_decoder import WordDecoder

torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

class Dhvani(torch.nn.Module):
    def __init__(
        self,
        *,
        phoneme_encoder: torch.nn.Module = PhonemeEncoder(),
        mel_spectrogram_encoder: torch.nn.Module = MelSpectrogramEncoder(),
        phoneme_decoder: torch.nn.Module = PhonemeDecoder(),
        word_decoder: torch.nn.Module = WordDecoder(),
        mdn_classification_head: torch.nn.Module = MDNClassificationHead(),
        prn_classification_head: torch.nn.Module = PRNClassificationHead(),
    ):
        super(Dhvani, self).__init__()

        self.phoneme_recognition_network = PhonemeRecognitionNetwork(
            mel_spectrogram_encoder=mel_spectrogram_encoder,
            phoneme_decoder=phoneme_decoder,
            classification_head=prn_classification_head,
        )
        self.mispronunciation_detection_network = MispronunciationDetectionNetwork(
            phoneme_encoder=phoneme_encoder,
            phoneme_decoder=phoneme_decoder,
            word_decoder=word_decoder,
            classification_head=mdn_classification_head,
        )

    def forward(self, ms, tokens):
        prn_out = self.phoneme_recognition_network(ms)
        mdn_out = self.mispronunciation_detection_network(tokens)

        return prn_out, mdn_out

    def train(self, dataloaders, *, epochs=N_EPOCHS):
        prn_optimizer = torch.optim.Adam(self.phoneme_recognition_network.parameters(), lr=LR)
        mdn_optimizer = torch.optim.Adam(
            self.mispronunciation_detection_network.parameters(), lr=LR
        )

        train_dataloader, validation_dataloader, test_dataloader = dataloaders

        prn_loss_fn = torch.nn.CrossEntropyLoss()
        mdn_loss_fn = torch.nn.BCELoss()

        for epoch_number in range(epochs):
            print(f"Training EPOCH:{epoch_number}")
            train_one_epoch(
                self.phoneme_recognition_network,
                train_dataloader,
                prn_loss_fn,
                xidx=0,
                yidx=2,
                classes=PRN_CLF_OUT_DIM,
                epoch_number=epoch_number,
                optimizer=prn_optimizer,
            )
            """
            train_one_epoch(
                self.mispronunciation_detection_network,
                train_dataloader,
                mdn_optimizer,
                mdn_loss_fn,
                xidx=2,
                yidx=1,
            )
            """
            print(f"Running Validation")
            train_one_epoch(
                self.phoneme_recognition_network,
                validation_dataloader,
                prn_loss_fn,
                xidx=0,
                yidx=2,
                epoch_number=epoch_number,
                classes=PRN_CLF_OUT_DIM,
                train=False
            )
            """
            validate_model(
                self.mispronunciation_detection_network,
                validation_dataloader,
                mdn_loss_fn,
                xidx=2,
                yidx=1,
            )
            """


if __name__ == "__main__":
    pass
