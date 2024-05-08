import torch
import random
import numpy as np
from .utils import train_one_epoch
from .models import MispronunciationDetectionNetwork, PhonemeRecognitionNetwork
from .models import MDNClassificationHead, PRNClassificationHead
from .hyperparameters import N_EPOCHS, PRN_CLF_OUT_DIM, LR, SEED, MDN_CLF_OUT_DIM
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

    def train(
        self,
        dataloaders,
        *,
        epochs=N_EPOCHS,
        fold_no=None,
        training_prn=True,
        training_mdn=True,
    ):
        prn_optimizer = torch.optim.Adam(
            self.phoneme_recognition_network.parameters(), lr=LR
        )
        mdn_optimizer = torch.optim.Adam(
            self.mispronunciation_detection_network.parameters(), lr=LR
        )

        train_dataloader, validation_dataloader, test_dataloader = dataloaders

        loss_fn = torch.nn.CrossEntropyLoss()

        if fold_no == None:
            fold_no = 1

        for epoch_number in range(epochs):
            print(f"Training EPOCH:{epoch_number}")

            if training_prn:
                train_one_epoch(
                    self.phoneme_recognition_network,
                    train_dataloader,
                    loss_fn,
                    xidx=0,
                    yidx=2,
                    classes=PRN_CLF_OUT_DIM,
                    epoch_number=epoch_number,
                    optimizer=prn_optimizer,
                    fold=fold_no,
                    task="PRN",
                )

            if training_mdn:
                train_one_epoch(
                    self.mispronunciation_detection_network,
                    train_dataloader,
                    loss_fn,
                    xidx=2,
                    yidx=1,
                    classes=MDN_CLF_OUT_DIM,
                    optimizer=mdn_optimizer,
                    fold=fold_no,
                    epoch_number=epoch_number,
                    task="MDN",
                )

            print(f"Running Validation")
            if training_prn:
                train_one_epoch(
                    self.phoneme_recognition_network,
                    validation_dataloader,
                    loss_fn,
                    xidx=0,
                    yidx=2,
                    epoch_number=epoch_number,
                    classes=PRN_CLF_OUT_DIM,
                    train=False,
                    fold=fold_no,
                    task="PRN",
                )

            if training_mdn:
                train_one_epoch(
                    self.mispronunciation_detection_network,
                    validation_dataloader,
                    loss_fn,
                    xidx=2,
                    yidx=1,
                    epoch_number=epoch_number,
                    classes=MDN_CLF_OUT_DIM,
                    train=False,
                    fold=fold_no,
                    task="MDN",
                )


if __name__ == "__main__":
    pass
