import torch
from utils import train_one_epoch, validate_model
from models import MispronunciationDetectionNetwork, PhonemeRecognitionNetwork
from hyperparameters import N_EPOCHS


class Dhvani(torch.nn.Module):
    def __init__(
        self,
        *,
        phoneme_encoder: torch.nn.Module,
        mel_spectrogram_encoder: torch.nn.Module,
        phoneme_decoder: torch.nn.Module,
        word_decoder: torch.nn.Module,
        phoneme_vocabulary: int,
        mdn_classification_head: torch.nn.Module,
        prn_classification_head: torch.nn.Module,
    ):
        super(Dhvani, self).__init__()

        self.phoneme_recognition_network = PhonemeRecognitionNetwork(
            phoneme_vocabulary=phoneme_vocabulary,
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

    """
    dataloaders = (train_dataloader, validation_dataloader, test_dataloader)
    """

    def train(self, dataloaders, *, epochs=N_EPOCHS):
        prn_optimizer = torch.optim.Adam(self.phoneme_recognition_network.parameters())
        mdn_optimizer = torch.optim.Adam(
            self.mispronunciation_detection_network.parameters()
        )

        train_dataloader, validation_dataloader, test_dataloader = dataloaders

        prn_loss_fn = torch.nn.CrossEntropyLoss()
        mdn_loss_fn = torch.nn.BCELoss()

        for epoch_number in range(epochs):
            print(f"Training EPOCH:{epoch_number}")
            train_one_epoch(
                self.phoneme_recognition_network,
                train_dataloader,
                prn_optimizer,
                prn_loss_fn,
                xidx=0,
                yidx=2,
            )
            train_one_epoch(
                self.mispronunciation_detection_network,
                train_dataloader,
                mdn_optimizer,
                mdn_loss_fn,
                xidx=2,
                yidx=1,
            )
            print(f"Running Validation")
            validate_model(
                self.phoneme_recognition_network,
                validation_dataloader,
                prn_loss_fn,
                xidx=0,
                yidx=2,
            )
            validate_model(
                self.mispronunciation_detection_network,
                validation_dataloader,
                mdn_loss_fn,
                xidx=2,
                yidx=1,
            )


if __name__ == "__main__":
    pass
