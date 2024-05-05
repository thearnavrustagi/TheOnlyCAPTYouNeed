import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from .constants import (
    SENTENCE_MAX_LEN,
    MS_MAX_LEN,
    N_BATCHES,
    WORD_MAX_LEN,
    VAL_SPLIT
)
from .l1_speech_dataset import L1SpeechDataset
from sklearn.model_selection import KFold


def train_val_split(full_dataset, val_percent, random_seed=None):
    amount = len(full_dataset)

    val_amount = (
        int(amount * val_percent)
        if val_percent is not None else 0)
    train_amount = amount - val_amount

    train_dataset, val_dataset = random_split(
        full_dataset,
        (train_amount, val_amount),
        generator=(
            torch.Generator().manual_seed(random_seed)
            if random_seed
            else None))
    
    return train_dataset, val_dataset

def collate_fn(data):
    audio_features = torch.zeros(len(data), data[0][0].shape[0], MS_MAX_LEN)
    transcriptions = torch.zeros(len(data), SENTENCE_MAX_LEN).type(torch.int64)
    error_p = torch.zeros(len(data), WORD_MAX_LEN)

    for idx, item in enumerate(data):
        audio_tensor = item[0]
        n_frames = item[0].shape[1]
        n_features = item[0].shape[0]
        diff = MS_MAX_LEN - n_frames
        diff = 0 if diff < 0 else diff
        audio_tensor = torch.from_numpy(audio_tensor[:, : min(MS_MAX_LEN, n_frames)])
        pad = torch.zeros(n_features, diff)
        audio_tensor = torch.hstack([audio_tensor, pad])
        audio_tensor = audio_tensor.permute(0, 1)

        transcription = np.array(item[2])
        n_token = transcription.shape[0]
        diff = SENTENCE_MAX_LEN - n_token
        diff = 0 if diff < 0 else diff
        transcription = torch.from_numpy(
            transcription[: min(SENTENCE_MAX_LEN, n_token)]
        )
        pad = torch.zeros(diff)
        transcription = torch.hstack([transcription, pad])

        e_p = np.array(item[1])
        n_token = e_p.shape[0]
        diff = WORD_MAX_LEN - n_token
        diff = 0 if diff < 0 else diff
        e_p = torch.from_numpy(e_p[: min(WORD_MAX_LEN, n_token)])
        pad = torch.zeros(diff)
        e_p = torch.hstack([e_p, pad])

        audio_features[idx] = audio_tensor
        transcriptions[idx] = transcription.type(torch.int64)
        error_p[idx] = e_p

    return audio_features, error_p, transcriptions.type(torch.int64)


def create_k_fold_dataloaders(k_folds=10, random_seed=None):
    dataset = L1SpeechDataset()
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=random_seed)
    dataloaders = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset), start=1):
        train_dataset = torch.utils.data.Subset(dataset, train_idx)
        val_dataset = torch.utils.data.Subset(dataset, val_idx)

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=N_BATCHES,
            shuffle=True,
            collate_fn=collate_fn,
        )

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=N_BATCHES,
            shuffle=False,
            collate_fn=collate_fn,
        )

        dataloaders.append((train_dataloader, val_dataloader))

    return dataloaders