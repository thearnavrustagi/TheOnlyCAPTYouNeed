import torch
import numpy as np
from torch.utils.data import DataLoader
from .constants import (
    SENTENCE_MAX_LEN,
    MS_MAX_LEN,
    N_BATCHES,
    WORD_MAX_LEN
)
from .l1_speech_dataset import L1SpeechDataset


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

        transcription = item[1]
        n_token = item[1].shape[0]
        diff = SENTENCE_MAX_LEN - n_token
        diff = 0 if diff < 0 else diff
        transcription = torch.from_numpy(
            transcription[: min(SENTENCE_MAX_LEN, n_token)]
        )
        pad = torch.zeros(diff)
        transcription = torch.hstack([transcription, pad])

        e_p = np.array(item[2])
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


train_dataloader = DataLoader(
    L1SpeechDataset(), shuffle=True, batch_size=N_BATCHES, collate_fn=collate_fn
)
