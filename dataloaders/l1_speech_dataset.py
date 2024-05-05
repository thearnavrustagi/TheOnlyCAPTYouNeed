from glob import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
import librosa
from .tokenizer import tokenize
from sklearn.preprocessing import StandardScaler

class L1SpeechDataset(Dataset):
    def __init__(self, *, data_path="./final_dataset/$split$/l1", annotation_file_name="temp.txt", audio_directory_name="audio", split="train"):
        self.split = split.lower()
        self.data_path = data_path.replace("$split$", split)
        self.annotation_file_name = annotation_file_name
        self.audio_directory_name = audio_directory_name
        self.annotation_file = f"{self.data_path}/{annotation_file_name}"
        self.audio_files_path = f"{self.data_path}/{audio_directory_name}"
        self.transcript_df = pd.read_csv(
            self.annotation_file, encoding="utf-8", sep="\t"
        )
        self.transcript_df = self.transcript_df[
            self.transcript_df["file_identifier"].str.startswith("ss-")
            | self.transcript_df["file_identifier"].str.startswith("mss-")
        ]
        self.transcript_df = self.transcript_df.map(
            lambda x: x.strip() if isinstance(x, str) else x
        )
        self.audio_files = glob(f"{self.audio_files_path}/*")

    def __getitem__(self, idx):
        base_idx = idx // 5
        version_idx = idx % 5
        file_id = self.transcript_df["file_identifier"].iloc[base_idx]
        filename = f"{self.audio_files_path}/{file_id}-v{version_idx}.npy"
        mask = self.transcript_df["file_identifier"] == file_id
        transcription = self.transcript_df.loc[mask].iloc[0]
        error_p = eval(transcription["error_p"])
        sentence = transcription["sentence"]
        ms = np.load(filename)
        scaler = StandardScaler()
        ms_scaled = scaler.fit_transform(ms.reshape(-1, 1)).flatten()
        return (ms_scaled, np.array(error_p), tokenize(sentence))

    def __len__(self):
        return len(self.transcript_df) * 5