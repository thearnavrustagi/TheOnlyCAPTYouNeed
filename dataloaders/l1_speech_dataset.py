from glob import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
import librosa 
from .tokenizer import tokenize

class L1SpeechDataset(Dataset):
    def __init__(self, *, data_path="./final_dataset/$split$/l1", annotation_file_name="transcript.tsv", audio_directory_name="audio", split="train"):
        self.split = split.lower()
        self.data_path = data_path.replace("$split$", split)
        self.annotation_file_name = annotation_file_name
        self.audio_directory_name = audio_directory_name

        self.annotation_file = f"{self.data_path}/{annotation_file_name}"
        self.audio_files_path = f"{self.data_path}/{audio_directory_name}"

        self.transcript_df = pd.read_csv(self.annotation_file, encoding="utf-8", sep="\t")
        self.audio_files = glob(f"{self.audio_files_path}/*")

    def __getitem__(self, idx):
        filename = self.audio_files[idx]
        file_id = "-".join(filename.split("/")[-1].split("-")[:2])
        print(file_id)
        mask = self.transcript_df["file_identifier"] == file_id
        transcription = self.transcript_df.loc[mask].iloc[0]
        error_p = eval(transcription["error_p"])
        sentence = transcription["sentence"]

        ms = np.load(filename)
        
        return (ms, np.array(error_p), tokenize(sentence))

    def __len__(self):
        return len(self.audio_files)