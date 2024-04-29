from glob import glob
import os
from scipy.io import wavfile
from sklearn.decomposition import FastICA
from tqdm import tqdm
import numpy as np
import librosa
import subprocess
from random import random

from utils import load_mucs_transcription, clean_audio, ukw2wav
from utils.constants import SAMPLING_RATE
from data_preprocessing import change_energy, change_speed
from json import load


def make_variations_and_save(path, sr, audio, for_test=False):
    funcs = [
        lambda y: y,
        lambda y: change_energy(y, increase_energy=False),
        lambda y: change_energy(y, increase_energy=True),
        lambda y: change_speed(y, make_faster=False),
        lambda y: change_speed(y, make_faster=True),
    ]

    if for_test:
        funcs = funcs[:1]

    for i, func in enumerate(funcs):
        y = func(audio.astype(np.float32))
        file = open(f"{path}-v{i}.npy", "wb")
        ms = librosa.feature.melspectrogram(y=y, sr=sr)
        np.save(file, ms)


class Preprocessor(object):
    def __init__(
        self,
        *,
        mucs_path="./raw_datasets/mucs",
        l2_ds_path="./raw_datasets/l2",
        speech_synthesis_path="./raw_datasets/speech_synthesis",
        l1_final_path="./final_dataset/train/l1",
        l2_final_path="./final_dataset/train/l2",
        test_final_path="./final_dataset/test",
    ):
        self.mucs_path = mucs_path
        self.l2_ds_path = l2_ds_path
        self.speech_synthesis_path = speech_synthesis_path

        self.l1_final_path = l1_final_path
        self.l2_final_path = l2_final_path
        self.test_final_path = test_final_path

        self.final_paths = [l1_final_path, l2_final_path, test_final_path]

    def preprocess(self):
        for path in self.final_paths:
            with open(f"{path}/transcript.txt", "w+") as file:
                file.write("file_identifier\terror_p\tsentence\n")
                if path + "/audio" not in glob(f"{path}/*"):
                    os.mkdir(path + "/audio")

        self.__preprocess_data__()

    def __preprocess_data__(self):
        print("[INFO] Preprocessing MUCS:train")
        self.__preprocess_mucs_split__(split="train")
        print("[SUCCESS] Preprocessing MUCS:train")
        print("[INFO] Preprocessing MUCS:test")
        self.__preprocess_mucs_split__(split="test")
        print("[SUCCESS] Preprocessing MUCS:test")

        print("[INFO] Preprocessing L2:all")
        self.__preprocess_l2_split__()
        print("[SUCCESS] Preprocessing L2:all")

        print("\n[INFO] Preprocessing SpeechSynthesis:all")
        self.__preprocess_speech_synthesis_split__()
        print("[SUCCESS] Preprocessing SpeechSynthesis:all")

    def __preprocess_mucs_split__(self, *, split="train"):
        transcription = load_mucs_transcription(
            f"{self.mucs_path}/{split}/transcription.txt"
        )
        audio_files = list(enumerate(glob(f"{self.mucs_path}/{split}/audio/*")))
        out_path = self.test_final_path if split == "test" else self.l1_final_path

        for i, audio_file in tqdm(audio_files):
            idx = f"mucs-{i}"
            basename = audio_file.split("/")[-1]
            file_identifier = basename.split(".")[0]
            sentence = transcription[file_identifier]

            sr, audio_data = wavfile.read(audio_file)
            audio_data = clean_audio(audio_data.astype(np.int16), sr).astype(np.float32)

            with open(f"{out_path}/transcript.txt", "a") as file:
                error_p = str([0] * len(sentence.strip().split(" ")))
                file.write(f'"{idx}"\t"{error_p}"\t"{sentence}"\n')
            make_variations_and_save(
                f"{out_path}/audio/{idx}",
                SAMPLING_RATE,
                audio_data,
                for_test=split == "test",
            )

    def __preprocess_l2_split__(self):
        out_path = self.l2_final_path
        audio_files = list(enumerate(glob(f"{self.l2_ds_path}/audio/*")))

        for i, audio_file in tqdm(audio_files):
            idx = f"l2-{i}"
            sr, audio_data = ukw2wav(audio_file)
            if len(audio_data.shape) == 2:
                audio_data = np.mean(audio_data, axis=-1)
            audio_data = clean_audio(audio_data.astype(np.int16), sr).astype(np.float32)
            make_variations_and_save(
                f"{out_path}/audio/{idx}", SAMPLING_RATE, audio_data
            )

    def __preprocess_speech_synthesis_split__(self, *, test_train_split=0.15):
        out_path = self.l1_final_path
        correct_data_path = lambda fname: f"{self.speech_synthesis_path}/data/{fname}"
        incorrect_data_path = (
            lambda fname: f"{self.speech_synthesis_path}/misp_data/{fname}"
        )
        correct_data = glob(correct_data_path("*.mp3"))
        incorrect_data = glob(incorrect_data_path("*.mp3"))

        basename = lambda fname: fname.split("/")[-1]

        audio_files = list(enumerate(correct_data + incorrect_data))
        error_p = []
        correct_script = []
        incorrect_script = []
        with open(f"{self.speech_synthesis_path}/e_err.json") as fpointer:
            error_p = load(fpointer)["e_err"]
        with open(f"{self.speech_synthesis_path}/script.txt") as file:
            correct_script = file.read().splitlines()
        with open(f"{self.speech_synthesis_path}/misp_script.txt") as file:
            incorrect_script = file.read().splitlines()

        for i, file in tqdm(audio_files[::-1]):
            file_id = int(basename(file).split(".")[0])
            sentence = correct_script[file_id]

            idx = f"ss-{i}"
            err_p = str([0] * len(sentence.strip().split(" ")))

            if "misp_data" in file:
                err_p = error_p[file_id]
                idx = "m" + idx
                sentence = incorrect_script[file_id]

            for_test = random() < test_train_split
            out_path = self.test_final_path if for_test else self.l1_final_path
            sr, audio_data = ukw2wav(file)
            audio_data = clean_audio(audio_data, sr)
            line = f'"{idx}"\t"{err_p}"\t"{sentence}"\r\n'
            with open(f"{out_path}/transcript.txt", "a") as file:
                file.write(line)
            make_variations_and_save(
                f"{out_path}/audio/{idx}", SAMPLING_RATE, audio_data, for_test=for_test
            )


if __name__ == "__main__":
    print("[INFO] Deleting all the pre-existing dataset")
    subprocess.run(["bash", "clean_final_dataset.bash"])
    print("[INFO] Starting data preprocessing")
    preprocessor = Preprocessor()
    preprocessor.preprocess()
