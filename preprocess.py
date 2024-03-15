from glob import glob
from scipy.io import wavfile
from sklearn.decomposition import FastICA
from tqdm import tqdm
import numpy as np

from utils import load_mucs_transcription, clean_audio
from data_preprocessing import change_energy, change_speed
from constants import SAMPLING_RATE


def make_variations_and_save(path, sr, audio):
    funcs = [
            lambda y: y,
            lambda y: change_energy(y, increase_energy=False),
            lambda y: change_energy(y, increase_energy=True),
            lambda y: change_speed(y, make_faster=False),
            lambda y: change_speed(y, make_faster=True)
        ]

    for i, func in enumerate(funcs):
        modified_audio = func(audio.astype(np.float32))
        wavfile.write(f"{path}-v{i}.wav", sr, modified_audio.astype(np.int16))

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
                file.write("file_identifier, error_p, sentence\n")

        self.__preprocess_mucs__()

    def __preprocess_mucs__(self):
        """
        print("[INFO] Preprocessing MUCS:train")
        self.__preprocess_mucs_split__(split="train")
        print("[SUCCESS] Preprocessing MUCS:train")
        print("[INFO] Preprocessing MUCS:test")
        self.__preprocess_mucs_split__(split="test")
        print("[SUCCESS] Preprocessing MUCS:test")
        """

        print("[INFO] Preprocessing L2:all")
        self.__preprocess_l2_split__(split="train")
        print("[SUCCESS] Preprocessing L2:all")

        print("\n[INFO] Preprocessing SpeechSynthesis:all")
        self.__preprocess_speech_synthesis_split__(split="train")
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
                error_p = str([0] * len(sentence.strip().split()))
                file.write(f'"{idx}", "{error_p}", "{sentence}"\n')
            make_variations_and_save(
                f"{out_path}/audio/{idx}", SAMPLING_RATE, audio_data
            )

    def __preprocess_l2_split__(self, test_train_split=0.1):
        out_path = self.l2_final_path
        audio_files = list(enumerate(glob(f"{self.l2_ds_path}/audio/*")))

        for i, audio_file in tqdm(audio_files):
            idx = f"l2-{i}"
            sr, audio_data = wavfile.read(audio_file)
            audio_data = clean_audio(audio_data.astype(np.int16), sr).astype(np.float32)
            make_variations_and_save(
                f"{out_path}/audio/{idx}", SAMPLING_RATE, audio_data
            )

    def __preprocess_speech_synthesis_split__(self):
        pass


if __name__ == "__main__":
    print("[INFO] Starting data preprocessing")
    preprocessor = Preprocessor()
    preprocessor.preprocess()
