from data_preprocessing import change_energy, change_duration
from glob import glob
from utils import load_mucs_transcription, make_variations_and_save
from scipy.io import wavfile
from sklearn.decomposition import FastICA
from tqdm import tqdm
import numpy as np

class Preprocessor(object):
    def __init__(
        self,
        *,
        mucs_path="./raw_datasets/mucs",
        l2_ds_path="./raw_datasets/l2",
        speech_synthesis_path="./raw_datasets/speech_synthesis",
        l1_final_path="./final_dataset/train/l1",
        l2_final_path="./final_dataset/train/l2",
        test_final_path="./final_dataset/test"
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
        self.__preprocess_mucs_split__(split="test")
        #self.__preprocess_mucs_split__()

    def __preprocess_mucs_split__(self, *, split="train"):
        transcription = load_mucs_transcription(f"{self.mucs_path}/{split}/transcription.txt")
        audio_files = enumerate(glob(f"{self.mucs_path}/{split}/audio/*"))
        out_path = self.test_final_path if split == "test" else split.l1_final_path

        for idx, audio_file in tqdm(audio_files):
            basename = audio_file.split("/")[-1]
            file_identifier = basename.split(".")[0]
            sentence = transcription[file_identifier]
            sr, audio_data = wavfile.read(audio_file)
            with open(f"{out_path}/transcript.txt", "a") as file:
                error_p = str([0]*len(sentence.strip().split()))
                file.write(f'"{idx}", "{error_p}", "{sentence}"\n')
            make_variations_and_save(f"{out_path}/audio/{idx}", sr, audio_data)

if __name__ == "__main__":
    print("[INFO] Starting data preprocessing")
    preprocessor = Preprocessor()
    preprocessor.preprocess()
