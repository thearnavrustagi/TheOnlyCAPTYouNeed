import librosa
import soundfile as sf
from random import random


def change_speed(y, *, make_faster=None, scaling_factor=None):
    if not scaling_factor:
        scaling_factor = 0.5 + random() / 2
        if make_faster:
            scaling_factor += 0.5

    y_stretched = librosa.effects.time_stretch(y, rate=scaling_factor)
    return y_stretched


if __name__ == "__main__":
    input_audio_file = "download.wav"
    output_audio_file = "output_audio_duration_changed.wav"
    duration_scaling_factor = 1.2
    change_duration(input_audio_file, output_audio_file, make_faster)
