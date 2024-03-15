import librosa
import soundfile as sf
from random import random


def change_duration(
    input_audio_file, output_audio_file, *, make_faster=None, scaling_factor=None
):
    """
    Change the duration of an audio file and save the resulting audio.

    Parameters:
    input_audio_file (str): Path to the input audio file (in .wav format).
    output_audio_file (str): Path to save the audio file with changed duration (in .wav format).
    duration_scaling_factor (float): Scaling factor for changing the duration of the audio.
    """
    if not scaling_factor:
        scaling_factor = random() / 2
        if make_faster:
            scaling_factor += 1

    y, sr = librosa.load(input_audio_file)

    y_stretched = librosa.effects.time_stretch(y, rate=scaling_factor)

    sf.write(output_audio_file, y_stretched, sr)

    print("Audio with changed duration saved as:", output_audio_file)


if __name__ == "__main__":
    input_audio_file = "download.wav"
    output_audio_file = "output_audio_duration_changed.wav"
    duration_scaling_factor = 1.2
    change_duration(input_audio_file, output_audio_file, make_faster)
