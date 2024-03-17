import librosa
import soundfile as sf
from random import random


def change_energy(y, *, increase_energy=None, scaling_factor=None):
    """
    Change the energy (amplitude) of an audio file and save the resulting audio.

    Parameters:
    input_audio_file (str): Path to the input audio file (in .wav format).
    output_audio_file (str): Path to save the audio file with changed energy (in .wav format).
    energy_scaling_factor (float): Scaling factor for changing the energy of the audio.
    """
    if not scaling_factor:
        scaling_factor = random() / 2 + 0.5
        if increase_energy:
            scaling_factor += 1

    y_scaled = y * scaling_factor

    return y_scaled


# Example usage
if __name__ == "__main__":
    input_audio_file = "input_audio.wav"
    output_audio_file = "output_audio_energy_changed.wav"
    energy_scaling_factor = 0.5  # Decrease energy by half
    change_energy(input_audio_file, output_audio_file, energy_scaling_factor)
