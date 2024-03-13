import librosa
import soundfile as sf

def change_energy(input_audio_file, output_audio_file, energy_scaling_factor):
    """
    Change the energy (amplitude) of an audio file and save the resulting audio.
    
    Parameters:
    input_audio_file (str): Path to the input audio file (in .wav format).
    output_audio_file (str): Path to save the audio file with changed energy (in .wav format).
    energy_scaling_factor (float): Scaling factor for changing the energy of the audio.
    """
    # Load the audio file
    y, sr = librosa.load(input_audio_file)

    # Change energy by scaling the amplitude
    y_scaled = y * energy_scaling_factor

    # Save the modified audio
    sf.write(output_audio_file, y_scaled, sr)

    print("Audio with changed energy saved as:", output_audio_file)

# Example usage
input_audio_file = 'input_audio.wav'
output_audio_file = 'output_audio_energy_changed.wav'
energy_scaling_factor = 0.5  # Decrease energy by half
change_energy(input_audio_file, output_audio_file, energy_scaling_factor)
