import librosa
import matplotlib.pyplot as plt
import numpy as np

def compute_mel_spectrogram(audio_file):
    """
    Compute the Mel spectrogram of an audio file.
    
    Parameters:
    audio_file (str): Path to the input audio file (in .wav format).
    
    Returns:
    mel_spectrogram_db (np.ndarray): Mel spectrogram in dB scale.
    sr (int): Sample rate of the audio file.
    """
    y, sr = librosa.load(audio_file)
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    return mel_spectrogram_db, sr

def plot_mel_spectrogram(mel_spectrogram_db, sr):
    """
    Plot the Mel spectrogram.
    
    Parameters:
    mel_spectrogram_db (np.ndarray): Mel spectrogram in dB scale.
    sr (int): Sample rate of the audio file.
    """
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_spectrogram_db, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel spectrogram')
    plt.tight_layout()
    plt.show()

audio_file = 'download.wav'
mel_spectrogram_db, sr = compute_mel_spectrogram(audio_file)
plot_mel_spectrogram(mel_spectrogram_db, sr)
