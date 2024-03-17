import numpy as np
from sklearn.preprocessing import StandardScaler

def standardize_data(data):
    if len(data.shape) > 2:
        data = data.reshape(data.shape[0], -1)

    scaler = StandardScaler()

    standardized_data = scaler.fit_transform(data)

    return standardized_data

# Example usage
if __name__ == "__main__":
    mel_spectrograms = np.random.rand(100, 128, 128)
    standardized_mel_spectrograms = standardize_data(mel_spectrograms)
    print("Shape of standardized mel spectrogram data:", standardized_mel_spectrograms.shape)
