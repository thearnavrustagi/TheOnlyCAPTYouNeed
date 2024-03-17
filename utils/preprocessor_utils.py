import librosa
import numpy as np
import pydub

from scipy.signal import resample
from .constants import SAMPLING_RATE


def load_mucs_transcription(fpath, *, lines=None):
    with open(fpath) as file:
        if not lines:
            lines = file.read().splitlines()
        return dict(tuple(line.strip().split(maxsplit=1)) for line in lines)


def clean_audio(y, sr):
    y = resample(y, int(y.shape[0] * SAMPLING_RATE / sr))

    return y


def detect_leading_silence(sound, *, silence_threshold=-100.0, chunk_size=10):
    trim_ms = 0

    assert chunk_size > 0
    while sound[
        trim_ms : trim_ms + chunk_size
    ].dBFS < silence_threshold and trim_ms < len(sound):
        trim_ms += chunk_size

    return trim_ms


def trim_silence(y, sr):
    sound = pydub.AudioSegment(
        y.tobytes(), frame_rate=sr, sample_width=y.dtype.itemsize, channels=1
    )

    start_trim = detect_leading_silence(sound)
    end_trim = detect_leading_silence(sound.reverse())

    duration = len(sound)
    return sound[start_trim : duration - end_trim]


def ukw2wav(fname):
    audio_segment = pydub.AudioSegment.from_file(fname)
    return audio_segment.frame_rate, np.array(audio_segment.get_array_of_samples())


if __name__ == "__main__":
    print(
        load_mucs_transcription(
            "__init__.py",
            lines=["a key and value pair", "b of very cool keys", "c i want"],
        )
    )
