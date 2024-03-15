from typing import List
from random import choice
from models import MODELS


class TTS(object):
    def __init__(
        self,
        models: List[str] = MODELS,
        tokenizer_checkpoint: str = "facebook/mms-tts-hin",
    ):
        self.models = models

    def __call__(self, text: str, filename: str = "output.wav") -> None:
        choice(self.models)(text, filename)


if __name__ == "__main__":
    tts = TTS()
    tts("मेरे पास बीस बकदर हैं")
    # tts("मेरे पास बीस बंदर हैक")
