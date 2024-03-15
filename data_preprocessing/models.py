from transformers import VitsModel, AutoTokenizer
from transformers import AutoProcessor, BarkModel
import torch
from scipy.io.wavfile import write
from random import choice


class FaceBookTTS(object):
    def __init__(self, model_checkpoint):
        self.model_checkpoint = model_checkpoint

        self.model = VitsModel.from_pretrained(self.model_checkpoint)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)

    def __call__(self, text: str, filename: str):
        print("using mms")
        inputs = self.tokenizer(text, return_tensors="pt")

        with torch.no_grad():
            output = self.model(**inputs).waveform
            write(
                filename,
                rate=self.model.config.sampling_rate,
                data=output.float().numpy().T,
            )


class BarkTTS(object):
    def __init__(self, model_checkpoint, voice_presets):
        self.model_checkpoint = model_checkpoint
        self.voice_presets = voice_presets

        self.processor = AutoProcessor.from_pretrained("suno/bark-small")
        self.model = BarkModel.from_pretrained("suno/bark-small")

    def __call__(self, text: str, filename: str):
        print("using bark")
        inputs = self.processor(text, voice_preset=choice(self.voice_presets))

        output = self.model.generate(**inputs)
        output = output.cpu().numpy().squeeze()
        print(output.shape)
        write(filename, rate=self.model.generation_config.sample_rate, data=output)


MODELS = [
    FaceBookTTS("facebook/mms-tts-hin"),
    #    BarkTTS("suno/bark-small",[f"v2/hi_speaker_{i}" for i in range(10)])
]
