from transformers import VitsModel, AutoTokenizer
import torch
from scipy.io.wavfile import write

class TTS(object):
    def __init__(self, model_checkpoint:str="facebook/mms-tts-hin", tokenizer_checkpoint:str="facebook/mms-tts-hin"):
        self.model_checkpoint = model_checkpoint
        self.tokenizer_checkpoint = tokenizer_checkpoint

        self.model = VitsModel.from_pretrained(self.model_checkpoint)
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_checkpoint)

    def __call__(self, text:str,filename:str="output.wav") -> None:
        inputs = self.tokenizer(text, return_tensors="pt")

        with torch.no_grad():
            output = self.model(**inputs).waveform
            write(filename, rate=self.model.config.sampling_rate, data=output.float().numpy().T)

if __name__ == "__main__":
    tts = TTS()
    tts("मेरे पास बीस बकदर हैं")
    #tts("मेरे पास बीस बंदर हैक")