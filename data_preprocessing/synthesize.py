from phonemedb import PhonemeDataBase
from tts_hin import TTS

class Modifier(object):
    def __init__(self): 
        self.phonemedb = PhonemeDataBase()

    def t2s_modification(self,prompt):
        head = ""
        tail = prompt
        for char in prompt:
            print(char, end=" ")
            if char in " \n\r\t": 
                head += tail[:1]
                tail = tail[1:]
                continue
            phoneme = self.phonemedb.get_random_phoneme()
            yield head + phoneme + tail[1:]
            head += tail[:1]
            tail = tail[1:]

if __name__ == "__main__":
    from tqdm import tqdm
    modifier = Modifier()
    modifications = list(modifier.t2s_modification("मेरे पास बीस बंदर हैं"))
    tts = TTS()
    print("starting tts")
    for i,modified in tqdm(enumerate(modifications)):
        tts(modified, f"./dump/{i}.wav")

