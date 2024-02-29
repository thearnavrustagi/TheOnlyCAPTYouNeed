from phonemedb import PhonemeDataBase
from tts_hin import TTS

def t2s_modification(prompt):
    head = ""
    tail = prompt
    for char in prompt:
        print(char, end=" ")
        if char in " \n\r\t": 
            head += tail[:1]
            tail = tail[1:]
            continue
        phoneme = PhonemeDataBase.get_random_phoneme()
        yield head + phoneme + tail[1:]
        head += tail[:1]
        tail = tail[1:]

if __name__ == "__main__":
    from tqdm import tqdm
    modifications = list(t2s_modification("मेरे पास बीस बंदर हैं"))
    tts = TTS()
    for i,modified in tqdm(enumerate(modifications)):
        tts(modified, f"{i}.wav")

