from random import choice, random


class PhonemeDataBase(object):
    def __init__(self, *, p=0.2):
        self.probability = p
        self.phonemes = """
ँ
ं
ः
अ
आ
इ
ई
उ
ऊ
ए
ए
ओ
औ
घ
च
छ
ज
झ
ट
ठ
ड
ढ
ण
त
थ
द
ध
न
ऩ
प
फ
ब
भ
म
य
र
ऱ
ल
व
श
ष
स
ह
ा
ि
ी
ु
ू
ृ
ॅ
े
ै
ॉ
ो
ौ
्
क़
ख़
ग़
ज़
ड़
ढ़
फ़
य़
""".split()

    def get_random_phoneme(self):
        if random() < self.probability:
            return choice(self.phonemes)
        else:
            return None


if __name__ == "__main__":
    pdb = PhonemeDataBase()
    print(len(pdb.phonemes))
