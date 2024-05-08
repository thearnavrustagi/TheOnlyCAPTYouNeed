from .constants import END_OF_SENTENCE, END_OF_WORD

phonemes = """
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


def tokenize(arg):
    global phonemes
    arg = arg.replace(" ", "")
    tokenized_line = []

    for char in arg:
        if char in phonemes:
            tokenized_line.append(phonemes.index(char) + 1)
        elif char == " ":
            tokenized_line.append(END_OF_WORD)

    return tokenized_line + [END_OF_SENTENCE]
