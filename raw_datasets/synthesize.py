from phonemedb import PhonemeDataBase
from json import dump


class Modifier(object):
    def __init__(self):
        self.phonemedb = PhonemeDataBase(p=0.05)

    def p2p_modification(self, prompt):
        head = ""
        tail = prompt
        e_err = []
        for char in prompt:
            if phoneme := self.phonemedb.get_random_phoneme():
                e_err.append(1)
                head += phoneme
                tail = tail[1:]
            else:
                e_err.append(0)
                head += tail[:1]
                tail = tail[1:]
        return e_err, head


if __name__ == "__main__":
    from tqdm import tqdm

    data = open("./script.txt").read().splitlines()
    modifier = Modifier()
    modified_sentences = []
    e_errs = []

    for line in tqdm(data):
        e_err, mod_sent = modifier.p2p_modification(line)
        modified_sentences.append(mod_sent)
        e_errs.append(e_err)

    open("misp_script.txt", "w+").write("\n".join(modified_sentences))
    dump({"e_err": e_errs}, open("e_err.json", "w+"))
