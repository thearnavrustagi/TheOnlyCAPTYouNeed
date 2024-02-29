# This file converts words to phoneme
import numpy as np

from typing import List
from functools import wraps
from types import GeneratorType

def listify(func):
    """decorator for making generator functions return a list instead"""
    @wraps(func)
    def new_func(*args, **kwargs):
        r = func(*args, **kwargs)
        if isinstance(r, GeneratorType):
            return list(r)
        else:
            return r
    return new_func

class Phonemizer(object):
    """
    The Phonemizer object is responsible for
    converting a word to its corresponding encoding
    representation
    """
    def __init__(self):
        """
        Initialise the phonemizer object
        """
        self.phoneme_map = []

    def __call__(self, words:List[str] | List[List[str]]) -> List[int]:
        """
        Call the phonemizer object, this is the function
        we will be using to convert strings to integer lists
        """
        if type(word[0]) == list:
            return np.array(self.word2phoneme(word) for word in words)
        return self.word2phoneme(words)

    @listify
    def word2phoneme(self, word:str) -> List[int]:
        """
        Converts a word to a phoneme equivalent list,
        all phonemes are if stored as integer ids, and each id
        corresponds to one and only one phoneme
        """
        phonemes = self.get_corresponding_phonemes(word)
        for phoneme in phonemes:
            if phoneme not in self.phoneme_map:
                yield len(phonemes)
                phonemes.append(phoneme)
            else:
                yield phonemes.index(phoneme)
    
    def get_corresponding_phonemes(self, word:str) -> List[str]:
        """
        takes a word, and returns a list of phonemes
        """
        # todo
        return []


if __name__ == "__main__":
    # example usage
    phonemizer = Phonemizer()
    phonemizer.word2phoneme("कर्त्तव्य शब्द का अभिप्राय उन कार्यों से होता है, जिन्हें करने के लिए व्यक्ति नैतिक रूप से प्रतिबद्ध होता है")
