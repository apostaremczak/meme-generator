import unicodedata
from functools import reduce
from pandas import DataFrame
from string import ascii_letters, digits
from typing import Dict, List

ALLOWED_ALPHABET = ascii_letters + " .,;'-" + digits
MIN_CAPTION_LENGTH = 5


def normalize_caption(caption: str,
                      allowed_alphabet: str = ALLOWED_ALPHABET) -> str:
    """
    Convert Unicode string to lowercase ASCII.
    The caption has the end of line character added (';').
    """

    def char_to_ascii(char: str) -> str:
        """
        """
        normalized = unicodedata.normalize("NFD", char)
        if normalized in allowed_alphabet:
            return normalized
        else:
            return ""

    return "".join(map(char_to_ascii, caption.lower())) + ";"


def preprocess_captions(captions_db: Dict[str, DataFrame],
                        min_caption_len: int = MIN_CAPTION_LENGTH) \
        -> Dict[str, List[str]]:
    """
    Makes a dictionary of all captions within the categories.
    Turns the original caption lowercase and converts it to ASCII,
    filtering out captions shorter than <min_caption_len> characters.
    """
    processed_captions = {}
    for category_name, memes in captions_db.items():
        category_captions = map(normalize_caption, memes["caption"].tolist())
        # Filter out captions of length smaller than min_caption_len
        processed_captions[category_name] = list(
            filter(
                lambda caption: len(caption) >= min_caption_len,
                category_captions
            )
        )

    return processed_captions


def get_alphabet(caption_db: Dict[str, List[str]]) -> str:
    alphabet = set()
    for category_memes in caption_db.values():
        captions_alphabets = [set(caption) for caption in category_memes]
        category_alphabet = reduce(lambda a, b: a.union(b), captions_alphabets)
        alphabet = alphabet.union(category_alphabet)
    return "".join(sorted(alphabet))
