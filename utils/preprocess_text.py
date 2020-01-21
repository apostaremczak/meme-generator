import unicodedata
from functools import reduce
from nltk.tokenize import TweetTokenizer
from pandas import DataFrame
from string import ascii_letters
from typing import Dict, List

ALLOWED_ALPHABET = ascii_letters + " .,;'-"
MIN_CAPTION_LENGTH = 5


def normalize_caption(caption: str,
                      allowed_alphabet: str = ALLOWED_ALPHABET) -> str:
    """
    Convert Unicode string to lowercase ASCII.
    The caption has the end of line character added (';').
    """

    def char_to_ascii(char: str) -> str:
        """
        Makes sure that a character is included in the allowed ASCII alphabet.
        """
        normalized = unicodedata.normalize("NFD", char)
        if normalized in allowed_alphabet:
            return normalized
        else:
            return ""

    return "".join(map(char_to_ascii, caption.lower())) + ";"


def preprocess_captions(captions_db: Dict[str, DataFrame],
                        min_caption_len: int = MIN_CAPTION_LENGTH) \
        -> Dict[str, List[List[str]]]:
    """
    Makes a dictionary of all captions within the categories.
    Turns the original caption lowercase and converts it to ASCII,
    filtering out captions shorter than <min_caption_len> characters.

    :param captions_db: Dictionary of pandas data frame with the memes in
    the format {category_name: pandas_memes}
    :param min_caption_len: Minimum length of a caption for it not to be
    removed during preprocessing.

    :return: Dictionary of memes in the following format:
    {category_name: [[<category name>, <caption>, ";"], ...]}
    """
    processed_captions = {}
    for category_name, memes in captions_db.items():
        # Normalize each captions
        category_captions = map(normalize_caption, memes["caption"].tolist())

        # Filter out captions of length smaller than min_caption_len
        filtered_captions = list(
            filter(
                lambda caption: len(caption) >= min_caption_len,
                category_captions
            )
        )

        # Use word tokenizer
        tokenizer = TweetTokenizer()
        tokenized_captions = list(map(tokenizer.tokenize, filtered_captions))

        processed_captions[category_name] = tokenized_captions

    return processed_captions


def get_alphabet(caption_db: Dict[str, List[str]]) -> str:
    alphabet = set()
    for category_memes in caption_db.values():
        captions_alphabets = [set(caption) for caption in category_memes]
        category_alphabet = reduce(lambda a, b: a.union(b), captions_alphabets)
        alphabet = alphabet.union(category_alphabet)
    return "".join(sorted(alphabet))


def get_vocabulary(caption_db: Dict[str, List[List[str]]]) -> List[str]:
    """
    :param caption_db: Meme database, already preprocessed, in the format:
    {category_name: [[words in caption 1], ...]}
    :return: List of unique words in the meme database, ordered alphabetically.
    """
    vocabulary = set()
    for category_captions in caption_db.values():
        category_words_sets = list(map(set, category_captions))
        category_words = reduce(lambda a, b: a.union(b), category_words_sets)
        vocabulary = vocabulary.union(category_words)
    return sorted(vocabulary)
