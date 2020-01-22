import pickle
from typing import List, Dict

from utils.glove import read_vocabulary
from utils.load_captions import load_memes
from utils.load_categories import read_categories
from utils.preprocess_text import preprocess_captions


def encode_caption(caption: List[str], category_name: str,
                   word_to_index: Dict[str, int]) -> List[int]:
    """
    Encodes a sequence of words with their indices. If a word is not found,
    it will be encoded with the unknown token - <unk>.

    :param caption: List of words
    :param category_name: Full category name
    :param word_to_index: Mapping of words to indices, as later used with
    embeddings.
    :return: List of indices of words' embeddings
    """
    return [
        word_to_index.get(word, word_to_index["<unk>"])
        for word in [category_name] + caption
    ]


def prepare_dataset(preprocessed_memes: Dict[str, List[List[str]]],
                    word_to_index: Dict[str, int]) -> List[List[int]]:
    """
    :param preprocessed_memes: Dictionary of memes in the following format:
    {category_name: [[<category ID>, <caption>, ";"], ...]}
    :param word_to_index: Mapping of words to indices, as later used with
    embeddings.
    :return: Encoded captions, where each word of the caption is substituted
    with its index in the vocabulary set.
    """
    encoded_captions = []
    for category_name, captions in preprocessed_memes.items():
        encoded_category = list(map(
            lambda caption: encode_caption(caption, category_name,
                                           word_to_index),
            captions
        ))
        encoded_captions.extend(encoded_category)

    return encoded_captions


def generate_and_save_training_dataset(
        dataset_file_name: str = "encoded_meme_dataset.pickle"):
    all_categories = read_categories()
    memes = preprocess_captions(load_memes(list(all_categories.keys())))
    vocabulary = read_vocabulary()
    word_to_int = {
        word: i for i, word in enumerate(vocabulary)
    }

    training_ds = prepare_dataset(memes, word_to_int)

    with open(dataset_file_name, 'wb') as f:
        pickle.dump(training_ds, f)
