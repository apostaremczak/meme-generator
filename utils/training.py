import numpy as np
import pickle
from functools import reduce
from typing import List, Dict

from utils.glove import read_vocabulary
from utils.load_captions import load_memes
from utils.load_categories import read_categories
from utils.preprocess_text import preprocess_captions


def generate_training_examples(caption: np.ndarray) -> List[List[np.array]]:
    n_words = len(caption)

    caption_sequenced = []
    for i in range(1, n_words):
        input_tensor = np.array([caption[:i]], dtype=np.int32)
        target_tensor = np.array([caption[i]], dtype=np.int32)
        caption_sequenced.append([input_tensor, target_tensor])
    return caption_sequenced


def encode_caption(caption: List[str],
                   category_name: str,
                   word_to_index: Dict[str, int]) -> np.array:
    encoded_caption = np.array(
        [word_to_index.get(word, word_to_index["<unk>"])
         for word in [category_name] + caption],
        dtype=np.int32
    )

    return generate_training_examples(encoded_caption)


def get_training_dataset(preprocessed_memes: Dict[str, List[List[str]]],
                         word_to_index: Dict[str, int]) -> List[List[np.ndarray]]:
    """
    :param preprocessed_memes: Dictionary of memes in the following format:
    {category_name: [[<category ID>, <caption>, ";"], ...]}
    :param word_to_index:
    :return: Encoded captions, where each word of the caption is substituted
    but its index in the vocabulary set.
    """
    encoded_captions = []
    for category_name, captions in preprocessed_memes.items():
        encoded_category = list(map(
            lambda caption: encode_caption(caption, category_name,
                                           word_to_index),
            captions
        ))
        encoded_captions.extend(
            list(
                reduce(
                    lambda a, b: a + b, encoded_category
                )
            )
        )

    return encoded_captions


def generate_and_save_training_dataset(
        training_file: str = "training_dataset.pickle"):

    all_categories = read_categories()
    memes = preprocess_captions(load_memes(list(all_categories.keys())))
    vocabulary = read_vocabulary()
    word_to_int = {
        word: i for i, word in enumerate(vocabulary)
    }

    training_ds = get_training_dataset(memes, word_to_int)

    with open(training_file, 'wb') as f:
        pickle.dump(training_ds, f)
