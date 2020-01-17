import numpy as np
import random
from functools import reduce
from typing import List, Dict


def get_training_dataset(preprocessed_memes: Dict[str, List[List[str]]],
                         vocabulary: List[str]) -> List[np.ndarray]:
    """
    :param preprocessed_memes: Dictionary of memes in the following format:
    {category_name: [[<category ID>, <caption>, ";"], ...]}
    :param vocabulary: List of unique words found in the meme dataset.
    :return: Encoded captions, where each word of the caption is substituted
    but its index in the vocabulary set.
    """
    captions = list(reduce(lambda a, b: a + b, preprocessed_memes.values()))
    indexed_vocabulary = {
        word: index
        for index, word in enumerate(vocabulary)
    }

    return [
        np.array(
            [indexed_vocabulary[word] for word in caption],
            dtype=np.int32
        )
        for caption in captions
    ]


def generate_random_training_example(training_dataset: List[np.ndarray]) \
        -> List[List[np.ndarray]]:
    caption = random.choice(training_dataset)
    # Make this caption sequential
    n_words = len(caption)
    caption_sequenced = [
        [caption[:i], caption[i:]]
        for i in range(n_words)
    ]
    return caption_sequenced
