import torch

from csv import QUOTE_NONE
from pandas import read_table
from typing import List


def save_vocabulary(vocabulary: List[str],
                    file_name: str = "meme_vocabulary.txt"):
    with open(file_name, "w") as f:
        for word in vocabulary:
            f.write(word + "\n")


def read_vocabulary(file_name: str = "meme_vocabulary.txt"):
    with open(file_name, "r") as f:
        words = [word.strip() for word in f.readlines()]
    return words


def find_glove_embeddings(meme_vocabulary: List[str],
                          category_names: List[str],
                          glove_file: str = "glove.6B.50d.txt",
                          target_embedding_file: str = "meme_glove_embeddings.pt",
                          vocabulary_file: str = "meme_vocabulary.txt") -> torch.tensor:
    """
    Read pre-trained GloVe word embeddings.
    :param meme_vocabulary: List of words found in the memes.
    :param category_names: List of meme category names
    :param glove_file: GloVe text file name.
    :param target_embedding_file: Name of the file where embedding weights
    should be saved.
    """
    glove = read_table(
        glove_file,
        sep=" ",
        index_col=0,
        header=None,
        quoting=QUOTE_NONE
    )

    # Find the intersection of words in both datasets
    meme_vocab_set = set(meme_vocabulary)
    glove_vocab_set = set(glove.index.tolist())
    common_words = sorted(meme_vocab_set.intersection(glove_vocab_set))
    print(f"Words found in glove: {len(common_words)}"
          f"/{len(meme_vocab_set)}")

    # Use GloVe embeddings for words from the intersection,
    # and randomly set representations for category names and the "unknown
    # word" token "<unk>"
    common_glove = glove.loc[common_words]
    vocabulary = common_words + category_names + ["<unk>"]

    # Save the newest vocabulary
    save_vocabulary(vocabulary, vocabulary_file)

    embedding_size = common_glove.shape[1]
    category_embeddings = torch.rand(
        (len(category_names) + 1, embedding_size),
        dtype=torch.float64)

    meme_glove = torch.cat([
        torch.from_numpy(common_glove.to_numpy()),
        category_embeddings
    ])

    # Save embedding weights
    torch.save(meme_glove, target_embedding_file)


def read_glove_embeddings(embedding_file: str = "meme_glove_embeddings.pt"):
    """
    :return: Tensor of size (vocabulary_size, embedding_size)
    """
    return torch.load(embedding_file).float()
