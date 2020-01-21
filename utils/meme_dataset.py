import pickle
import torch
from random import shuffle
from torch.utils.data import Dataset
from utils.glove import read_vocabulary, read_glove_embeddings


class MemeDataset(Dataset):
    def __init__(self,
                 data_file_name: str = "training_dataset.pickle",
                 vocabulary_file_name: str = "meme_vocabulary.txt",
                 glove_embedding_file_name: str = "meme_glove_embeddings.pt"):
        # Read training examples and shuffle them
        with open(data_file_name, "rb") as f:
            self.data = pickle.load(f)
        shuffle(self.data)

        # Read dictionary of words used in the examples
        self.vocabulary = read_vocabulary(vocabulary_file_name)

        # Read embedding weights
        self.embedding_weights = read_glove_embeddings(
            glove_embedding_file_name)

        # Create encoding dictionary
        self.word_to_index = {
            word: i for i, word in enumerate(self.vocabulary)
        }

        # Create decoding dictionary
        self.index_to_word = {
            i: word for i, word in enumerate(self.vocabulary)
        }

    def decode_index(self, index: int) -> str:
        return self.index_to_word[index]

    def encode_word(self, word: str) -> int:
        return self.word_to_index[word]

    def get_word_embedding(self, word: str) -> torch.tensor:
        return self.embedding_weights[self.encode_word(word)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sentence, next_word = self.data[index]
        return torch.from_numpy(sentence).type(torch.long), \
               torch.from_numpy(next_word).type(torch.long)
