import pickle
import random
import torch

from torch.utils.data import Dataset
from typing import List
from utils.glove import read_vocabulary, read_glove_embeddings


class MemeDataset(Dataset):
    def __init__(self,
                 validation_fraction: float = 0.2,
                 data_file_name: str = "encoded_meme_dataset.pickle",
                 vocabulary_file_name: str = "meme_vocabulary.txt",
                 glove_embedding_file_name: str = "meme_glove_embeddings.pt"):
        super().__init__()

        # Read training examples and shuffle them
        with open(data_file_name, "rb") as f:
            self.data = pickle.load(f)
        random.shuffle(self.data)

        validation_sample_count = int(validation_fraction * len(self.data))
        self.train = self.data[:-validation_sample_count]
        self.valid = self.data[-validation_sample_count:]

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

    @staticmethod
    def generate_training_examples(caption: List[int]) -> List[List[torch.tensor]]:
        n_words = len(caption)

        caption_sequenced = []
        for i in range(1, n_words):
            input_tensor = torch.tensor([caption[:i]], dtype=torch.long)
            target_tensor = torch.tensor([caption[i]], dtype=torch.long)
            caption_sequenced.append([input_tensor, target_tensor])
        return caption_sequenced

    def __getitem__(self, index) -> List[List[torch.tensor]]:
        """
        Returns items from the training part of the dataset.
        """
        caption = self.train[index]
        return self.generate_training_examples(caption)

    def get_validation_example(self) -> List[torch.tensor]:
        """
        Get a random caption from the validation dataset.
        """
        caption = random.choice(self.valid)
        index = random.randrange(1, len(caption))
        return [
            torch.tensor([caption[:index]], dtype=torch.long),
            torch.tensor([caption[index]], dtype=torch.long)
        ]
