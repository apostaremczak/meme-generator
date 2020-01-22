import torch
import torch.nn as nn
import torch.optim
from utils.meme_dataset import MemeDataset


class WordLSTM(nn.Module):
    def __init__(
            self,
            meme_dataset: MemeDataset,
            hidden_dim: int,
            n_lstm_layers: int = 3,
            dropout_rate: float = 0.4
    ):
        """
        Word-level LSTM RNN.
        It's supposed to take in a sequence of words, and predict the next
        word.

        :param vocabulary_size: Number of distinct words used in texts.
        :param hidden_dim: Dimension of the hidden LSTM layer.
        :param n_lstm_layers: Number of LSTM layers to be used.
        :param dropout_rate: Dropout probability, used to fight overfitting.
        """
        super().__init__()

        # GloVe word embeddings
        self.encoder = nn.Embedding.from_pretrained(
            meme_dataset.embedding_weights)

        # Don't train GloVe embeddings
        self.encoder.weight.requires_grad = False

        self.embedding_dim = meme_dataset.embedding_weights.size(1)
        self.vocabulary_size = len(meme_dataset.vocabulary)

        # Dropout layer
        self.dropout = nn.Dropout(p=dropout_rate)

        # LSTM input dim is embedding_dim, output dim is hidden_dim
        self.lstm = nn.LSTM(self.embedding_dim,
                            hidden_dim,
                            batch_first=True,
                            num_layers=n_lstm_layers,
                            dropout=dropout_rate)

        # Linear layer to map hidden states to vocabulary space
        self.decoder = nn.Linear(hidden_dim, self.vocabulary_size)

    def forward(self, sentence: torch.LongTensor) -> torch.tensor:
        encoded = self.encoder(sentence)
        dropped = self.dropout(encoded)
        lstm_output, _ = self.lstm(dropped)
        return self.decoder(lstm_output[:, -1])
