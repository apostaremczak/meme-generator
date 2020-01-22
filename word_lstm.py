import torch
import torch.nn as nn
import torch.optim
from utils.meme_dataset import MemeDataset


class WordLSTM(nn.Module):
    def __init__(self,
                 dataset: MemeDataset,
                 hidden_dim: int,
                 n_lstm_layers: int = 3,
                 dropout_rate: float = 0.4,
                 train_embeddings: bool = False):
        """
        Word-level LSTM RNN.
        It's supposed to take in a sequence of words, and predict the next
        word.

        :param vocabulary_size: Number of distinct words used in texts.
        :param embedding_dim: Dimension of the word embedding layer.
        :param hidden_dim: Dimension of the hidden LSTM layer.
        :param n_lstm_layers: Number of LSTM layers to be used.
        :param dropout_rate: Dropout probability, used to fight overfitting.
        """
        super().__init__()

        # Dataset details
        self.dataset = dataset
        self.embedding_dim = dataset.embedding_weights.size(1)
        self.vocabulary_size = len(dataset.vocabulary)

        # Word embeddings
        self.encoder = nn.Embedding.from_pretrained(dataset.embedding_weights)
        self.encoder.weight.requires_grad = train_embeddings

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

        # Track model losses
        self.train_losses = []
        self.validation_losses = []

    def forward(self, sentence: torch.LongTensor) -> torch.tensor:
        x = self.encoder(sentence)
        x = self.dropout(x)
        x, _ = self.lstm(x)
        return self.decoder(x[:, -1])
