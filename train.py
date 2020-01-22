import logging
import numpy as np
import torch
import torch.nn as nn
import random
import torch.optim as optim

from utils.meme_dataset import MemeDataset
from word_lstm import WordLSTM

HIDDEN_DIM = 64
LEARNING_RATE = 0.0005
NUM_EPOCHS = 5
PLOT_EVERY = 100
SAVE_MODEL_EVERY = 300
NUM_LAYERS = 5
CHECKPOINT_FILE_NAME = "shuffled_glove_lstm_checkpoints.pt"


if __name__ == '__main__':
    # Set up logging
    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                        level=logging.INFO)
    logger = logging.getLogger("LSTMTrainLogger")

    # Prepare dataset
    meme_dataset = MemeDataset(validation_fraction=0.1)

    # # Initialize model
    # rnn = WordLSTM(meme_dataset, HIDDEN_DIM, NUM_LAYERS)
    # optimizer = optim.SGD(rnn.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # Read model from checkpoints
    checkpoints = torch.load(CHECKPOINT_FILE_NAME)
    rnn = WordLSTM(meme_dataset, HIDDEN_DIM, NUM_LAYERS)
    rnn.load_state_dict(checkpoints["model_state_dict"])
    rnn.eval()

    optimizer = optim.SGD(rnn.parameters(), lr=LEARNING_RATE)
    optimizer.load_state_dict(checkpoints["optimizer_state_dict"])

    rnn.train_losses = checkpoints["train_losses"]
    rnn.validation_losses = checkpoints["validation_losses"]

    epoch_length = len(rnn.dataset.train)

    # Start training
    for epoch in range(NUM_EPOCHS):
        batch_losses = []

        for i, training_example in enumerate(rnn.dataset):
            optimizer.zero_grad()
            random.shuffle(training_example)
            for sentence, next_word in training_example:
                output = rnn(sentence)
                loss = criterion(output, next_word)
                loss.backward()
                batch_losses.append(loss.item())
            optimizer.step()

            # Make a checkpoint
            if i % SAVE_MODEL_EVERY == 0:
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": rnn.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_losses": rnn.train_losses,
                    "validation_losses": rnn.validation_losses
                }, CHECKPOINT_FILE_NAME)

            # Save loss function value on both train and validation sets
            if i % PLOT_EVERY == 0:
                avg_batch_loss = np.mean(batch_losses)
                rnn.train_losses.append(avg_batch_loss)

                with torch.no_grad():
                    valid_sentence, target_word = rnn.dataset \
                        .get_validation_example()
                    model_output = rnn(valid_sentence)
                    validation_loss = criterion(model_output, target_word)
                    rnn.validation_losses.append(validation_loss.item())

                logger.info(
                    f"Epoch: {epoch:,}/{NUM_EPOCHS:,} "
                    f"({i / epoch_length:.1%}); "
                    f"Train loss: {avg_batch_loss:.2f}, "
                    f"validation loss: {validation_loss.item():.2f};"
                )
