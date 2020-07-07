import os
import pickle
import tensorflow as tf
from datetime import datetime
from typing import Optional

from utils.logger import get_logger
from model.meme_generator import get_model

DATA_PATH = "data"
CHECKPOINT_DIR = "checkpoint"
LOGS_DIR = "logs"
NUM_EPOCHS = 10


def run(num_epochs: int = NUM_EPOCHS,
        learning_rate: float = 1e-3,
        validation_split: float = 0.2,
        resume_from: Optional[str] = None,
        data_path: str = DATA_PATH,
        checkpoint_dir: str = CHECKPOINT_DIR,
        logs_dir: str = LOGS_DIR):
    logger = get_logger()

    with open(f"{data_path}/train_captions.pickle", "rb") as f:
        captions = pickle.load(f)
        captions = tf.convert_to_tensor(captions)

    with open(f"{data_path}/train_labels.pickle", "rb") as f:
        labels = pickle.load(f)
        labels = tf.convert_to_tensor(labels)

    model = get_model(model_weights_path=resume_from, logger=logger)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    model.compile(optimizer=optimizer,
                  loss=[loss],
                  metrics=["sparse_categorical_accuracy"])

    # Prepare training callbacks
    start_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    checkpoint_path_base = os.path.join(checkpoint_dir,
                                        f"training_{start_time}")
    trial_checkpoint_path = f"{checkpoint_path_base}/cp.ckpt"
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        trial_checkpoint_path,
        save_weights_only=True,
        verbose=1
    )

    trial_log_dir = os.path.join(logs_dir, start_time)
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=trial_log_dir)

    # Train the model
    model.fit(x=captions, y=labels, epochs=num_epochs, batch_size=1,
              callbacks=[cp_callback, tb_callback],
              validation_split=validation_split)
    logger.info("Finished training")

    # Save full model and its weights just in case
    logger.info("Saving model weights")
    model.save(f"{checkpoint_path_base}/final_model")
    model.save_weights(f"{checkpoint_path_base}/final_weights")

    model.save_pretrained(f"{checkpoint_path_base}/transformer_pretrained")

    logger.info(f"Saved trained model to {checkpoint_path_base}")
