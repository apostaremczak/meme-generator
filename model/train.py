import pickle
import tensorflow as tf

from model.meme_generator import get_tokenizer, get_model

with open("../data/train_captions.pickle", "rb") as f:
    captions = pickle.load(f)
    captions = tf.convert_to_tensor(captions)

with open("../data/train_labels.pickle", "rb") as f:
    labels = pickle.load(f)
    labels = tf.convert_to_tensor(labels)

tokenizer = get_tokenizer()
gpt_model = get_model(model_path=None)

optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08,
                                     clipnorm=1.0)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

gpt_model.compile(optimizer=optimizer,
                  loss=[loss, *[None] * gpt_model.config.n_layer],
                  metrics=["sparse_categorical_accuracy"])

gpt_model.fit(x=captions, y=labels, epochs=1, batch_size=1)
