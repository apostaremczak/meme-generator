import numpy as np
import string
import tensorflow as tf
import unicodedata

from logging import Logger
from pandas import DataFrame
from transformers import GPT2Tokenizer
from tqdm import tqdm
from typing import Dict, List

from utils.file_loaders import load_captions
from utils.logger import get_logger
from model.config import MemeGeneratorConfig
from model.meme_generator import get_tokenizer
from model.special_tokens import END_OF_BOX_TOKEN, END_OF_TEXT_TOKEN

DATA_RECORD_PATH = "../data/tokenized_captions.tfrecord"
MAX_SEQUENCE_LENGTH = MemeGeneratorConfig().max_seq_length


def _category_name_to_token(category_name: str) -> str:
    return f"<|{category_name}|>"


def _char_to_ascii(char: str) -> str:
    """
    Makes sure that a character is included in the allowed ASCII alphabet.
    """
    normalized = unicodedata.normalize("NFD", char)
    if normalized in string.ascii_letters:
        return normalized
    else:
        return ""


def _process_single_caption(category_token: str, caption: str) -> str:
    caption = caption.replace(";", END_OF_BOX_TOKEN)
    caption = "".join(map(_char_to_ascii, caption))
    caption_with_tokens = [
        category_token,
        caption,
        END_OF_TEXT_TOKEN
    ]
    return " ".join(caption_with_tokens)


def _tokenize_captions(captions_db: Dict[str, DataFrame],
                       tokenizer: GPT2Tokenizer,
                       max_seq_length: int = MAX_SEQUENCE_LENGTH) \
        -> List[List[int]]:
    """
    Makes a dictionary of all captions within the categories.
    Turns the original caption lowercase and converts it to ASCII,
    filtering out captions shorter than <min_caption_len> characters.

    :param captions_db:     Dictionary of pandas data frame with the memes in
                            the format {category_name: pandas_memes}
    :param tokenizer:       Pre-trained tokenizer with added category names
                            as special tokens.
    :param max_seq_length:

    :return: List of tokenized captions
    """
    processed_captions = []
    for category_name, memes in tqdm(captions_db.items(),
                                     desc="Tokenizing meme captions"):
        # Create a special token for category name
        category_token = _category_name_to_token(category_name)

        # Normalize each captions and add category token
        category_captions = list(map(
            lambda caption: _process_single_caption(category_token, caption),
            memes["caption"].tolist()
        ))

        # Tokenize all the captions
        tokenized_captions = map(
            lambda caption: tokenizer.encode(caption,
                                             max_length=max_seq_length,
                                             truncation=True),
            category_captions
        )

        processed_captions.extend(list(tokenized_captions))

    return processed_captions


def _int64_feature(value: np.ndarray) -> tf.train.Feature:
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _create_data_record(tokenized_captions: List[List[int]],
                        output_file_path: str = DATA_RECORD_PATH,
                        logger: Logger = get_logger()):
    """
    Save training data as a TFRecord for faster training
    """
    with tf.io.TFRecordWriter(output_file_path) as record_writer:
        for meme_caption in tokenized_captions:
            try:
                caption_array = np.array(meme_caption, dtype="int64")
            except ValueError as e:
                logger.error(meme_caption)
                raise e

            feature = {
                "input_token": _int64_feature(caption_array)
            }

            features = tf.train.Features(feature=feature)
            example = tf.train.Example(features=features)
            record_writer.write(example.SerializeToString())


def run():
    tokenizer = get_tokenizer(tokenizer_path="../model/tokenizer/")
    captions_db = load_captions()
    tokenized_captions = _tokenize_captions(captions_db, tokenizer)
    _create_data_record(tokenized_captions)


if __name__ == '__main__':
    run()
