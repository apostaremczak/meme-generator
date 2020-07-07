import pickle
import string
import unicodedata

from pandas import DataFrame
from transformers import GPT2Tokenizer
from tqdm import tqdm
from typing import Dict, List

from utils.file_loaders import load_captions
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
    if normalized in string.printable + " ":
        return normalized
    else:
        return ""


def _process_single_caption(category_token: str, caption: str) -> str:
    caption = "".join(map(_char_to_ascii, caption))
    caption = caption.replace(";", END_OF_BOX_TOKEN)
    caption_with_tokens = [
        category_token,
        caption,
        END_OF_TEXT_TOKEN
    ]
    return " ".join(caption_with_tokens)


def _create_caption_labels(tokenized_text: List[int], block_size=50):
    examples = []
    for i in range(0, len(tokenized_text) - block_size + 1, block_size):
        examples.append(tokenized_text[i:i + block_size])

    inputs, labels = [], []
    for ex in examples:
        inputs.append(ex[:-1])
        labels.append(ex[1:])

    return inputs, labels


def _tokenize_single_caption(caption: str,
                             tokenizer: GPT2Tokenizer,
                             max_seq_length: int = MAX_SEQUENCE_LENGTH):
    return tokenizer.encode(
        caption,
        max_length=max_seq_length,
        truncation=True
    )


def _tokenize_captions(captions_db: Dict[str, DataFrame],
                       tokenizer: GPT2Tokenizer):
    """
    Makes a dictionary of all captions within the categories.
    Turns the original caption lowercase and converts it to ASCII,
    filtering out captions shorter than <min_caption_len> characters.

    :param captions_db:     Dictionary of pandas data frame with the memes in
                            the format {category_name: pandas_memes}
    :param tokenizer:       Pre-trained tokenizer with added category names
                            as special tokens.
    """
    tokenized_captions = []
    for category_name, memes in tqdm(captions_db.items(),
                                     desc="Tokenizing meme captions"):
        # Create a special token for category name
        category_token = _category_name_to_token(category_name)

        # Normalize each captions and add category token
        category_captions = list(map(
            lambda caption: _process_single_caption(category_token, caption),
            memes["caption"].tolist()
        ))

        category_captions_merged = "\n".join(category_captions)
        tokenized_category = tokenizer.encode(category_captions_merged)
        tokenized_captions.extend(tokenized_category)

    inputs, labels = _create_caption_labels(tokenized_captions)
    return inputs, labels


def run():
    tokenizer = get_tokenizer(None)
    captions_db = load_captions()
    tokenized_captions, labels = _tokenize_captions(captions_db, tokenizer)

    with open("../data/train_captions.pickle", "wb") as f:
        pickle.dump(tokenized_captions, f)

    with open("../data/train_labels.pickle", "wb") as f:
        pickle.dump(labels, f)


if __name__ == '__main__':
    run()
