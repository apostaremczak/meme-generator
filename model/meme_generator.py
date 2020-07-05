from logging import Logger
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel
from typing import List, Optional

from model.special_tokens import SPECIAL_TOKENS
from utils.logger import get_logger

TOKENIZER_PATH = "tokenizer/"
MODEL_PATH = ""


def get_tokenizer(tokenizer_path: Optional[str] = TOKENIZER_PATH,
                  special_tokens: List[str] = SPECIAL_TOKENS,
                  logger: Logger = get_logger()) -> GPT2Tokenizer:
    """

    :param tokenizer_path:  Path to a pre-trained tokenizer. If None, a new
                            tokenizer will be created and saved after adding
                            special tokens.
    :param special_tokens:  Category name tokens, along with end of box and
                            end of text tokens.
    :param logger:          Logger object.

    :return:                Enriched GPT2Tokenizer.
    """
    if tokenizer_path is None:
        logger.info("Creating a new tokenizer with added special tokens")
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.add_special_tokens({
            "additional_special_tokens": special_tokens
        })
        tokenizer.save_pretrained(TOKENIZER_PATH)
    else:
        logger.info(f"Loading a pre-trained tokenizer from {tokenizer_path}")
        tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)

    logger.info("Successfully loaded a tokenizer")
    return tokenizer


def get_model(model_path: Optional[str] = MODEL_PATH,
              logger: Logger = get_logger()):
    if model_path is None:
        logger.info("Loading a new model")
        model = TFGPT2LMHeadModel.from_pretrained("gpt2")
    else:
        logger.info(f"Loading a pre-trained model from {model_path}")
        model = TFGPT2LMHeadModel.from_pretrained(model_path)

    logger.info("Successfully loaded GTP-2")
    return model
