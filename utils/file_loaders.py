import pandas as pd
from typing import Dict, List

CATEGORIES_FILE_PATH = f"../data/categories.txt"
CAPTIONS_PATH = f"../data/captions"


def _load_category_names(categories_file_path: str = CATEGORIES_FILE_PATH) \
        -> List[str]:
    with open(categories_file_path, "r") as f:
        categories = [line.strip().split(",") for line in f]

    return [category_name for _, category_name in categories]


def _load_single_category_memes(category_name: str,
                                database_path: str = CAPTIONS_PATH) \
        -> pd.DataFrame:
    """
    Loads all captions within one category, located in
    <path>/<category_name>.json
    """
    filename = f"{database_path}/{category_name}.json"
    return pd.read_json(filename, lines="series")


def load_captions(categories_file_path: str = CATEGORIES_FILE_PATH,
                  captions_path: str = CAPTIONS_PATH) \
        -> Dict[str, pd.DataFrame]:
    """
    Read all memes from the database.
    :return Dictionary {<category name>: <Pandas df with columns: id, caption>}
    """
    category_names = _load_category_names(categories_file_path)
    return {
        category_name: _load_single_category_memes(category_name,
                                                   captions_path)
        for category_name in category_names
    }
