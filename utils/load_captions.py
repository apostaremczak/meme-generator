import pandas as pd
from typing import Dict, List

DATABASE_PATH = "meme_database"


def load_category_memes(category_name: str,
                        database_path: str = DATABASE_PATH) -> pd.DataFrame:
    """
    Loads all captions within one category, located in
    <database_path>/<category_name>.json
    """
    filename = f"{DATABASE_PATH}/{category_name}.json"
    return pd.read_json(filename, lines="series")


def load_memes(categories: List[str],
               database_path: str = DATABASE_PATH) -> Dict[str, pd.DataFrame]:
    """
    Read all memes from the database.
    """
    return {
        category_name: load_category_memes(category_name, database_path)
        for category_name in categories
    }
