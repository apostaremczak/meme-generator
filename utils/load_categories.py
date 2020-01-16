from typing import Dict

DATABASE_PATH = "meme_database"


def read_categories(database_path: str = DATABASE_PATH,
                    categories_filename: str = "categories.txt",
                    padded_id_length: int = 12) \
        -> Dict[str, str]:
    """
    :return: Dictionary {category_name: category_id}.
    All IDs have the same length, padded with zeroes if necessary.
    """
    with open(f"{database_path}/{categories_filename}", "r") as f:
        categories = [line.strip().split(",") for line in f]

    categories = {
        category_name: category_id.zfill(padded_id_length)
        for category_id, category_name in categories
    }

    return categories
