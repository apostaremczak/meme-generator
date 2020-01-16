from typing import List

DATABASE_PATH = "meme_database"


def read_categories(database_path: str = DATABASE_PATH,
                    categories_filename: str = "categories.txt") -> List[str]:
    with open(f"{database_path}/{categories_filename}", "r") as f:
        categories = [line.strip() for line in f]

    return categories
