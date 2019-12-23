import json
import logging
import requests
from bs4 import BeautifulSoup
from requests import get
from selenium import webdriver
from time import sleep
from typing import List

BASE_URL = "https://imgflip.com"
CATEGORIES_FILE = "../meme_database/categories.txt"
SAVING_DIR = "../meme_database"


class Scraper:
    def __init__(self, base_url: str, categories: List[str]):
        self.base_url = base_url
        self.categories = categories
        self.saving_dir = SAVING_DIR

        # Set logging details
        logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                            level=logging.INFO)
        self.logger = logging.getLogger("ScraperLogger")

    def extract_html(self, url: str, sleep_time: int = 5,
                     headers: json = None) -> BeautifulSoup:
        """
        :param url:        String with an URL.
        :param sleep_time: Time to wait (in seconds) until next try
                           when connection error occurs.
        :param headers:
        :return:           BeautifulSoup object with HTML found on the page.
        """
        try:
            request = get(url, headers=headers)
        except requests.exceptions.ConnectionError:
            self.logger.warning(f"Failed to connect, sleeping for "
                                f"{sleep_time} seconds")
            sleep(sleep_time)
            return self.extract_html(url, sleep_time)
        else:
            return BeautifulSoup(request.text, features="html.parser")

    def get_meme_caption(self, meme_id: str) -> str:
        """
        :param meme_id: Meme's ID.
        :return:        Caption used on this meme.
        """
        soup = self.extract_html(f"{self.base_url}/i/{meme_id}")
        description = soup.find("div", class_="img-desc")
        if description is not None:
            meme_text = description.text.strip() \
                .replace("IMAGE DESCRIPTION:", "") \
                .replace("\n", " ")
            return meme_text

    def get_top_meme_ids(self, category: str, driver: webdriver,
                         num_pages: int = 1) -> List[str]:
        """
        :param category:      Category to be extracted,
        :param driver:        Selenium browser driver.
        :param num_pages:     Number of pages to download.
        :return:              List of memes' IDs.
        """
        meme_ids = []
        for page in range(1, num_pages + 1):
            self.logger.info(
                f"Downloading page {page} for category {category}")

            category_url = f"{self.base_url}/meme/" \
                           f"{category}?sort=top-365d&page={page}"
            driver.get(category_url)

            meme_links = driver.find_elements_by_class_name("base-img-link")
            for meme_link in meme_links:
                meme_id = meme_link.get_attribute("href").split("/")[-1]

                if meme_id not in meme_ids:
                    meme_ids.append(meme_id)

        self.logger.info(f"Found {len(meme_ids)} memes in {category}")
        return meme_ids

    def save_caption(self, category: str, meme_id: str, caption: str) -> None:
        category_file = f"{self.saving_dir}/{category}.json"
        meme_details = {"id": meme_id, "caption": caption}

        with open(category_file, "a+") as captions_file:
            captions_file.write(json.dumps(meme_details) + "\n")

    def save_category(self, category: str, driver: webdriver,
                      num_pages: int = 1) -> None:
        category_ids = set(self.get_top_meme_ids(category, driver, num_pages))
        cardinality = len(category_ids)

        for i, meme_id in enumerate(category_ids):
            self.logger.info(f"Saving meme to {category}: {meme_id} "
                             f"({i}/{cardinality})")
            caption = self.get_meme_caption(meme_id)
            self.save_caption(category, meme_id, caption)

    def save_all_categories(self, num_pages: int = 1) -> None:
        for category in self.categories:
            driver = webdriver.Firefox()
            self.logger.info(f"Downloading category {category}")
            self.save_category(category, driver, num_pages)
            driver.close()


if __name__ == '__main__':
    with open(CATEGORIES_FILE, "r") as f:
        meme_categories = [line.replace("\n", "") for line in f.readlines()]

    scraper = Scraper(BASE_URL, meme_categories)
    scraper.save_all_categories(num_pages=100)
