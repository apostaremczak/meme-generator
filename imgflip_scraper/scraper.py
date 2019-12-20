import requests
from bs4 import BeautifulSoup
from requests import get
from time import sleep
from typing import List

BASE_URL = "https://imgflip.com"


def extract_html(url: str, sleep_time: int = 5) -> BeautifulSoup:
    """
    :param url:        String with an URL.
    :param sleep_time: Time to wait (in seconds) until next try when connection
                       error occurs.
    :return:           BeautifulSoup object with HTML found on the page.
    """
    try:
        request = get(url)
    except requests.exceptions.ConnectionError:
        print(f"Failed to connect, sleeping for {sleep_time} seconds")
        sleep(sleep_time)
        return extract_html(url)
    else:
        return BeautifulSoup(request.text, features="html.parser")


def get_meme_caption(meme_id: str) -> str:
    """
    :param meme_id: Meme's ID.
    :return:        Caption used on this meme.
    """
    soup = extract_html(f"{BASE_URL}/i/{meme_id}")
    description = soup.find("div", class_="img-desc").text
    meme_text = description.strip() \
        .replace("IMAGE DESCRIPTION:", "") \
        .replace("\n", " ")
    return meme_text


def get_top_meme_ids(category_name: str, num_pages: int = 1) -> List[str]:
    """
    :param category_name: Category to be extracted,
    :param num_pages:     Number of pages to download.
    :return:              List of memes' IDs.
    """
    meme_ids = []
    for i in range(1, num_pages + 1):
        category_url = BASE_URL + "/meme/" + category_name \
                       + "?sort=top-365d&page=" + str(i)
        soup = extract_html(category_url)

        for link in soup.find_all("a"):
            if link.attrs.get("class") == ['base-img-link']:
                meme_ids.append(link.get("href").replace("/i/", ""))

    return meme_ids
