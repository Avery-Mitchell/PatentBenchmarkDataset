from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from webdriver_manager.chrome import ChromeDriverManager
import time
import requests
import os
from bs4 import BeautifulSoup
from typing import Optional
import re
import json

def get_articles_from_page(search_term: str, page_num: int) -> list[tuple[str, str]]:
    """
    Uses Selenium to scrape article titles and links from a single Snopes search results page.

    Arguments:
        search_term: the term used in the website's search bar
        page_num: the page number of the search results

    Returns:
        list of (title, link) tuples
    """

    url = f"https://www.snopes.com/search/?q={search_term}#gsc.tab=0&gsc.q={search_term}&gsc.page={page_num}"

    # Headless Chrome
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--window-size=1920x1080")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.get(url)

    # Allows JavaScript to load
    time.sleep(5)  

    try:
        WebDriverWait(driver, 20).until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, "#search-results a.gs-title"))
        )
    except TimeoutException:
        print(f"[Page {page_num}] Timed out waiting for search results.")
        driver.quit()
        return []

    links = driver.find_elements(By.CSS_SELECTOR, "#search-results a.gs-title")

    articles: list[tuple[str, str]] = []
    for link in links:
        title = link.text.strip()
        href = link.get_attribute("href")
        if title and href:
            articles.append((title, href))

    driver.quit()
    return articles

def download_articles(articles: list[tuple[str, str]]) -> None:
    """
    Downloads all the snopes articles' html to snopes_raw folder

    Arguments:
        articles: list of tuples (title, url) from get_articles_from_page

    Returns:
        None
    """
    
    os.makedirs("snopes_raw", exist_ok=True)

    for title, url in articles:
        try:
            new_title = url.rstrip('/').split('/')[-1]
            filename = os.path.join("snopes_raw", f"{new_title}.txt")
            
            response = requests.get(url)
            response.raise_for_status()

            with open(filename, 'w', encoding='utf-8') as f:
                f.write(response.text)

            print(f"Saved {url} as {filename}")
        except Exception as e:
            print(f"Failed to download {url}: {e}")

def extract_article_text(html_file_path: str) -> Optional[str]:
    """
    Extracts text from raw html

    Arguments:
        html_file_path: path to raw html file

    Returns:
        text from raw html
    """
    with open(html_file_path, 'r', encoding='utf-8') as f:
        html = f.read()
    soup = BeautifulSoup(html, 'html.parser')

    article = soup.find('article')
    if not article:
        article = soup.find('div', class_='single-body-card')

    if not article:
        article = soup

    script_tags = soup.find_all("script")
    author = "Unknown Author"
    for script in script_tags:
        if script.string and "snopes_author_1" in script.string:
            match = re.search(r'"snopes_author_1"\s*:\s*"([^"]+)"', script.string)
            if match:
                author = match.group(1)
                break

    date_tag = soup.select_one(".publish_date")
    date = date_tag.get_text(strip=True) if date_tag else "Unknown Date"

    claim = "N/A"
    rating = "N/A"
    for script in soup.find_all("script", {"type": "application/ld+json"}):
        try:
            data = json.loads(script.string)
            if isinstance(data, dict) and data.get("@type") == "ClaimReview":
                claim = data.get("claimReviewed", "N/A")
                rating = data.get("reviewRating", {}).get("alternateName", "N/A")
                break
        except (json.JSONDecodeError, TypeError):
            continue

    content_blocks = article.find_all(['p', 'h2', 'h3'])

    clean_paragraphs = []
    for block in content_blocks:
        text = block.get_text(separator=' ', strip=True) 
        if text:
            clean_paragraphs.append(text)

    body_text = '\n\n'.join(clean_paragraphs)
    if body_text:
        full_text = (
            f"Author: {author}\n"
            f"Date: {date}\n"
            f"Claim: {claim}\n"
            f"Rating: {rating}\n\n"
            f"{body_text}"
        )
        return full_text
    else:
        return None
    
def clean_all_articles() -> None:
    """
    Cleans all the articles in snopes_raw and saves them to snopes_cleaned

    Arguments:
        None
    Returns:
        None
    """
    os.makedirs("snopes_cleaned", exist_ok=True)

    for filename in os.listdir("snopes_raw"):
        if not filename.endswith(".txt"):
            continue 

        input_path = os.path.join("snopes_raw", filename)
        output_path = os.path.join("snopes_cleaned", filename)

        article_text = extract_article_text(input_path)

        if article_text:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(article_text)
            print(f"Cleaned: {filename}")
        else:
            print(f"Skipped: {filename}")

if __name__ == "__main__":
    """
    search_term = "patents"
    results = get_articles_from_page("patents", 1)
    for x in range(1, 7):
        articles = get_articles_from_page("patents", x)
        download_articles(articles)

    print(f"\nTotal articles found: {len(results)}\n")
    for i, (title, link) in enumerate(results, start=1):
        print(f"{i}. {title}\n   {link}\n")
    """
    clean_all_articles()
