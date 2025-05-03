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
            slug = url.rstrip('/').split('/')[-1]
            filename = f"{slug}.txt"
            path = os.path.join("snopes_raw", filename)

            resp = requests.get(url)
            resp.raise_for_status()

            # ← PREPEND the URL as an HTML comment
            with open(path, 'w', encoding='utf-8') as f:
                f.write(f"<!-- snopes_url: {url} -->\n")
                f.write(resp.text)

            print(f"Saved {url} → {path}")
        except Exception as e:
            print(f"Error downloading {url}: {e}")

def extract_article_text(html: str) -> Optional[str]:
    """
    Extracts text from raw html

    Arguments:
        html: path to raw html file

    Returns:
        text from raw html
    """
    # 1) get URL from our prepended comment
    url = "Unknown URL"
    m = re.match(r'<!--\s*snopes_url:\s*(.*?)\s*-->', html)
    if m:
        url = m.group(1)

    soup = BeautifulSoup(html, 'html.parser')

    # 2) author
    author = "Unknown Author"
    for script in soup.find_all("script"):
        if script.string and "snopes_author_1" in script.string:
            match = re.search(r'"snopes_author_1"\s*:\s*"([^"]+)"', script.string)
            if match:
                author = match.group(1)
                break

    # 3) title & date
    title = soup.find("meta", property="og:title")["content"] if soup.find("meta", property="og:title") else "Unknown Title"
    date_tag = soup.select_one(".publish_date")
    date = date_tag.get_text(strip=True) if date_tag else "Unknown Date"

    # 4) claim & rating
    claim, rating = "N/A", "N/A"
    for script in soup.find_all("script", {"type": "application/ld+json"}):
        try:
            data = json.loads(script.string)
            if data.get("@type") == "ClaimReview":
                claim = data.get("claimReviewed", "N/A")
                rating = data.get("reviewRating", {}).get("alternateName", "N/A")
                break
        except (ValueError, TypeError):
            continue

    # 5) body text
    blocks = soup.find('article') or soup.find('div', class_='single-body-card') or soup
    paras = [p.get_text(" ", strip=True) for p in blocks.find_all(['p','h2','h3']) if p.get_text(strip=True)]
    body = "\n\n".join(paras)

    return title, author, date, claim, rating, body, url
    
def clean_all_articles() -> None:
    """
    Cleans all the articles in snopes_raw and saves them to snopes_cleaned

    Arguments:
        None
    Returns:
        None
    """
    os.makedirs("snopes_cleaned", exist_ok=True)

    for fn in os.listdir("snopes_raw"):
        if not fn.endswith(".txt"):
            continue

        in_path = os.path.join("snopes_raw", fn)
        out_path = os.path.join("snopes_cleaned", fn)

        with open(in_path, 'r', encoding='utf-8') as f:
            raw = f.read()

        title, author, date, claim, rating, body, url = extract_article_text(raw)

        if not body:
            print(f"Skipped {fn} (no body)")
            continue

        cleaned = (
            f"Article Title: {title}\n"
            f"Author: {author}\n"
            f"Date Published: {date}\n"
            f"Claim: {claim}\n"
            f"Rating: {rating}\n"
            f"URL: {url}\n\n"
            f"Article Text:\n{body}"
        )

        with open(out_path, 'w', encoding='utf-8') as outf:
            outf.write(cleaned)
        print(f"Cleaned → {out_path}")

if __name__ == "__main__":
    SEARCH_TERM = "patents"
    NUM_PAGES   = 6

    # 1) Collect all (title, url) tuples across pages 0–5
    all_articles: list[tuple[str,str]] = []
    for p in range(NUM_PAGES):
        print(f"Fetching page {p+1}/{NUM_PAGES} for “{SEARCH_TERM}”…")
        page_articles = get_articles_from_page(SEARCH_TERM, p)
        print(f"  → found {len(page_articles)} articles")
        all_articles.extend(page_articles)

    # 2) Download them into snopes_raw/ (with our URL comment)
    print(f"\nDownloading {len(all_articles)} total articles…")
    download_articles(all_articles)

    # 3) Clean them all out into snopes_cleaned/
    print("\nCleaning all downloaded articles…")
    clean_all_articles()

    print("\nDone!  Raw HTMLs are in snopes_raw/, cleaned text in snopes_cleaned/")
