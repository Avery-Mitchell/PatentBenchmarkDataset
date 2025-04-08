from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from webdriver_manager.chrome import ChromeDriverManager
import time

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

# May scrap this and manually input page numbers 
def scrape_all_articles(search_term: str) -> list[tuple[str, str]]:
    """
    Scrapes all Snopes search results for a given term across all pages.

    Arguments:
        search_term: the term used in the website's search bar
        page_num: the page number of the search results

    Returns:
        A combined list of (title, link) tuples from all pages.
    """
    all_articles: list[tuple[str, str]] = []
    page = 1

    while True:
        print(f"Scraping page {page}...")
        articles = get_articles_from_page(search_term, page)

        if not articles:
            print("No more articles found. Stopping.")
            break

        all_articles.extend(articles)
        page += 1
        time.sleep(1) 

    return all_articles


if __name__ == "__main__":
    search_term = "patents"
    results = scrape_all_articles(search_term)

    print(f"\n✅ Total articles found: {len(results)}\n")
    for i, (title, link) in enumerate(results, start=1):
        print(f"{i}. {title}\n   {link}\n")
