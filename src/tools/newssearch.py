from selenium import webdriver
from duckduckgo_search import DDGS
from bs4 import BeautifulSoup
from langchain.tools import tool


@tool("news_search", return_direct=False)
def news_search(query: str) -> str:
    """Searches news from the internet based on query."""
    with DDGS() as ddgs:
        results = ddgs.news(query, max_results=5)

    output = []
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    driver = webdriver.Chrome(options=options)
    for result in results:
        driver.get(result['url'])
        response = driver.page_source
        print(response)
        soup = BeautifulSoup(response, 'html.parser')
        output.append("#" + result['title'] + "\n\n" + "\n\n".join([x for x in soup.get_text().strip().splitlines() if bool(x)]))

    driver.quit()

    output = '\n\n'.join(output)
    return output
