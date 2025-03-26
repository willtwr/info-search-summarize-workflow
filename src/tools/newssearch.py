import re
from duckduckgo_search import DDGS
from selenium import webdriver
from bs4 import BeautifulSoup
from langchain.tools import tool


@tool("news_search", return_direct=False)
def news_search(query: str) -> str:
    """Useful for searching news."""
    with DDGS() as ddgs:
        results = ddgs.news(query, max_results=10)

    output = []
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    driver = webdriver.Chrome(options=options)
    for result in results:
        driver.get(result['url'])
        response = driver.page_source
        soup = BeautifulSoup(response, 'html.parser')
        content = "\n\n".join([x for x in soup.get_text().strip().splitlines() if bool(x)])
        if len(re.findall(r'\b\w+\b', content)) > 20:
            output.append("#" + result['title'] + "\n\n" + content)

    driver.quit()
    output = '\n\n'.join(output)
    return output
