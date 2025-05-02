import re
from duckduckgo_search import DDGS
from selenium import webdriver
from bs4 import BeautifulSoup
from langchain.tools import tool


@tool("news_search", return_direct=False)
def news_search(query: str) -> str:
    """News-specific search tool for retrieving current information.
    
    This tool uses DuckDuckGo's news search to find recent articles and news content.
    It processes each result using Selenium to extract the full article text while
    filtering out short or irrelevant content.
    """
    with DDGS(timeout=20) as ddgs:
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
