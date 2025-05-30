import re
from duckduckgo_search import DDGS
from selenium import webdriver
from bs4 import BeautifulSoup
from langchain.tools import tool


@tool("web_search", return_direct=False)
def web_search(query: str) -> str:
    """Web search tool for retrieving information from websites.
    
    This tool uses DuckDuckGo to search the web and retrieves content from matching
    pages using Selenium. It processes the content to extract meaningful text while
    filtering out short or irrelevant sections.
    """
    with DDGS(timeout=20) as ddgs:
        results = ddgs.text(query, max_results=10)

    output = []
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    driver = webdriver.Chrome(options=options)
    for result in results:
        driver.get(result['href'])
        response = driver.page_source
        soup = BeautifulSoup(response, 'html.parser')
        content = "\n\n".join([x for x in soup.get_text().strip().splitlines() if bool(x)])
        if len(re.findall(r'\b\w+\b', content)) > 20:
            output.append("#" + result['title'] + "\n\n" + content)

    driver.quit()
    output = '\n\n'.join(output)
    return output
