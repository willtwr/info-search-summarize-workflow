import requests
from duckduckgo_search import DDGS
from bs4 import BeautifulSoup
from langchain.tools import tool


@tool("news_search", return_direct=False)
def news_search(query: str) -> str:
    """Searches news from the internet based on query."""
    with DDGS() as ddgs:
        results = ddgs.news(query, max_results=5)

    output = []
    for result in results:
        try:
            response = requests.get(result['url'], timeout=5)
        except requests.exceptions.ReadTimeout:
            output.append("#" + result['title'])
            continue
        finally:
            soup = BeautifulSoup(response.content, 'html.parser')
            output.append("#" + result['title'] + "\n\n" + "\n\n".join([x for x in soup.get_text().strip().splitlines() if bool(x)]))

    output = '\n\n'.join(output)
    return output
