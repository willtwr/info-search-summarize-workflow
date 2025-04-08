# Infomation Searching & Summarization Workflow (work in progress)
Sample workflow for information searching and summarization.

## Features
- Web search: search the web for information.
- New search: search the web for news.
- Document search: search content from documents via RAG. 

## Workflow Graph
![](./assets/workflow-graph.png)

- websearcher: generate query from user prompt and call the suitable tools to look for information.
- tools:
  - newssearch: search news based on query.
  - websearch: search websites for information based on query.
  - vector_store_retriever: content searching from documents stored in vector store via RAG.
- summarizer: summarize the found contents into 100-250 words.

## Installation (WIP)
- Install chromedriver from [here](https://googlechromelabs.github.io/chrome-for-testing/)
  - Linux: copy chromedriver to /usr/bin/
