# Info Search & Summarize Workflow (work in progress)
Sample workflow for information searching and summarization.

## Workflow Graph
![](./assets/workflow-graph.png)

- websearcher: generate query from user prompt and call the suitable tools to look for information.
- tools:
  - newssearch: search news based on query.
- summarizer: summarize the found contents into 100-250 words.

## Installation (WIP)
- Install chromedriver from [here](https://googlechromelabs.github.io/chrome-for-testing/)
  - Linux: copy chromedriver to /usr/bin/
