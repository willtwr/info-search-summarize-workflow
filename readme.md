# Information Searching & Summarization Workflow

A comprehensive system for information retrieval and summarization that combines web search,
document search, and intelligent summarization capabilities.

## Features

### Search Capabilities
- **Web Search**: General web information retrieval using DuckDuckGo
- **News Search**: Current events and news article search
- **Document Search**: RAG-based local document search with vector store backend
- **File Upload**: Support for reading and indexing PDF documents

### Key Components
- **Smart Search Routing**: Automatically selects the most appropriate search tool
- **Vector Store Integration**: Efficient document indexing and similarity search
- **Intelligent Summarization**: Context-aware, focused summaries (100-250 words)
- **Streaming Updates**: Real-time response streaming in the UI

## Architecture

The system follows a modular architecture with several key components:

### Agents
- **WebSearcher**: Interprets queries and coordinates search tool selection
- **Summarizer**: Processes search results into coherent summaries

### Models
- **Language Models**:
  - Qwen 2.5 (3B params) with AWQ quantization
  - SmolLM2 (1.7B params) for efficient processing
- **Embedding Model**:
  - Stella EN 1.5B v5 for document vectorization

### Tools
- Web and news search via DuckDuckGo
- Vector store retrieval for document search
- PDF processing and indexing

## Workflow Graph
![](./assets/workflow-graph.png)

The workflow operates as follows:
1. WebSearcher agent processes the user query
2. Appropriate tools are selected and executed
3. Results are passed to the Summarizer agent
4. Final summary is presented to the user

## Installation

### Prerequisites
- Python 3.11 or higher
- CUDA-capable GPU for model inference

### Setup
1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Install Chrome and ChromeDriver:
   - Linux: Download ChromeDriver from [Google Chrome for Testing](https://googlechromelabs.github.io/chrome-for-testing/)
   - Copy chromedriver to /usr/bin/

### Environment Setup
- Ensure CUDA is properly configured for GPU acceleration
- At least 12GB GPU memory recommended
- SSD storage recommended for vector database

## Usage

1. Start the application:
   ```bash
   python src/app.py
   ```
2. Access the web interface (default: http://localhost:7860)
3. Upload documents or start chatting to search for information

## Evaluation

LLM generated content can be evaluated using:
- **ROUGE scores** for summarization quality
- **BLEU scores** for translation accuracy (Reminder: this repo does not have translation feature)