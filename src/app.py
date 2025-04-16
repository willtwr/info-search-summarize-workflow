"""Main application module providing the Gradio web interface.

This module implements the web interface for the information search and summarization
workflow using Gradio. It provides a chat interface with document upload capabilities
and real-time interaction with the workflow system.

The interface allows users to:
- Chat with the system to search for information
- Upload PDF documents for vector store indexing
- View search results and summaries in a split-panel interface
"""

import uuid
import gradio as gr
from graph import WorkflowGraph
from vectordb.chroma import ChromaVectorStore


config = {
    "configurable": {
        "thread_id": uuid.uuid4()  # Unique identifier for each chat session
    }
}

# Initialize models here so that they are not loaded more than once.
if gr.NO_RELOAD:
    # Load the vector database
    vectordb = ChromaVectorStore()

    # Load the workflow graph with Qwen model and vector store retriever
    workflow = WorkflowGraph(model_name="qwen", vectorstore=vectordb.get_retriever())


def stream_chat_graph_updates(chat_history: list, markdown_box: str):
    """Update the chat interface with streaming responses from the workflow.

    This function processes workflow updates in real-time, showing both the chatbot's
    responses and any intermediate tool outputs.

    Args:
        chat_history: List of message dictionaries representing the chat history
        markdown_box: Current content of the markdown display box

    Yields:
        Tuple of updated chat_history and markdown_box content
    """
    for event in workflow().stream({"messages": [("user", chat_history[-1]["content"])]}, config, stream_mode="updates"):
        print("-----------event----------------")
        print(event)

        if "tools" in event:
            message = event['tools']['messages'][-1]
            markdown_box = message.content
        else:
            message = event[list(event.keys())[0]]['messages'][-1]
            chat_history.append({"role": "assistant", "content": message.content})

        print("--------Print From Stream-----------")
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()
        
        yield chat_history, markdown_box


def stream_user_message(message: str, chat_history: list):
    """Add a user message to the chat history.

    Args:
        message: The user's input message
        chat_history: Current chat history

    Returns:
        Tuple of empty string (to clear input) and updated chat history
    """
    chat_history.append({"role": "user", "content": message})
    return "", chat_history


def upload_document(uploaded_file: gr.UploadButton, progress=gr.Progress()):
    """Process and index an uploaded PDF document.

    Reads the PDF file, splits it into chunks, and adds it to the vector store
    for future retrieval.

    Args:
        uploaded_file: The uploaded PDF file information
        progress: Gradio progress indicator

    Returns:
        The uploaded file information for display
    """
    progress(0, desc="Reading document...")
    doc = vectordb.read_pdf(uploaded_file.name)
    progress(0.2, desc="Uploading document...")
    vectordb.add_documents(doc)
    progress(1, desc="Document uploaded successfully.")
    return uploaded_file


with gr.Blocks() as demo:
    with gr.Row():
        gr.Label("Bot")
    
    with gr.Row(equal_height=True):
        with gr.Column():
            chat = gr.Chatbot(type="messages")

            with gr.Row():
                with gr.Column(scale=1):
                    filebox = gr.File()
                    upload_button = gr.UploadButton("Upload a PDF file", file_count="single", size="sm")
                
                msg = gr.Textbox(placeholder="Type your message here...", submit_btn=True, lines=1, max_lines=2, scale=9)

        with gr.Column():
            md = gr.Markdown("Content here...", container=True, height="75vh", max_height="75vh")

    upload_button.upload(upload_document, [upload_button], [filebox], show_progress_on=filebox)
    msg.submit(stream_user_message, [msg, chat], [msg, chat], queue=False).then(stream_chat_graph_updates, [chat, md], [chat, md])


if __name__ == "__main__":
    demo.launch()
