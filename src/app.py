import uuid
import gradio as gr
from graph import WorkflowGraph
from vectordb.chroma import ChromaVectorStore


config = {"configurable": {"thread_id": uuid.uuid4()}}

# Initialize models here so that they are not loaded more than once.
if gr.NO_RELOAD:
    # Load the vector database
    vectordb = ChromaVectorStore()

    # Load the workflow graph
    workflow = WorkflowGraph(model_name="qwen", vectorstore=vectordb.get_retriever())


def stream_chat_graph_updates(chat_history: list, markdown_box: str):
    """Update assistant chat here"""
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
    """Update user chat here and clear textbox"""
    chat_history.append({"role": "user", "content": message})
    return "", chat_history


def upload_document(uploaded_file: gr.UploadButton, progress=gr.Progress()):
    """Upload documents to vector database"""
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
