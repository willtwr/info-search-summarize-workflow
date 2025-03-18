import uuid
import gradio as gr
from graph import WorkflowGraph


config = {"configurable": {"thread_id": uuid.uuid4()}}

# Initialize models here so that they are not loaded more than once.
if gr.NO_RELOAD:
    workflow = WorkflowGraph(model_name="qwen")


def stream_chat_graph_updates(chat_history: list):
    """Update assistant chat here"""
    for event in workflow().stream({"messages": [("user", chat_history[-1]["content"])]}, config, stream_mode="updates"):
        print("-----------event----------------")
        print(event)

        if "tools" in event:
            message = event['tools']['messages'][-1]
            chat_history.append({"role": "assistant", "content": message.content, "metadata": {"title": f"🛠️ Used tool {message.name}"}})
        else:
            message = event[list(event.keys())[0]]['messages'][-1]
            chat_history.append({"role": "assistant", "content": message.content})

        print("--------Print From Stream-----------")
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()
        
        yield chat_history


def stream_user_message(message: str, chat_history: list):
    """Update user chat here and clear textbox"""
    chat_history.append({"role": "user", "content": message})
    return "", chat_history


with gr.Blocks() as demo:
    gr.Label("Bot")
    chat = gr.Chatbot(type="messages")
    msg = gr.Textbox(placeholder="Type your message here...", submit_btn=True)
    msg.submit(stream_user_message, [msg, chat], [msg, chat], queue=False).then(stream_chat_graph_updates, chat, chat)


if __name__ == "__main__":
    demo.launch()
