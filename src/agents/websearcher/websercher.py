import os
import json
import uuid
from typing import Optional
from models.llm_factory import llm_factory
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_huggingface import ChatHuggingFace
from langgraph.graph import MessagesState


class WebSearcherAgent:
    """WebSearcher Agent"""
    def __init__(
            self, 
            model_name: str = "qwen",
            model: Optional[BaseChatModel] = None,
            sys_prompt_path: Optional[str] = None
    ):
        # Initialize model
        self.model_name = model_name
        if model is None:
            self.build_model()
        else:
            self.model = model

        # Load system prompt
        if sys_prompt_path is None:
            sys_prompt_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "websearcher_system_prompt.txt"
            )

        with open(sys_prompt_path, "r") as file:
            self.sys_prompt = file.read()

    def build_model(self) -> None:
        """Build the LLM model"""
        llm = llm_factory(self.model_name)
        if not isinstance(llm, BaseChatModel):
            self.model = ChatHuggingFace(llm=llm)
        else:
            self.model = llm

    def bind_tools(self, tools: list) -> None:
        """Bind tools for the agent to use"""
        functions = [
            {
                "type": "function",
                "function": {
                    "name": "news_search",
                    "description": "Search news based on query",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"}
                        },
                        "required": ["query"]
                    }
                }
            }
        ]
        tools_str = "[" + ',\n'.join([json.dumps(func) for func in functions]) + "]"
        print(tools_str)
        self.sys_prompt = self.sys_prompt.replace("{tools}", tools_str)
        print(self.sys_prompt)

    def invoke(self, state: MessagesState) -> dict:
        messages = [SystemMessage(self.sys_prompt)] + [msg for msg in state["messages"] if isinstance(msg, (HumanMessage, AIMessage))]
        output = {"messages": [self.model.invoke(messages)]}
        print(output)
        if output["messages"][-1].content.startswith('<tool_call>'):
            contents = output["messages"][-1].content.replace('<tool_call>\n', '').replace('\n</tool_call>', '')
            contents = json.loads(contents)
            output = {"messages": [AIMessage(content="", tool_calls=[{"name": content["name"], "args": content["arguments"], "type": "tool_call", "id": str(uuid.uuid4())} for content in contents])]}

        return output
    
    def __call__(self, state: MessagesState) -> dict:
        return self.invoke(state)
