import os
import json
import uuid
from agents.base_agent import BaseAgent
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langgraph.graph import MessagesState


class WebSearcherAgent(BaseAgent):
    """WebSearcher Agent"""
    def load_system_prompt(self) -> None:
        if self.sysprompt_path is None:
            self.sysprompt_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "websearcher_system_prompt.txt"
            )

        with open(self.sysprompt_path, "r") as file:
            self.sys_prompt = file.read()

    def bind_tools(self, tools: list) -> None:
        """Add tools to system prompt"""
        functions = [
            {
                "function_name": tool.name,
                "description": tool.description,
                "parameters": {
                    "properties": tool.args,
                    "required": list(tool.args.keys())
                }
            }
            for tool in tools
        ]

        tools_str = "[" + ','.join([json.dumps(func) for func in functions]) + "]"
        self.sys_prompt = self.sys_prompt.replace("{tools}", tools_str)
        print(self.sys_prompt)
        
    def invoke(self, state: MessagesState) -> dict:
        messages = [SystemMessage(self.sys_prompt)] + [msg for msg in state["messages"] if isinstance(msg, (HumanMessage, AIMessage))]
        output = {"messages": [self.model.invoke(messages)]}
        if output["messages"][-1].content.startswith('<tool_call>'):
            contents = output["messages"][-1].content.replace('<tool_call>', '').replace('</tool_call>', '').strip()
            contents = json.loads(contents)
            output = {"messages": [AIMessage(content="", tool_calls=[{"name": content["name"], "args": content["arguments"], "type": "tool_call", "id": str(uuid.uuid4())} for content in contents])]}

        return output
