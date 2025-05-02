import os
import json
import uuid
from utils import remove_think
from agents.base_agent import BaseAgent
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langgraph.graph import MessagesState


class WebSearcherAgent(BaseAgent):
    """Web Search Agent that coordinates web-based information retrieval.

    This agent is responsible for interpreting user queries and coordinating various web search
    tools to find relevant information. It can use tools like web search, news search, and
    vector store retrieval to gather information.

    The agent uses a system prompt (loaded from a file) that guides its behavior and decision
    making process for tool selection and query formulation.
    """

    def load_system_prompt(self) -> None:
        """Load the websearcher system prompt from file.

        The prompt is loaded from a text file in the same directory as the agent.
        If no specific path is provided, uses the default websearcher_system_prompt.txt.
        """
        if self.sysprompt_path is None:
            self.sysprompt_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "websearcher_system_prompt.txt"
            )

        with open(self.sysprompt_path, "r") as file:
            self.sys_prompt = file.read()

    def bind_tools(self, tools: list) -> None:
        """Configure the available tools in the system prompt.

        Processes the provided tools list and updates the system prompt with their
        descriptions and parameters, allowing the agent to understand how to use them.

        Args:
            tools (list): List of tool objects that implement the LangChain tool interface
        """
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
        """Process the current conversation state and determine next actions.

        Takes the current message state, prepends the system prompt, and generates
        either a direct response or tool calls to gather information.

        Args:
            state (MessagesState): Current conversation state with message history

        Returns:
            dict: Contains either new messages or tool calls to be executed
        """
        messages = [SystemMessage(self.sys_prompt)] + [msg for msg in state["messages"] if isinstance(msg, (HumanMessage, AIMessage))]
        output = {"messages": [self.model.invoke(messages)]}
        contents = output["messages"][-1].content
        contents = remove_think(contents)
        if contents.startswith('<tool_call>'):
            contents = contents.replace('<tool_call>', '').replace('</tool_call>', '').strip()
            contents = json.loads(contents)
            if not isinstance(contents, list):
                contents = [contents]
            
            output = {"messages": [AIMessage(content="", tool_calls=[{"name": content["name"], "args": content["arguments"], "type": "tool_call", "id": str(uuid.uuid4())} for content in contents])]}

        return output
