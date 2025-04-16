from typing import Union, Any, Literal
from pydantic import BaseModel
from langchain_core.messages import AnyMessage


def tools_condition(
    state: Union[list[AnyMessage], dict[str, Any], BaseModel],
    messages_key: str = "messages"
) -> Literal["tools", "__end__"]:
    """Determine if tool execution is needed based on the current state.

    This function analyzes the current workflow state to decide whether to proceed
    with tool execution or end the workflow. It specifically looks for tool calls
    in the most recent AI message.

    The function handles different state formats:
    - List of messages directly
    - Dictionary with messages under a key
    - BaseModel with messages as an attribute

    Args:
        state: The current workflow state containing messages
        messages_key: Key to access messages in dict/model states. Defaults to "messages"

    Returns:
        "tools" if tool execution is needed, "__end__" otherwise

    Raises:
        ValueError: If no messages can be found in the provided state
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif isinstance(state, dict) and (messages := state.get(messages_key, [])):
        ai_message = messages[-1]
    elif messages := getattr(state, messages_key, []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    
    return "__end__"
