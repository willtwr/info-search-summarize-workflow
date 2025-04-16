import os
from agents.base_agent import BaseAgent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import MessagesState


class SummarizerAgent(BaseAgent):
    """Summarization agent for processing and condensing retrieved information.

    This agent takes search results or other retrieved content and generates concise,
    focused summaries. It uses a template-based system prompt to ensure consistent
    summarization behavior, with constraints on summary length and relevance.

    The agent is designed to:
    - Maintain focus on the original question
    - Produce summaries between 100-250 words
    - Only use information from the provided context
    - Acknowledge when information is insufficient
    """

    def load_system_prompt(self) -> None:
        """Load the summarizer system prompt from file.

        The prompt template is loaded from summarizer_system_prompt.txt in the same
        directory. It includes placeholders for the question and context to be
        summarized.
        """
        if self.sysprompt_path is None:
            self.sysprompt_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "summarizer_system_prompt.txt"
            )

        with open(self.sysprompt_path, "r") as f:
            self.sys_prompt = PromptTemplate(
                input_variables=['context', 'question'], 
                template=f.read()
            )

    def invoke(self, state: MessagesState) -> dict:
        """Process the current state and generate a summary.

        Takes the most recent question and retrieved content from the message history,
        applies the summarization prompt template, and generates a concise summary.

        Args:
            state: Current conversation state containing the question and retrieved content

        Returns:
            dict: Contains the generated summary as a new message
        """
        messages = state["messages"]
        for item in reversed(messages):
            if isinstance(item, HumanMessage):
                question = item.content
                break
        
        docs = messages[-1].content
        
        # Chain
        summarize_chain = self.sys_prompt | self.model | StrOutputParser()

        # Run
        response = summarize_chain.invoke({"context": docs, "question": question})
        return {"messages": [AIMessage(content=response)]}
