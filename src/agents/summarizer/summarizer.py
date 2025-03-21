import os
from agents.base_agent import BaseAgent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import MessagesState


class SummarizerAgent(BaseAgent):
    """Summarizer Agent
    To summarize the retrieved contents.
    """
    def load_system_prompt(self) -> None:
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
        print("-----Summarize-----")
        messages = state["messages"]
        print(messages)
        
        for item in reversed(messages):
            if isinstance(item, HumanMessage):
                question = item.content
                break
        
        print("___Question___")
        print(question)

        last_message = messages[-1]
        docs = last_message.content
        print("___Contents___")
        print(docs)
        
        # Chain
        summarize_chain = self.sys_prompt | self.model | StrOutputParser()

        # Run
        response = summarize_chain.invoke({"context": docs, "question": question})
        return {"messages": [AIMessage(content=response)]}
