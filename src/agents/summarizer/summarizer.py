import os
from typing import Optional
from models.llm_factory import llm_factory
from langchain_huggingface import ChatHuggingFace
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import MessagesState


class SummarizerAgent:
    """Summarizer Agent
    To summarize the retrieved contents.
    """
    def __init__(
            self,
            model_name: str = "qwen",
            model: Optional[BaseChatModel] = None,
            sysprompt_path: Optional[str] = None
    ):
        # Model initialization
        self.model_name = model_name
        if model is None:
            self.build_model()
        else:
            self.model = model

        # System prompt loading
        if sysprompt_path is None:
            sysprompt_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "summarizer_system_prompt.txt"
            )

        with open(sysprompt_path, "r") as f:
            self.prompt = ChatPromptTemplate(
                input_variables=['context', 'question'], 
                messages=[
                    HumanMessagePromptTemplate(
                        prompt=PromptTemplate(
                            input_variables=['context', 'question'], 
                            template=f.read()
                        )
                    )
                ]
            )

    def build_model(self) -> None:
        """Build the LLM model"""
        llm = llm_factory(self.model_name)
        if not isinstance(llm, BaseChatModel):
            self.model = ChatHuggingFace(llm=llm)
        else:
            self.model = llm

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
        summarize_chain = self.prompt | self.model | StrOutputParser()

        # Run
        response = summarize_chain.invoke({"context": docs, "question": question})
        return {"messages": [AIMessage(content=response)]}
    
    def __call__(self, state: MessagesState) -> dict:
        return self.invoke(state)
