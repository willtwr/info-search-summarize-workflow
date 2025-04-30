import os
from agents.base_agent import BaseAgent
from langchain_core.messages import AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import MessagesState


class InvoiceDataExtractorAgent(BaseAgent):
    """Invoice Data Extractor agent for extracting data from invoice converted from invoice via OCR.

    This agent takes the OCR-ed invoice text and extract the relevant data into JSON format. It only
    extract the following fields:
    - total amount
    - bank account number
    - bank name
    """

    def load_system_prompt(self) -> None:
        """Load the invoice data extractor system prompt from file.

        The prompt template is loaded from invoice_data_extractor_system_prompt.txt in the same
        directory.
        """
        if self.sysprompt_path is None:
            self.sysprompt_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "invoice_data_extractor_system_prompt.txt"
            )

        with open(self.sysprompt_path, "r") as f:
            self.sys_prompt = PromptTemplate(
                input_variables=['context'], 
                template=f.read()
            )

    def invoke(self, state: MessagesState) -> dict:
        """Process the current state and extract invoice data.

        Takes the OCR-ed invoice text and extract data.

        Args:
            state: Current conversation state (append the OCR-ed invoice text to the end)

        Returns:
            dict: Contains the extracted data in JSON format
        """
        messages = state["messages"]
        docs = messages[-1]["content"]

        # Chain
        summarize_chain = self.sys_prompt | self.model | StrOutputParser()

        # Run
        response = summarize_chain.invoke({"context": docs})
        return {"messages": [AIMessage(content=response)]}
