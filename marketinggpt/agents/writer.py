import os
from typing import List, Dict, Any
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Tải các biến môi trường
dotenv_path = r"D:\gpt-marketer\marketinggpt\.env"
load_dotenv(dotenv_path)

# Set environment variables
os.environ['PINECONE_API_KEY'] = os.getenv('PINECONE_API_KEY')
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_SMITH_API_KEY")
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')


prompt_template = """
Based on the search result provided below, write a detailed and informative paragraph:

Search Result: {search_result}
"""

# Định nghĩa lớp Writer
class Writer:
    def __init__(self, llm, verbose=False, **kwargs):
        self.llm = llm
        self.verbose = verbose
        self.prompt = PromptTemplate(template=prompt_template, input_variables=["search_result"])
        self.chain = self.prompt | llm

    @property
    def output_keys(self) -> List[str]:
        return ["written_output"]

    def _call(self, search_result: str) -> str:
        inputs = {
            "search_result": search_result,
        }
        response = self.chain.invoke(inputs)
        return response


def writer_agent_node(state: Dict[str, Any]) -> Dict[str, str]:

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.1)   
    writer_agent = Writer(llm, prompt_template, verbose=True)

    result = writer_agent._call(state["search_result"])
    return {"written_output": result}

