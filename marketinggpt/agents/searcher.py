import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain import hub
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

from typing import List, Dict, Any
from pydantic import BaseModel, EmailStr, Field
from typing import Optional
from datetime import date
from pydantic import BaseModel, validator
from datetime import datetime

from marketinggpt.tools.customer_search import customer_search_tool

dotenv_path = r"D:\gpt-marketer\marketinggpt\.env"
load_dotenv(dotenv_path)

# Set environment variables
os.environ['PINECONE_API_KEY'] = os.getenv('PINECONE_API_KEY')
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_SMITH_API_KEY")
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')

class Customer(BaseModel):
    customer_id: input
    name: str
    email: EmailStr
    phone_number: int
    gender: Optional[str] = None
    marital_status: Optional[str] = None
    income: Optional[int] = None
    age: int
    interests: Optional[str] = None
    purchase_history: Optional[str] = None


class Searcher:
    def __init__(self, llm, verbose=True, **kwargs):
        self.llm = llm
        self.verbose = verbose
        self._show_welcome_message()

    def _show_welcome_message(self):
        """Hiển thị thông báo chào mừng khi khởi tạo Searcher."""
        print("Xin chào, tôi là marketingGPT. Bạn hãy nhập thông tin khách hàng mà bạn muốn tìm kiếm...")

    @property
    def output_keys(self) -> List[str]:
        return ["response"]

    def _call(self, query: str) -> str:
        tools = [customer_search_tool]
        inputs = {
            "input": query,
        }
        prompt = hub.pull("hwchase17/openai-tools-agent")
        agent = create_tool_calling_agent(self.llm, tools, prompt)

        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=self.verbose,
            handle_parsing_errors=True,
        )
        ai_message = agent_executor.invoke(inputs)
        agent_output = ai_message['output']
        return agent_output
    

def search_agent_node(state: Dict[str, Any]) -> Dict[str, str]:
    "Define search agent node "
    search_agent = Searcher(llm=state["llm"], verbose=True)
    result = search_agent._call(state["query"])
    return {"search_result": result.dict()}




if __name__ == "__main__":
    # Một ví dụ về state để test
    query = "tìm kiếm cho tôi thông tin khách hàng tên nguyễn văn a cho tôi"
    llm = ChatGroq(model="llama3-70b-8192", temperature=0.3)
    search_agent = Searcher(llm)
    result = search_agent._call(query = query)
    print(result)




