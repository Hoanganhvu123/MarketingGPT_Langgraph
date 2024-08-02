import os
from typing import List
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain import hub
from marketinggpt.tools.product_search import product_search_tool
from marketinggpt.tools.customer_search import customer_search_tool
from dotenv import load_dotenv


dotenv_path = r"D:\gpt-marketer\marketinggpt\.env"
load_dotenv(dotenv_path)

# Set environment variables
os.environ['PINECONE_API_KEY'] = os.getenv('PINECONE_API_KEY')
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_SMITH_API_KEY")
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')


class Searcher:
    def __init__(self, llm, verbose=False, **kwargs):
        self.llm = llm
        self.verbose = verbose

    @property
    def output_keys(self) -> List[str]:
        return ["response"]

    def _call(self, query: str) -> str:
        tools = [product_search_tool, customer_search_tool]
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





