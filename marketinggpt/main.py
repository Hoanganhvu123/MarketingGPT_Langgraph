from langgraph.graph import StateGraph, START, END
from marketinggpt.agents.searcher import Searcher
from marketinggpt.agents.writer import Writer
import os
from typing import TypedDict, Dict, Any
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()
# Set environment variables
os.environ['PINECONE_API_KEY'] = os.getenv('PINECONE_API_KEY')
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_SMITH_API_KEY")
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')


# Định nghĩa graph state
class GraphState(TypedDict):
    query: str
    search_result: str
    written_output: str
    
    
def search_agent_node(state: Dict[str, Any]) -> Dict[str, str]:
    "Define search agent node "
    search_agent = Searcher(llm, verbose=True)
    result = search_agent._call(state["query"])
    return {"search_result": result}


def writer_agent_node(state: Dict[str, Any]) -> Dict[str, str]:
    "Define search agent node "
    writer_agent = Writer(llm, verbose=True)
    result = writer_agent._call(state["search_result"])
    return {"written_output": result}

# Tạo và chạy đồ thị
def create_graph():
    graph = StateGraph(GraphState)

    # Node for search agent
    graph.add_node("search", lambda state: search_agent_node(state))
    graph.add_node("write", lambda state: writer_agent_node(state))

    # Kết nối các node
    graph.add_edge(START, "search")
    graph.add_edge("search", "write")
    graph.add_edge("write", END)

    return graph.compile()

if __name__ == "__main__":
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.1)
    graph = create_graph()

    # Một ví dụ về state để test
    initial_state = {"query": "viết email marketing cho tôi khách hàng a"}
    result = graph.invoke(initial_state)
    print(result["written_output"]["content"])