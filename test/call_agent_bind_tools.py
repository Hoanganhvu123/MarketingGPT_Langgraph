import os
from typing import List, Dict, Any, TypedDict, Annotated
from langchain.agents import AgentExecutor, create_openai_functions_agent, create_react_agent
from langchain import hub
from marketinggpt.tools.product_search import product_search_tool
from marketinggpt.tools.customer_search import customer_search_tool
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langgraph.prebuilt import ToolExecutor
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
import operator
import json

# Load environment variables
dotenv_path = r"D:\gpt-marketer\marketinggpt\.env"
load_dotenv(dotenv_path)

# Set environment variables
os.environ['PINECONE_API_KEY'] = os.getenv('PINECONE_API_KEY')
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_SMITH_API_KEY")
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')

# Set up the language model
llm = ChatGroq(model="llama3-70b-8192", temperature=0.3)

# Define tools
tools = [product_search_tool, customer_search_tool]

# Create tool executor
tool_executor = ToolExecutor(tools)

# Bind tools to the model

llm = ChatGroq(model="llama3-70b-8192", temperature=0.3)
llm = llm.bind_tools(tools)

# Define the state
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    last_tool_result: str
    
    
def call_model(state: AgentState) -> Dict[str, Any]:
    messages = state["messages"]
    response = llm.invoke(messages)
    print("Model response:", response)  # Add this line for debugging
    print("---")
    return {"messages": [response]}


# Define the nodes
def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1] if state["messages"] else None
    if isinstance(last_message, AIMessage) and last_message.content:
        return "end"
    return "continue"



def call_tool(state: AgentState) -> Dict[str, Any]:
    last_message = state["messages"][-1]
    tool_calls = last_message.additional_kwargs.get('tool_calls', [])
    
    if not tool_calls:
        return {"messages": [AIMessage(content="No tool calls were made.")]}
    
    tool_call = tool_calls[0]  # Only process the first tool call
    action = {
        "tool": tool_call['name'],
        "tool_input": json.loads(tool_call['function']['arguments']),
    }
    print("Calling tool:", action)  # Add this line for debugging
    result = tool_executor.invoke(action)
    
    return {
        "messages": [ToolMessage(content=str(result), tool_call_id=tool_call['id'])],
        "last_tool_result": str(result)
    }
    
    
def process_result(state: AgentState) -> Dict[str, Any]:
    last_tool_result = state.get("last_tool_result", "")
    response = llm.invoke([
        HumanMessage(content=f"Based on the search result: {last_tool_result}, please provide a summary of products under 500k.")
    ])
    return {"messages": [response]}

# Define the graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("call_model", call_model)
workflow.add_node("call_tool", call_tool)
workflow.add_node("process", process_result)

# Add edges
workflow.set_entry_point("call_model")
workflow.add_conditional_edges(
    "call_model",
    should_continue,
    {
        "continue": "call_tool",
        "end": "process",
    },
)
workflow.add_edge("call_tool", "call_model")
workflow.add_edge("process", END)

# Compile the graph
app = workflow.compile()

if __name__ == "__main__":
    query = "tìm cho tôi những sản phẩm giá dưới 500k"
    initial_state = AgentState(messages=[HumanMessage(content=query)], last_tool_result="")
    
    try:
        for output in app.stream(initial_state):
            for key, value in output.items():
                print()
                print()
                print("---")
                print(f"Output from node '{key}':")
                print(value)
            print("\n---\n")
            print()
            print()
            print()
            
            if key == "process":
                break
        
        if "process" in output:
            print("Final result:", output["process"]["messages"][0].content)
        else:
            print("Process did not complete. Last output:", output)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Current state:")
        print(initial_state)