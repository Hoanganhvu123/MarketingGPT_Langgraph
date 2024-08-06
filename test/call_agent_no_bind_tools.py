import os
from typing import List, Dict, Any, TypedDict, Annotated
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
import logging
from langgraph.prebuilt.tool_executor import ToolExecutor, ToolInvocation

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
# tools = [product_search_tool, customer_search_tool]

tools = [product_search_tool]

# Create tool executor
tool_executor = ToolExecutor(tools)

# Define the state
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    last_tool_result: str

def call_model(state: AgentState) -> Dict[str, Any]:
    messages = state["messages"]
    response = llm.invoke(messages)
    logging.info(f"Model response: {response}")
    return {"messages": [response]}

def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1] if state["messages"] else None
    if isinstance(last_message, AIMessage) and "FINAL ANSWER:" in last_message.content.upper():
        return "end"
    return "continue"

def call_tool(state: AgentState) -> Dict[str, Any]:
    last_message = state["messages"][-1]
    content = last_message.content
    
    for tool in tools:
        if tool.name.lower() in content.lower():
            tool_input = content.split(tool.name)[-1].strip()
            action = {
                "tool": tool.name,
                "tool_input": tool_input,
            }
            logging.info(f"Calling tool: {action}")
            invocation = ToolInvocation(tool=action["tool"], tool_input=action["tool_input"])
            result = tool_executor.invoke(invocation)
            return {
                "messages": [ToolMessage(content=str(result), tool_call_id=f"call_{tool.name}")],
                "last_tool_result": str(result)
            }
    
    logging.warning("No specific tool was mentioned.")
    return {"messages": [AIMessage(content="No specific tool was mentioned.")]}
    
    logging.warning("No specific tool was mentioned.")
    return {"messages": [AIMessage(content="No specific tool was mentioned.")]}

def process_result(state: AgentState) -> Dict[str, Any]:
    last_message = state["messages"][-1]
    final_answer = last_message.content.split("FINAL ANSWER:")[-1].strip()
    return {"messages": [AIMessage(content=final_answer)]}

# Define the graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("call_model", call_model)
workflow.add_node("action", call_tool)
workflow.add_node("process", process_result)

# Add edges
workflow.set_entry_point("call_model")
workflow.add_conditional_edges(
    "call_model",
    should_continue,
    {
        "continue": "action",
        "end": "process",
    },
)
workflow.add_edge("action", "call_model")
workflow.add_edge("process", END)

# Compile the graph
app = workflow.compile()

if __name__ == "__main__":
    query = "tìm cho tôi những sản phẩm giá dưới 500k"
    initial_message = """
You are an AI assistant
You have access to the following tools:
1. product_search_tool: Use this to search for products.
2. customer_search_tool: Use this to search for customer information.

Your task is to find products under 500k. If you need to use a tool, mention 
its name clearly in your response.
When you have a final answer, start your response with 'FINAL ANSWER:'.
"""
    initial_state = AgentState(
        messages=[HumanMessage(content=initial_message)],
        last_tool_result=""
    )

    try:
        for output in app.stream(initial_state):
            for key, value in output.items():
                print(f"Output from node '{key}':")
                print("---")
                print(value)
            print("\n---\n")

        final_state = app.get_state()
        if "process" in final_state:
            print("Final result:", final_state["process"]["messages"][0].content)
        else:
            print("Process did not complete as expected.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")