import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Union, List, Dict

from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_experimental.tools.python.tool import PythonAstREPLTool
import pandas as pd
from langchain.tools import tool
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings

dotenv_path = r"D:\gpt-marketer\marketinggpt\.env"

# Load environment variables từ tệp .env
load_dotenv(dotenv_path)
os.environ['PINECONE_API_KEY'] = os.getenv('PINECONE_API_KEY')
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_SMITH_API_KEY")
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

CUSTOMER_SEARCH_PROMPT = """
    You are a chatbot assistant specializing in providing customer information and
    recommendations using Pandas in Python.
    Your primary tasks are:

    Provide detailed information about a specific customer based on user queries.
    Recommend relevant products to customers based on their preferences and requirements.

    The dataframe df contains the following columns about customer information:

    customer_id: A unique identifier for each customer (string)
    name: The name of the customer (string)
    email: The email of the customer (string)
    phone_number: The phone number of the customer (string)
    gender: The gender of the customer (e.g., male, female) (string)
    marital_status: The marital status of the customer (e.g., single, married) (string)
    income: The income of the customer (numeric)
    age: The age of the customer (integer)
    interests: The interests of the customer (string)
    purchase_history: The purchase history of the customer (string)

    To provide customer information, generate a Python command that:

    Handles customer names in a case-insensitive manner and allows for partial matches.
    Retrieves all relevant columns of information about the requested customer.
    Uses efficient indexing and filtering techniques to retrieve data.
    Validates input to prevent potential errors.

    To recommend products to customers, generate a Python command that:

    Filters customers based on user-specified criteria such as age range, income level, gender, marital status, or interests.
    Handles multiple criteria combined with logical operators (and, or).
    Recommends products based on the customer's interests and purchase history.
    Ensures code readability and maintainability.

    Output only the Python command(s). Do not include any explanations, comments, quotation marks, or additional information. Only output the command(s) themselves.
    Start!
    Question: {input}
"""

class CustomerSearchInput(BaseModel):
    input: str = Field(description="Useful for when you need to answer questions about customer information. Please use Vietnamese input commands when using this tool.")

class CustomerDataLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.data_frame = None

    def load_data(self):
        self.data_frame = pd.read_csv(self.file_path)

    def get_data_frame(self):
        if self.data_frame is None:
            self.load_data()
        return self.data_frame

@tool
def customer_search_tool(input: str) -> Union[List[Dict], str]:
    """
    Tìm kiếm thông tin khách hàng và trả về các thông tin liên quan.

    Args:
        input (str): Chuỗi tìm kiếm để tìm các khách hàng.

    Returns:
        Union[List[Dict], str]: Kết quả tìm kiếm dưới dạng danh sách từ điển hoặc thông báo lỗi nếu có.
    """
    try:
        llm = ChatGroq(temperature=0, model_name="llama3-70b-8192")
        prompt = PromptTemplate(
            template=CUSTOMER_SEARCH_PROMPT,
            input_variables=["input"]
        )
        customer_data_loader = CustomerDataLoader("D:\\gpt-marketer\\marketinggpt\\data\\customers.csv")
        customer_data_loader.load_data()
        customer_data = customer_data_loader.get_data_frame()
        python_tool = PythonAstREPLTool(globals={"df": customer_data})
        
        # Construct the chain
        chain = (
            {"input": RunnablePassthrough()}
            | prompt
            | llm
            | (lambda x: python_tool.invoke(x.content))
        )
        result = chain.invoke(input)
        return result
    except Exception as e:
        return repr(e)

# Example usage of the tool
# def main():
#     query = "Find customers with income over 10,000,000 VND"
#     result = customer_search_tool(query)
#     print(result)

# if __name__ == "__main__":
#     main()
