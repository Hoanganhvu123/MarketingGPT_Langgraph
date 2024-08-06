import os
import sqlite3
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Union, List, Dict

from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_experimental.tools.python.tool import PythonAstREPLTool
from langchain.tools import tool
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings

dotenv_path = r"D:\gpt-marketer\marketinggpt\.env"

# Load environment variables từ tệp .env
load_dotenv(dotenv_path)
os.environ['PINECONE_API_KEY'] = os.getenv('PINECONE_API_KEY')
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_SMITH_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"


embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")


CUSTOMER_SEARCH_PROMPT = """
    You are a chatbot assistant specializing in providing customer information and
    recommendations using SQL queries.
    Your primary tasks are:

    Provide detailed information about a specific customer based on user queries.
    Recommend relevant products to customers based on their preferences and requirements.

    The database table 'customers' contains the following columns about customer information:

    customer_id: A unique identifier for each customer (TEXT)
    name: The name of the customer (TEXT)
    email: The email of the customer (TEXT)
    phone_number: The phone number of the customer (TEXT)
    gender: The gender of the customer (e.g., male, female) (TEXT)
    marital_status: The marital status of the customer (e.g., single, married) (TEXT)
    income: The income of the customer (REAL)
    age: The age of the customer (INTEGER)
    interests: The interests of the customer (TEXT)
    purchase_history: The purchase history of the customer (TEXT)

    To provide customer information or recommend products, generate an SQL query that:

    Handles customer names in a case-insensitive manner and allows for partial matches.
    Retrieves all relevant columns of information about the requested customer or filters customers based on criteria.
    Uses efficient indexing and filtering techniques to retrieve data.
    Ensures SQL injection prevention by using parameterized queries.

    Output only the SQL query. Do not include any explanations, comments, quotation marks, or additional information. Only output the query itself.
    Start!
    Question: {input}
"""

class CustomerSearchInput(BaseModel):
    input: str = Field(description="Useful for when you need to answer questions about customer information. Please use Vietnamese input commands when using this tool.")

class CustomerDataLoader:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = None

    def connect(self):
        self.conn = sqlite3.connect(self.db_path)

    def close(self):
        if self.conn:
            self.conn.close()

    def execute_query(self, query: str, params: tuple = ()) -> List[Dict]:
        if not self.conn:
            self.connect()
        cursor = self.conn.cursor()
        cursor.execute(query, params)
        columns = [col[0] for col in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

@tool
def customer_search_tool(input: str) -> Union[List[Dict], str]:
    """
    Tìm kiếm thông tin khách hàng và trả về các thông tin liên quan sử dụng SQLite.

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
        customer_data_loader = CustomerDataLoader("D:\\gpt-marketer\\marketinggpt\\data\\customers.db")
        
        def execute_sql_query(query: str) -> List[Dict]:
            return customer_data_loader.execute_query(query)
        
        # Construct the chain
        chain = (
            {"input": RunnablePassthrough()}
            | prompt
            | llm
            | (lambda x: execute_sql_query(x.content))
        )
        result = chain.invoke(input)
        return result
    except Exception as e:
        return repr(e)
    finally:
        customer_data_loader.close()

# Example usage of the tool
# def main():
#     query = "Tìm khách hàng có thu nhập trên 10,000,000 VND"
#     result = customer_search_tool(query)
#     print(result)

# if __name__ == "__main__":
#     main()