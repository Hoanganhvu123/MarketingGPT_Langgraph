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

PRODUCT_RECOMMENDATION_PROMPT = """
    You are a chatbot assistant specializing in providing product information and
    recommendations using Pandas in Python.
    Your primary tasks are:

    Provide detailed information about a specific product based on user queries.
    Recommend relevant products to users based on their preferences and requirements.

    The dataframe df contains the following columns about product information:

    product_code: A unique identifier for each product (string)
    product_name: The name of the product (string)
    material: The material composition of the product (string)
    size: The available sizes of the product (string)
    color: The available colors of the product (string)
    brand: The brand that manufactures or sells the product (string)
    gender: The product for target gender(e.g., male, female, unisex) (string)
    stock_quantity: The quantity of the product available in stock (integer)
    price: The price of the product, which can be a string or numeric value (string or numeric)

    To provide product information, generate a Python command that:

    Handles product names in a case-insensitive manner and allows for partial matches.
    Retrieves all relevant columns of information about the requested product.
    Uses efficient indexing and filtering techniques to retrieve data.
    Converts string values to float for the 'price' column if necessary.
    Validates input to prevent potential errors.

    To recommend products, generate a Python command that:

    Filters products based on user-specified criteria such as price range, material, color, size, or brand.
    Handles multiple criteria combined with logical operators (and, or).
    Recommends a specified number of products that match the criteria.
    Ensures code readability and maintainability.

    Output only the Python command(s). Do not include any explanations, comments, quotation marks, or additional information. Only output the command(s) themselves.
    Start!
    Question: {input}
"""

class ProductSearchInput(BaseModel):
    input: str = Field(description="Useful for when you need to answer questions about product information. Please use Vietnamese input commands when using this tool.")

@tool
def product_search_tool(input: str) -> Union[List[Dict], str]:
    """
    Tìm kiếm thông tin sản phẩm và trả về các thông tin liên quan.

    Args:
        input (str): Chuỗi tìm kiếm để tìm các sản phẩm.

    Returns:
        Union[List[Dict], str]: Kết quả tìm kiếm dưới dạng danh sách từ điển hoặc thông báo lỗi nếu có.
    """
    try:
        llm = ChatGroq(temperature=0, model_name="llama3-70b-8192")
        prompt = PromptTemplate(
            template=PRODUCT_RECOMMENDATION_PROMPT,
            input_variables=["input"]
        )
        product_data = pd.read_csv("E:\\chatbot\\ShoppingGPT\\backend\\shoppinggpt\\data\\products.csv")
        python_tool = PythonAstREPLTool(globals={"df": product_data})
        
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
def main():
    query = "Recommend products under 500,000 VND"
    result = product_search_tool(query)
    print(result)

if __name__ == "__main__":
    main()
