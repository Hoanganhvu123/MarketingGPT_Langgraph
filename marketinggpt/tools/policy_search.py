import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Union, List, Dict

from langchain.tools import tool
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
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

index_name = "doan"
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")


class DocumentProcessor:
    def __init__(self, data_path: str):
        self.index_name = "doan"
        self.data_path = data_path

    def load_and_process_documents(self):
        """Load documents, split into chunks, create embeddings, and store in Pinecone."""
        loader = TextLoader(self.data_path, encoding='utf8')
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
        document_chunks = text_splitter.split_documents(documents)
        vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
        vectorstore.add_documents(document_chunks)
        print(f"Successfully indexed {len(document_chunks)} document chunks into Pinecone.")


class PolicySearchInput(BaseModel): 
    query: str = Field(description="...")


@tool
def policy_search_tool(query: str) -> Union[List[Dict], str]:
    """
    Tìm kiếm các tài liệu chính sách và trả về thông tin liên quan.

    Args:
        query (str): Chuỗi tìm kiếm để tìm các tài liệu chính sách.

    Returns:
        Union[List[Dict], str]: Kết quả tìm kiếm dưới dạng danh sách từ điển hoặc thông báo lỗi nếu có.
    """

    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={'k': 5})
    try:
        result = retriever.invoke(query)
        return result
    except Exception as e:
        return repr(e)





