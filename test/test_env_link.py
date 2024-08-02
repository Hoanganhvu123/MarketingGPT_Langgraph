import os
from dotenv import load_dotenv

dotenv_path = r"D:\gpt-marketer\marketinggpt\.env"

if not load_dotenv(dotenv_path):
    raise FileNotFoundError(f"Không tìm thấy tệp .env tại đường dẫn: {dotenv_path}")

required_env_vars = [
    'PINECONE_API_KEY', 'GROQ_API_KEY', 'LANGCHAIN_SMITH_API_KEY', 'GOOGLE_API_KEY'
]

for var in required_env_vars:
    if os.getenv(var) is None:
        raise ValueError(f"Biến môi trường {var} không được load đúng cách từ tệp .env")
    else:
        print(f"{var}: {os.getenv(var)}")


os.environ['PINECONE_API_KEY'] = os.getenv('PINECONE_API_KEY')
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_SMITH_API_KEY")
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')


print("PINECONE_API_KEY:", os.environ['PINECONE_API_KEY'])
print("GROQ_API_KEY:", os.environ['GROQ_API_KEY'])
print("LANGCHAIN_API_KEY:", os.environ["LANGCHAIN_API_KEY"])
print("GOOGLE_API_KEY:", os.environ['GOOGLE_API_KEY'])
print("LANGCHAIN_TRACING_V2:", os.environ["LANGCHAIN_TRACING_V2"])
print("LANGCHAIN_ENDPOINT:", os.environ["LANGCHAIN_ENDPOINT"])
