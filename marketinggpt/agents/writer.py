import os
from typing import List, Dict, Any
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Tải các biến môi trường
dotenv_path = r"D:\gpt-marketer\marketinggpt\.env"
load_dotenv(dotenv_path)

# Set environment variables
os.environ['PINECONE_API_KEY'] = os.getenv('PINECONE_API_KEY')
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_SMITH_API_KEY")
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')


prompt_template = """
Based on the search result provided below, craft a meticulously composed 
and compelling email marketing message:

Search Result: {search_result}

Composition Guidelines:

1. Subject Line: Formulate a concise, engaging, and pertinent subject line that encapsulates 
   the email's essence.

2. Salutation: Commence the correspondence with an appropriate and cordial greeting.

3. Introductory Paragraph: Construct an attention-grabbing opening paragraph that clearly articulates the email's purpose.

4. Core Content: Develop 2-3 substantive paragraphs derived from the search result information. Ensure the content is lucid, concise, and provides significant value to the recipient.

5. Call-to-Action (CTA): Incorporate a distinct and persuasive call-to-action.

6. Conclusion: Craft a brief yet courteous closing paragraph.

7. Signature: Conclude the email with a professional signature.

Critical Considerations:

- Employ language that is both professional and approachable.
- Emphasize the benefits to the customer throughout the message.
- Maintain an optimal length of approximately 200-300 words.
- Ensure the content is optimized for mobile device viewing.
- Adhere strictly to email marketing regulations and GDPR compliance.

Please compose the email marketing message in accordance with the aforementioned guidelines.
"""

# Định nghĩa lớp Writer
class Writer:
    def __init__(self, llm, verbose=False, **kwargs):
        self.llm = llm
        self.verbose = verbose
        self.prompt = PromptTemplate(template=prompt_template, input_variables=["search_result"])
        self.chain = self.prompt | llm

    @property
    def output_keys(self) -> List[str]:
        return ["written_output"]

    def _call(self, search_result: str) -> str:
        inputs = {
            "search_result": search_result,
        }
        response = self.chain.invoke(inputs)
        return response



