import os
from langchain_groq import ChatGroq

from dotenv import load_dotenv

load_dotenv()

api=os.environ.get("GROQ")


llm=ChatGroq(
    
)