'''
import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv
#from langchain_openai import ChatOpenAI
import getpass
import os

if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = getpass.getpass("Enter your Groq API key: ")
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
#OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
os.environ["GROQ_API_KEY"]=GROQ_API_KEY

class LLMModel:
    def __init__(self, model_name="deepseek-r1-distill-llama-70b"):
        if not model_name:
            raise ValueError("Model is not defined.")
        self.model_name = model_name
        self.openai_model=ChatGroq(model=self.model_name)
        
    def get_model(self):
        return self.openai_model

if __name__ == "__main__":
    llm_instance = LLMModel()  
    llm_model = llm_instance.get_model()
    response=llm_model.invoke("hi")

    print(response)
'''

import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import getpass

# Load environment variables from .env
load_dotenv()

# Check if GROQ_API_KEY is set, otherwise ask user
if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = getpass.getpass("Enter your Groq API key: ")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

class LLMModel:
    def __init__(self, model_name="meta-llama/llama-4-maverick-17b-128e-instruct"):
        if not model_name:
            raise ValueError("Model is not defined.")
        self.model_name = model_name
        # Corrected: use model_name parameter
        self.llm = ChatGroq(model_name=self.model_name)
        
    def get_model(self):
        return self.llm

if __name__ == "__main__":
    llm_instance = LLMModel()  
    llm_model = llm_instance.get_model()
    
    # Invoke the model
    response = llm_model.invoke("hi")
    
    # Print only the content
    print(response.content)
