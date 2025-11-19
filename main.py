# main.py
import uvicorn
import os
from fastapi import FastAPI, Depends
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

# LangChain Imports
from langchain_google_genai import ChatGoogleGenerativeAI
from pinecone import Pinecone
from langchain_pinecone import Pinecone as LangchainPinecone
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Pinecone Imports
from langchain_huggingface import HuggingFaceEmbeddings



load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = "rag-chat"

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2'
    )

pc = Pinecone(api_key=PINECONE_API_KEY)

vectorstore = LangchainPinecone.from_existing_index(
    index_name=PINECONE_INDEX_NAME,
    embedding=embeddings
)

retriever = vectorstore.as_retriever()

template = """
You are a helpful assistant. Answer the user's question based ONLY on the 
following context. If the context doesn't contain the answer, 
say "I'm sorry, I don't have that information in my knowledge base."

Context:
{context}

Question:
{question}

Answer:
"""
prompt = PromptTemplate.from_template(template)

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)



app = FastAPI(title="RAG Agent Backend")

class ChatRequest(BaseModel):
    question: str
    
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (for development only)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def read_root():
    return {"status": "Backend is running!"}

@app.post("/chat")
def handle_chat(request: ChatRequest):
    """
    Handles a user's chat question.
    Invokes the RAG chain to get an answer.
    """
    try:
        answer = rag_chain.invoke(request.question)
        return {"answer": answer}
    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)