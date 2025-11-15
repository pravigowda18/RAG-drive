# main.py
import uvicorn
from fastapi import FastAPI

app = FastAPI(title="RAG Agent Backend")

@app.get("/")
def read_root():
    return {"status": "Backend is running!"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)