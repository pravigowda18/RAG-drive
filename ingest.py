import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import GoogleDriveLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Pinecone as LangChainPinecone


load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "rag-chat1"
GOOGLE_DRIVE_FOLDER_ID = "144xxRiZbhLnGlGoUrOg6OMunvmjjeseQ"

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "service-account.json"


def load_from_drive(folder_id):
    return GoogleDriveLoader(
        folder_id=folder_id,
        recursive=False,
        scopes=["https://www.googleapis.com/auth/drive.readonly"]
    )

def document_splitter(docs, chunk_size=1000, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks.")
    return chunks


def download_embeddings():
    return HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2'
    )


print("\n--- Loading documents from Google Drive... ---")
drive_loader = load_from_drive(GOOGLE_DRIVE_FOLDER_ID)
docs = drive_loader.load()

if not docs:
    print(" No documents found. Check your folder ID / access.")
    exit()

print(f" Loaded {len(docs)} document(s).")


print("\n--- Splitting Documents ---")
chunked_docs = document_splitter(docs)


print("\n--- Downloading Embeddings ---")
embeddings = download_embeddings()

test_vector = embeddings.embed_query("hello world")
dimension = len(test_vector)


print(f"\n--- Initializing Pinecone (Index: {PINECONE_INDEX_NAME}) ---")

pc = Pinecone(api_key=PINECONE_API_KEY)

index_names = [index.name for index in pc.list_indexes()]

if PINECONE_INDEX_NAME not in index_names:
    print(" Creating new Pinecone index...")
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

print(" Pinecone index ready.")

print("\n--- Uploading chunks to Pinecone... ---")

vectorstore = LangChainPinecone.from_documents(
    documents=chunked_docs,
    embedding=embeddings,
    index_name=PINECONE_INDEX_NAME
)

print("\n---  SUCCESS! Ingestion complete. ---")
