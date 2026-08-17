"""
ingest.py

Loads a source PDF, splits it into chunks, generates embeddings,
and stores them in a Pinecone vector index. Run this once whenever
the source document is added or updated.
"""

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# Load and parse the source document
print("Step 1: Loading PDF...")
loader = PyPDFLoader("restaurant_brochure.pdf")
documents = loader.load()
print(f"Loaded {len(documents)} page(s).")

# Split into smaller chunks for embedding
print("Step 2: Splitting text into chunks...")
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(documents)
print(f"Created {len(chunks)} chunk(s).")

# Connect to Pinecone and ensure the target index exists
print("Step 3: Connecting to Pinecone...")
pc = Pinecone(api_key=PINECONE_API_KEY)

existing_indexes = [index.name for index in pc.list_indexes()]
if INDEX_NAME not in existing_indexes:
    print(f"Index '{INDEX_NAME}' not found, creating it...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=3072,  # matches the Gemini embedding model's default output size
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# Generate embeddings and upsert into Pinecone
print("Step 4: Generating embeddings and storing in Pinecone...")
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=GEMINI_API_KEY
)
PineconeVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    index_name=INDEX_NAME
)

print("Done. Document data has been indexed in Pinecone.")