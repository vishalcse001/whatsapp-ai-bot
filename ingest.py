"""
ingest.py (Gemini version)
Iska kaam: brochure.pdf ko padhna, chhote chunks me todna,
aur Pinecone database me save karna (Gemini embeddings ke form me).
Ye script sirf tab chalao jab naya PDF add/update karna ho.
"""

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

# 1. .env file se saari keys load karo
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# 2. PDF ko load karo
print("Step 1: PDF padh rahe hain...")
loader = PyPDFLoader("brochure.pdf")
documents = loader.load()
print(f"PDF me {len(documents)} pages mile.")

# 3. Bade text ko chhote chunks me todo
print("Step 2: Text ko chunks me tod rahe hain...")
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(documents)
print(f"Total {len(chunks)} chunks bane.")

# 4. Pinecone se connect karo
print("Step 3: Pinecone se connect ho rahe hain...")
pc = Pinecone(api_key=PINECONE_API_KEY)

# NOTE: Gemini ke embeddings ka dimension 768 hota hai
# (OpenAI ka 1536 tha) - isliye index banate waqt ye number
# match hona zaroori hai, warna error aayega.
existing_indexes = [index.name for index in pc.list_indexes()]
if INDEX_NAME not in existing_indexes:
    print(f"Index '{INDEX_NAME}' nahi mila, naya bana rahe hain...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=768,  # Gemini embedding model ka fixed size
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# 5. Chunks ko embeddings (numbers) me convert karke Pinecone me save karo
print("Step 4: Chunks ko embeddings banake Pinecone me save kar rahe hain...")
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=GEMINI_API_KEY,
    output_dimensionality=768
)
PineconeVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    index_name=INDEX_NAME
)

print("✅ Done! PDF ka data Pinecone me safal save ho gaya.")