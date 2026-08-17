"""
chat.py

Core RAG (Retrieval-Augmented Generation) logic. Given a user question,
retrieves the most relevant chunks from Pinecone and generates a grounded
response using Gemini. Exposes get_ai_response() for reuse in main.py,
and supports standalone terminal testing via the __main__ block.
"""

import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# Must match the embedding model used in ingest.py
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=GEMINI_API_KEY
)

vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3,
    google_api_key=GEMINI_API_KEY
)


def get_ai_response(user_question: str) -> str:
    """
    Runs the retrieval-augmented generation flow:
    1. Retrieve the top-k relevant chunks from Pinecone
    2. Build a grounded prompt with that context
    3. Generate a response with the LLM
    """
    relevant_docs = retriever.invoke(user_question)
    context = "\n\n".join(doc.page_content for doc in relevant_docs)

    prompt = f"""You are a helpful assistant answering questions based on the brochure content below.
Use only the information provided to answer the user's question.
If the answer isn't contained in the information below, say "I don't have that information."

Context:
{context}

Question: {user_question}

Answer:"""

    response = llm.invoke(prompt)
    return response.content


if __name__ == "__main__":
    print("AI Bot ready. Type 'exit' to quit.\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        answer = get_ai_response(user_input)
        print(f"Bot: {answer}\n")