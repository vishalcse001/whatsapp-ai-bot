"""
chat.py (Gemini version - manual RAG chain)
Iska kaam: user ka sawal lena, Pinecone me relevant jaankari dhundna,
aur Gemini se ek human-jaisa jawab banwana.
Isko baar-baar chala sakte ho - ye terminal me chat jaisa chalega.

NOTE: Humne yaha RetrievalQA use nahi kiya (wo purane langchain
versions me tha, naye versions me hata diya gaya). Iski jagah
hum khud retriever aur LLM ko manually jodte hain - ye tarika
har version me kaam karega.
"""

import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# 1. Wahi embeddings model use karo jo ingest.py me use kiya tha
# (dono files me SAME embedding model hona zaroori hai, warna
# matching sahi se kaam nahi karegi)
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=GEMINI_API_KEY,
    output_dimensionality=768
)

# 2. Pinecone se already-saved data connect karo
vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)

# 3. Retriever banao - ye Pinecone me se sabse relevant chunks dhundega
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})  # top 3 relevant chunks

# 4. Gemini AI model set karo
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",   # fast aur accha model
    temperature=0.3,             # kam temperature = zyada factual jawab
    google_api_key=GEMINI_API_KEY
)


def get_ai_response(user_question: str) -> str:
    """
    Manual RAG flow:
    1. Retriever se relevant chunks nikalo
    2. Un chunks ko ek prompt me jodo
    3. LLM ko bhejo aur jawab lo
    Ye function baad me FastAPI (Phase 3) me bhi reuse hoga.
    """
    # Step 1: Relevant chunks dhundo
    relevant_docs = retriever.invoke(user_question)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    # Step 2: Prompt banao jisme context + sawal dono hon
    prompt = f"""Tum ek helpful assistant ho jo brochure ki jaankari ke base par jawab deta hai.
Neeche di gayi jaankari ka use karke user ke sawal ka jawab do.
Agar jaankari me answer nahi mile, to bolo "Ye jaankari mere paas nahi hai."

Jaankari:
{context}

Sawal: {user_question}

Jawab:"""

    # Step 3: LLM se jawab lo
    response = llm.invoke(prompt)
    return response.content


# Terminal me test karne ke liye loop
if __name__ == "__main__":
    print("🤖 AI Bot ready hai! (bahar aane ke liye 'exit' likho)\n")
    while True:
        user_input = input("Tum: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        answer = get_ai_response(user_input)
        print(f"Bot: {answer}\n")