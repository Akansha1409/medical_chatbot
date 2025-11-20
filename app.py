from flask import Flask, render_template, request, jsonify
from src.helper import (
    download_hugging_face_embeddings,
    load_medical_dataset,
    text_split
)
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv
import os
import time

app = Flask(__name__)

# -----------------------------
# LOAD ENV VARIABLES
# -----------------------------
load_dotenv()
os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")
os.environ["PINECONE_ENVIRONMENT"] = os.getenv("PINECONE_ENVIRONMENT", "us-east-1-gcp")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# -----------------------------
# 1. Load embeddings
# -----------------------------
print("Loading embeddings...")
embeddings = download_hugging_face_embeddings()
print("Embeddings loaded!")

# -----------------------------
# 2. Load and prepare dataset
# -----------------------------
print("Loading medical dataset...")
raw_docs = load_medical_dataset()

print("Splitting dataset into chunks...")
chunks = text_split(raw_docs)
print(f"Total chunks: {len(chunks)}")
print("Sample chunk:", chunks[0].page_content[:500])

# -----------------------------
# 3. Pinecone Setup
# -----------------------------
index_name = "medical-chatbot"

print("Connecting to Pinecone index...")
vectorstore = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

# OPTIONAL: upload chunks to Pinecone (run only once!)
# print("Uploading chunks to Pinecone...")
# start_time = time.time()
# vectorstore.add_documents(chunks)
# print("Upload complete!")
# print(f"Time taken: {time.time() - start_time:.2f} seconds")

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}  # Retrieve top 5 relevant chunks
)

# -----------------------------
# 4. Chat model
# -----------------------------
chatModel = ChatGroq(
    model="llama-3.3-70b-versatile"
)

# -----------------------------
# 5. Prompt (must include context)
# -----------------------------
system_prompt = "You are a medical assistant. Answer questions based on the provided context."

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "Answer the question based on the context below:\n\nContext:\n{context}\n\nQuestion: {input}")
    ]
)

# -----------------------------
# 6. RAG Chain
# -----------------------------
question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
base_chain = create_retrieval_chain(retriever, question_answer_chain)

# -----------------------------
# 7. Memory
# -----------------------------
chat_history_store = {}

def get_session_history(session_id: str):
    if session_id not in chat_history_store:
        chat_history_store[session_id] = ChatMessageHistory()
    return chat_history_store[session_id]

rag_chain = RunnableWithMessageHistory(
    base_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
    output_messages_key="answer"
)

# -----------------------------
# 8. Routes
# -----------------------------
@app.route("/get", methods=["POST"])
def chat():
    msg = request.form["msg"]

    # Invoke the RAG chain
    response = rag_chain.invoke(
        {"input": msg},
        config={"configurable": {"session_id": "user1"}}
    )

    # Extract the answer text
    answer_obj = response["answer"]

    # If the answer is a single object
    if hasattr(answer_obj, "content"):
        answer_text = answer_obj.content
    # If the answer is a list of documents
    elif isinstance(answer_obj, list):
        answer_text = "\n\n".join([doc.page_content for doc in answer_obj])
    else:
        # fallback to string conversion
        answer_text = str(answer_obj)

    return answer_text


# -----------------------------
# 9. Run Flask app
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
