from datasets import load_dataset
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from typing import List

# -----------------------------
# LOAD MEDICAL DATASET
# -----------------------------
def load_medical_dataset() -> List[Document]:
    """
    Loads the ruslanmv/ai-medical-chatbot dataset and converts it into
    LangChain Document objects.
    """
    print("Loading dataset...")
    ds = load_dataset("ruslanmv/ai-medical-chatbot")

    documents: List[Document] = []

    for row in ds["train"]:
        patient = row.get("Patient", "")
        doctor = row.get("Doctor", "")
        description = row.get("Description", "")

        text = f"Patient: {patient}\nDoctor: {doctor}\nDescription: {description}"
        documents.append(
            Document(
                page_content=text,
                metadata={"source": "medical_dataset"}
            )
        )

    print(f"Total documents loaded: {len(documents)}")
    return documents


# -----------------------------
# CHUNK THE DOCUMENTS
# -----------------------------
def text_split(docs: List[Document]) -> List[Document]:
    """
    Split documents into smaller chunks for embeddings and retrieval.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(docs)
    print(f"Total chunks created: {len(chunks)}")
    return chunks


# -----------------------------
# EMBEDDINGS MODEL
# -----------------------------
def download_hugging_face_embeddings() -> HuggingFaceEmbeddings:
    """
    Returns HuggingFace embeddings model.
    """
    print("Loading HuggingFace embeddings model...")
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
