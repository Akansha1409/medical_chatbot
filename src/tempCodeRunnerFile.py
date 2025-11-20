
# -----------------------------
# LOAD MEDICAL DATASET
# -----------------------------
from datasets import load_dataset
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter



def load_medical_dataset():
    """
    Loads the ruslanmv/ai-medical-chatbot dataset and converts it into
    LangChain Document objects.
    """
    ds = load_dataset("ruslanmv/ai-medical-chatbot")

    # Dataset contains fields like: Patient, Doctor, Description
    documents = []

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

    return documents


# -----------------------------
# CHUNK THE DOCUMENTS
# -----------------------------
def text_split(docs):
    """
    Split documents into smaller chunks for embeddings and retrieval.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    return splitter.split_documents(docs)