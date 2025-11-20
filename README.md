# 🩺 Medical Chatbot with LangChain & Pinecone

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.3-green?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![LangChain](https://img.shields.io/badge/LangChain-Classic-orange?logo=data:image/png;base64)]()
[![Pinecone](https://img.shields.io/badge/Pinecone-VectorStore-purple?logo=data:image/png;base64)]()
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Embeddings-yellow?logo=huggingface&logoColor=black)]()
[![Groq](https://img.shields.io/badge/Groq-LLM-red)]()

---

## 💡 Project Description

**Medical Chatbot** is an AI-powered question-answering application that leverages **LangChain**, **Pinecone**, and **HuggingFace embeddings** to provide accurate medical responses based on a curated medical dataset.  

The bot uses a **Retrieval-Augmented Generation (RAG)** pipeline to fetch relevant document chunks from a vector database and generates context-aware answers using the **Groq LLM**.

---

## 🛠️ Tech Stack

- **Backend:** Flask  
- **Language Model:** Groq (`llama-3.3-70b-versatile`)  
- **Vector Database:** Pinecone  
- **Embeddings:** HuggingFace (`sentence-transformers/all-MiniLM-L6-v2`)  
- **Document Splitting:** LangChain Text Splitters  
- **RAG Pipeline:** LangChain Classic  
- **Environment Variables:** dotenv  
- **Deployment:** AWS Free Tier (Elastic Beanstalk / EC2)

---

## 📄 Features

✅ Ask medical questions and get context-aware answers

✅ Uses Pinecone vector database for efficient retrieval

✅ RAG-based system ensures accurate, relevant responses

✅ Handles large datasets with document chunking

✅ Maintains chat history per session

✅ Ready for deployment on AWS Free Tier

---

## 🌐 Deployment

The app can be deployed on AWS Elastic Beanstalk or EC2 Free Tier:

Use eb init → eb create <env> → eb deploy for Elastic Beanstalk

Make sure environment variables are set in the cloud

---

## 📚 Dataset

Dataset: ruslanmv/ai-medical-chatbot from HuggingFace

Contains patient-doctor conversations

Used to build vector store and train the retrieval system

Open port 8080 for access

---

## 🔗 References

LangChain Documentation

Pinecone Documentation

HuggingFace Datasets

Flask Documentation
