# 🤖 AI Document Chatbot (RAG-Based System)

## 📌 Overview

This project is a full-stack AI application that allows users to upload multiple PDF documents and interact with them through a conversational interface.

It uses a Retrieval-Augmented Generation (RAG) pipeline to generate accurate, context-aware answers strictly based on the uploaded documents, reducing hallucinations and improving reliability.

---

## 🚀 Key Features

* 📂 Upload and analyze multiple PDF documents
* 💬 Chat-based interface with conversation history
* 🧠 Context-aware answers using RAG pipeline
* 🔍 Semantic search using vector embeddings (FAISS)
* 📌 Source highlighting for transparency
* ⚡ Fast responses using Groq LLM
* 🖱️ Drag & drop document upload

---

## 🏗️ System Architecture

![Architecture Diagram](./data/architecture.png)

### 🔍 Workflow

1. User uploads one or more PDF documents
2. Documents are converted into text using PDF loader
3. Text is split into smaller chunks for processing
4. Each chunk is converted into embeddings (vector form)
5. Embeddings are stored in FAISS vector database
6. User query is converted into embedding
7. Relevant chunks are retrieved using similarity search
8. Retrieved context is passed to Groq LLM
9. LLM generates accurate answer based only on context
10. Answer is displayed in chat UI with source references

---

## 🛠️ Tech Stack

* **Frontend:** Streamlit
* **Backend:** Python
* **LLM:** Groq (LLaMA 3.1 Instant)
* **Embeddings:** HuggingFace Sentence Transformers
* **Vector Database:** FAISS
* **Framework:** LangChain

---

## ⚙️ Installation & Setup

### 1. Clone Repository

```bash
git clone <your-repo-link>
cd rag-chatbot
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API Key

Create `.streamlit/secrets.toml`:

```toml
GROQ_API_KEY = "your_api_key_here"
```

### 4. Run Application

```bash
streamlit run app_ui.py
```

---

## 💡 Example Use Cases

* Resume analysis
* Academic PDF Q&A
* Business reports understanding
* Multi-document knowledge assistant

---

## 📊 Project Highlights

* Built a **production-ready RAG system from scratch**
* Implemented **multi-document semantic search**
* Improved answer accuracy using **prompt engineering**
* Designed a **chat-based UI with history support**
* Added **source attribution for explainability**

---

## 🚀 Future Enhancements

* Excel/CSV data visualization support
* Authentication and user sessions
* Cloud deployment with scaling
* Multi-user collaboration

---

## 👨‍💻 Author

**Kiran Kaduluri**
