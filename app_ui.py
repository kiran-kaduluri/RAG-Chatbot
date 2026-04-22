import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from groq import Groq
import tempfile

# -------------------------
# CONFIG
# -------------------------
st.set_page_config(page_title="AI Chatbot", layout="wide")
st.title("🤖 AI Document Chatbot (RAG)")

# -------------------------
# SESSION STATE
# -------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -------------------------
# FILE UPLOAD
# -------------------------
uploaded_files = st.file_uploader(
    "📂 Drag & Drop or Upload PDFs",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:

    all_docs = []

    # Load PDFs
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            file_path = tmp_file.name

        loader = PyPDFLoader(file_path)
        docs = loader.load()
        all_docs.extend(docs)

    st.success(f"✅ {len(uploaded_files)} PDF(s) loaded!")

    # -------------------------
    # CHUNKING
    # -------------------------
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    texts = text_splitter.split_documents(all_docs)

    # -------------------------
    # EMBEDDINGS
    # -------------------------
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    # -------------------------
    # VECTOR DB
    # -------------------------
    db = FAISS.from_documents(texts, embeddings)

    # -------------------------
    # GROQ CLIENT
    # -------------------------
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])

    # -------------------------
    # DISPLAY CHAT HISTORY
    # -------------------------
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # -------------------------
    # CHAT INPUT
    # -------------------------
    query = st.chat_input("Ask something...")

    if query:
        # User message
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # -------------------------
        # RETRIEVAL
        # -------------------------
        results = db.similarity_search(query, k=5)

        context = ""
        sources = []

        for doc in results:
            context += doc.page_content + "\n\n"
            source = doc.metadata.get("source", "Unknown file")
            sources.append((source, doc.page_content[:200]))

        # -------------------------
        # PROMPT
        # -------------------------
        prompt = f"""
        You are an AI assistant.

        STRICT RULES:
        - Answer ONLY using the given context
        - Do NOT use outside knowledge
        - If answer is not found, say "Not found in document"

        Context:
        {context}

        Question:
        {query}

        Give a clear and accurate answer.
        """

        # -------------------------
        # LLM RESPONSE
        # -------------------------
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}]
        )

        answer = response.choices[0].message.content

        # Save response
        st.session_state.messages.append({"role": "assistant", "content": answer})

        with st.chat_message("assistant"):
            st.markdown(answer)

            # -------------------------
            # SOURCE DISPLAY
            # -------------------------
            st.markdown("### 📌 Sources Used:")
            for i, (src, preview) in enumerate(sources):
                st.markdown(f"**Source {i+1}: {src}**")
                st.code(preview)