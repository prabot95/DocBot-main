
# 📄 PDF Chatbot with RAG (LangChain + Streamlit)

A **Retrieval-Augmented Generation (RAG)** application that allows users to chat with their PDF documents. This project uses **LangChain**, **FAISS** for vector storage, and integrates **HuggingFace** and **Groq** LLMs to provide accurtae answers based on document context.



## 🚀 Features

* **Document Ingestion:** Loads and processes PDF files from a local directory.
* **Text Splitting:** Breaks down large documents into manageable chunks using `RecursiveCharacterTextSplitter`.
* **Vector Embeddings:** Uses `sentence-transformers/all-MiniLM-L6-v2` to create semantic embeddings.
* **Vector Store:** Stores embeddings locally using **FAISS** for fast similarity search.
* **Multi-Interface:**
    * **CLI Mode:** Test retrieval and generation via the terminal.
    * **Web UI:** A user-friendly chat interface built with **Streamlit**.
* **LLM Integration:** Supports **HuggingFace Endpoints** (Mistral) and 

---

## 📂 Project Structure

```text
├── data/                   # Directory to store input PDF files
├── vectorstore/            # Directory where FAISS index is saved
├── memory_llm.py           # Script to ingest PDFs and create vector store
├── connect_memory_llm.py   # Script to test RAG pipeline via CLI
├── docbot.py               # Streamlit application for the Chatbot UI
├── requirements.txt        # Python dependencies
└── .env                    # Environment variables (API Keys)

```

---

## 🛠️ Technologies Used

* **Python 3.10+**
* **LangChain** (Framework)
* **Streamlit** (Frontend)
* **FAISS** (Vector Database)
* **HuggingFace** (Embeddings & LLM)
    
* **PDFPlumber** (Document Loading)

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd <repository-folder>

```

### 2. Create a Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

```

### 3. Install Dependencies

Create a `requirements.txt` file (if not present) with the following content, then install:

```text
langchain
langchain-community
langchain-huggingface
langchain-groq
faiss-cpu
pdfplumber
streamlit
python-dotenv
huggingface_hub

```

Run command:

```bash
pip install -r requirements.txt

```

---

## 🔑 Configuration

Create a `.env` file in the root directory and add your API keys:

```env
HF_TOKEN=your_huggingface_access_token
GROQ_API_KEY=your_groq_api_key

```

* **HF_TOKEN**: Get it from [HuggingFace Settings](https://huggingface.co/settings/tokens).
* **GROQ_API_KEY**: Get it from [Groq Console](https://console.groq.com/).

---

## 📖 Usage Guide

### Step 1: Ingest Data

Place your PDF files into the `data/` folder. Then, run the ingestion script to create the vector database.

```bash
python memory_llm.py

```

*This will create a `vectorstore/db_faiss` directory containing your embeddings.*

### Step 2: Test via CLI (Optional)

To test if the retrieval is working correctly without the web UI:

```bash
python connect_memory_llm.py

```

### Step 3: Run the Chatbot App

Launch the Streamlit web interface:

```bash
streamlit run docbot.py

```

Open your browser at `http://localhost:8501` to start chatting with your PDFs!

---

## 🧠 How It Works

1. **Ingestion (`memory_llm.py`):** The script loads PDFs, splits text into 500-character chunks, converts them into vectors using HuggingFace embeddings, and saves them to a local FAISS index.
2. **Retrieval:** When a user asks a question, the system searches the FAISS index for the top 3 most similar document chunks.
3. **Generation (`docbot.py`):** The retrieved chunks + the user's question are sent to the LLM (via Groq API). The LLM generates a concise answer based strictly on the provided context.

---

## ⚠️ Notes

* **Model Selection:** The `docbot.py` is currently configured to use Groq. Ensure your `.env` file has a valid `GROQ_API_KEY`.
* **Warnings:** You may see "dangerous deserialization" warnings from FAISS. This is normal when loading local files you created yourself; the code includes `allow_dangerous_deserialization=True` to handle this.

