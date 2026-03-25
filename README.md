# AI-Powered-Document-Question-Answering-System-using-Generative-AI
Build a Generative AI-based system that allows users to upload documents and ask questions, receiving accurate answers extracted from the document content.

# 🤖 AI-Powered Document Question Answering System using Generative AI

An end-to-end **Retrieval Augmented Generation (RAG) based Document Question Answering System** built using **Streamlit, LangChain, FAISS and HuggingFace LLMs**.

This application enables users to upload **PDF or TXT documents** and ask natural language questions.
The system retrieves the most relevant document chunks using vector similarity search and generates **context-aware answers** using a powerful open-source Large Language Model.

---

## 🚀 Key Features

✅ Upload and analyze **PDF / TXT documents**
✅ Intelligent **document chunking & semantic embedding**
✅ Fast **vector similarity search using FAISS**
✅ Accurate answer generation using **Llama-3 Instruct LLM**
✅ Modern interactive **Streamlit UI**
✅ Source chunk preview for **explainability**
✅ Adjustable **RAG parameters** (chunk size, overlap, top-K retrieval)
✅ Session-based conversational interaction

---

## 🧠 Models Used

### 🔹 Large Language Model (LLM)

* **Model Name:** `meta-llama/Meta-Llama-3-8B-Instruct`
* **Provider:** HuggingFace Inference API
* **Purpose:**

  * Generate final answers
  * Understand user questions
  * Perform reasoning based on retrieved document context
  * Produce structured and context-grounded responses

---

### 🔹 Embedding Model

* **Model Name:** `sentence-transformers/all-MiniLM-L6-v2`
* **Framework:** Sentence Transformers
* **Purpose:**

  * Convert document text chunks into dense vector embeddings
  * Enable semantic similarity search
  * Improve retrieval accuracy

---

### 🔹 Vector Database

* **Engine:** FAISS (Facebook AI Similarity Search)
* **Purpose:**

  * Store document embeddings efficiently
  * Perform fast nearest-neighbour search
  * Retrieve top-K most relevant chunks for answering

---

## ⚙️ System Architecture (RAG Pipeline)

1. User uploads document
2. Document is split into smaller chunks
3. Each chunk is converted into vector embeddings
4. Embeddings are stored in FAISS vector store
5. User asks a question
6. System retrieves most relevant chunks
7. Retrieved context is sent to LLM
8. LLM generates final grounded answer

---

## 🛠️ Tech Stack

* Python
* Streamlit
* LangChain
* HuggingFace Hub
* FAISS
* Sentence Transformers

---

## 📂 Project Structure

```
AI-Powered-Document-Question-Answering-System-using-Generative-AI/
│
├── rag_app.py
├── requirement.txt
├── README.md
├── test_rag.ipynb
└── .env   (not pushed to GitHub)
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone Repository

```
git clone https://github.com/souvikg420/AI-Powered-Document-Question-Answering-System-using-Generative-AI.git
cd AI-Powered-Document-Question-Answering-System-using-Generative-AI
```

### 2️⃣ Create Virtual Environment

```
python -m venv venv
```

Activate:

Windows:

```
venv\Scripts\activate
```

Mac/Linux:

```
source venv/bin/activate
```

### 3️⃣ Install Dependencies

```
pip install -r requirement.txt
```

---

## 🔐 Environment Variable Setup

Create `.env` file in root directory:

```
HF_TOKEN=your_huggingface_access_token
```

Get token from:
https://huggingface.co/settings/tokens

---

## ▶️ Run Application

```
streamlit run rag_app.py
```

Open in browser:

```
http://localhost:8501
```

---

## 💡 Use Cases

* AI Research Assistant
* Academic Document Analysis
* Policy / Legal Document QA
* Resume Analyzer
* Knowledge Base Chatbot
* Generative AI Learning Project

---

## 🔮 Future Enhancements

* Multi-document retrieval
* Persistent vector database
* Chat history memory
* Streaming LLM responses
* Cloud deployment (Streamlit Cloud / HuggingFace Spaces)
* Authentication & user sessions

---

## 👨‍💻 Author

**Souvik Ghosh**

If you found this project helpful ⭐ please consider starring the repository.

---
