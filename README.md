# 📄 DocQuery AI: Intelligent PDF Analysis System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-🦜-black)](https://www.langchain.com/)
[![Google Gemini](https://img.shields.io/badge/Google%20Gemini-1.5%20Flash-4285F4?logo=google&logoColor=white)](https://deepmind.google/technologies/gemini/)
[![FAISS](https://img.shields.io/badge/Vector%20DB-FAISS-green)](https://github.com/facebookresearch/faiss)

**DocQuery AI** is a production-grade Retrieval-Augmented Generation (RAG) pipeline designed for deep interaction with PDF documents. It leverages state-of-the-art Large Language Models (LLMs) and vector similarity search to provide precise, context-aware answers to complex queries, significantly reducing manual document review time.

---

## 🌟 Key Features

- **🚀 Instant Indexing:** High-performance PDF parsing and chunking for immediate document readiness.
- **🧠 Contextual Intelligence:** Powered by **Google Gemini 1.5 Flash**, ensuring high-fidelity responses based strictly on document content.
- **💬 Conversational Memory:** Maintains context across multiple queries, allowing for fluid, natural follow-up questions.
- **🔍 Semantic Search:** Utilizes **FAISS** vector embeddings for high-speed, relevant context retrieval.
- **💎 Premium UI:** A modern, glassmorphic Streamlit interface designed for an elite user experience.
- **📊 Real-time Insights:** Live monitoring of processing time, chunk counts, and system performance.

---

## 🛠️ Technology Stack

| Category | Technology |
| :--- | :--- |
| **Frontend** | Streamlit (Custom CSS/HTML) |
| **LLM** | Google Gemini 1.5 Flash |
| **Orchestration** | LangChain |
| **Vector Engine** | FAISS (Facebook AI Similarity Search) |
| **Embeddings** | Google Generative AI Embeddings |
| **Parser** | PyPDF |
| **Styling** | Vanilla CSS & Markdown |

---

## 📐 System Architecture

The application follows a standard RAG (Retrieval-Augmented Generation) workflow:

1.  **Ingestion:** PDF is uploaded and parsed using `PyPDF`.
2.  **Chunking:** Document is split into semantic chunks with overlap to preserve context.
3.  **Embedding:** Chunks are converted into high-dimensional vectors via `gemini-embedding-001`.
4.  **Indexing:** Vectors are stored in a `FAISS` local index.
5.  **Retrieval:** User queries trigger a similarity search to find the top-K relevant chunks.
6.  **Augmentation:** Context is injected into a specialized system prompt.
7.  **Generation:** `Gemini 1.5 Flash` generates a grounded, professional response.

---

## 🚀 Installation & Setup

### Prerequisites

- Python 3.10 or higher
- A Google Cloud Project with Gemini API enabled

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/DocQuery-RAG.git
cd DocQuery-RAG
```

### 2. Set Up Virtual Environment
```bash
python -m venv venv
# Windows
.\venv\Scripts\activate
# Linux/macOS
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Environment Configuration
Create a `.env` file in the root directory and add your API key:
```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

### 5. Launch the Application
```bash
streamlit run app.py
```

---

## 📂 Project Structure

```text
├── .env                # API Configuration
├── app.py              # Main Streamlit Application
├── requirements.txt    # Project Dependencies
├── utils/
│   ├── pdf_processor.py # PDF Parsing & Chunking Logic
│   ├── vector_store.py  # FAISS Index Management
│   └── chat_engine.py   # LangChain & Gemini Integration
└── README.md           # Project Documentation
```

---

## 💡 Usage Guide

1.  **Upload:** Drag and drop your PDF into the sidebar.
2.  **Process:** Click **"Process Document"** to initialize the vector index.
3.  **Query:** Type your question in the chat input.
4.  **Analyze:** View the AI's response, formatted in clean Markdown with key highlights.
5.  **Clear:** Use the "Clear Chat History" button to start a fresh analysis session.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an Issue for any feature requests or bug reports.



---

<p align="center">
  Developed with ❤️ for Intelligent Document Analysis
</p>