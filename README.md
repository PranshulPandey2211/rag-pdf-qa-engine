## Local PDF Q&A Engine (Offline)

This project is a lightweight, **fully offline** Retrieval-Augmented Generation (RAG) system. It allows users to chat with PDF documents locally without requiring any external API keys (like OpenAI or Anthropic).

---

### Features
* **Private & Local:** Runs entirely on your CPU using local embeddings and a quantized LLM.
* **Persistent Storage:** Uses **ChromaDB** to store document chunks, so you only process PDFs once.
* **Intelligent Retrieval:** Uses `all-MiniLM-L6-v2` for high-quality semantic search.
* **Performance Tracking:** Real-time query response time monitoring.

---

### Tech Stack
* **Orchestration:** LangChain
* **Vector Database:** ChromaDB
* **Embeddings:** Sentence-Transformers (`all-MiniLM-L6-v2`)
* **LLM:** Google FLAN-T5 Small (via Hugging Face Pipelines)
* **Frontend:** Streamlit

---

### Installation & Setup

1. **Clone the repository:**
   ```bash
   cd searchable_pdf_qna
   ```

2. **Install dependencies:**
   ```bash
   pip install langchain langchain-community langchain-text-splitters pypdf chromadb sentence-transformers transformers streamlit
   ```

3. **Prepare your data:**
   * Create a folder named `data`.
   * Drop your `.pdf` files into the `data/` folder.

4. **Run the Application:**
   ```bash
   streamlit run app.py
   ```

---

### How It Works
1. **Ingestion:** The system loads PDFs from the `/data` folder, splits them into 700-character chunks, and converts them into vector embeddings.
2. **Retrieval:** When you ask a question, the system searches ChromaDB for the top 2 most relevant text snippets.
3. **Generation:** The context snippets and your question are passed to the **FLAN-T5** model to generate a concise, context-aware answer.