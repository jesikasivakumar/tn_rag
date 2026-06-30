# 🌾 TN Agri Scheme AI Assistant

A Retrieval-Augmented Generation (RAG) chatbot for exploring Tamil Nadu agricultural welfare schemes — eligibility, subsidies, deadlines, and application details — grounded in your own indexed government documents, served through a multi-page Streamlit interface.

## ✨ Features

| Section | Status | Description |
|---|---|---|
| 🏠 Landing Page | ✅ Live | Hero banner, tagline, description, "Start Asking" CTA |
| 💬 AI Chat | ✅ Live (real RAG) | Chat interface backed by FAISS retrieval + OpenAI generation, with sources, confidence score, feedback, copy/download |
| 📋 Scheme Explorer | 🧪 Demo data | Browsable scheme cards by category |
| 🔍 Eligibility Checker | 🧪 Demo logic | Form-based matcher (placeholder rules) |
| 📄 Upload Documents | 🧪 UI only | File upload UI; indexing into FAISS not yet wired |
| 📚 Indexed Documents | 🧪 Demo data | Table of indexed files with mock scores |
| 📊 Dashboard | 🧪 Demo data | Charts for schemes, queries, feedback, districts |
| 📰 Latest Agriculture News | 🧪 Static | Sample notifications/circulars |
| ⭐ Saved Schemes | ✅ Live (session) | Bookmark schemes during your session |
| 🕒 Chat History | ✅ Live | Auto-grouped by Today / Yesterday / Last Week |
| ⚙ Settings | ✅ Live | Adjust embedding model, LLM, chunk size, top-k, temperature |
| ℹ About | ✅ Live | Tech stack summary |

> 🧪 = placeholder/demo data so the UI is fully explorable; swap in real data sources as your backend grows (see [Roadmap](#-roadmap--known-placeholders)).

## 🛠️ Tech Stack

- **Frontend:** Streamlit
- **Orchestration:** LangChain
- **Vector Store:** FAISS
- **Embeddings:** Sentence Transformers (`all-MiniLM-L6-v2`)
- **LLM:** OpenAI (`gpt-4o-mini` by default, via `langchain-openai`)

## 📂 Project Structure

```
.
├── app.py                 # Main Streamlit application
├── requirements.txt        # Python dependencies
├── .env                     # API keys (you create this — see below)
└── faiss_index/
    ├── index.faiss          # Pre-built FAISS vector index
    └── index.pkl
```

> **Note:** This app expects a pre-built FAISS index at `faiss_index/index.*`. It does not build the index from scratch — see [Building the FAISS Index](#-building-the-faiss-index) below.

## 🚀 Getting Started

### 1. Clone and install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set up environment variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your-openai-api-key-here
```

### 3. Provide a FAISS index

Place a pre-built FAISS index at `faiss_index/index.faiss` and `faiss_index/index.pkl`, embedded with the same model configured in `app.py` (`sentence-transformers/all-MiniLM-L6-v2` by default). See [Building the FAISS Index](#-building-the-faiss-index) if you don't have one yet.

### 4. Run the app

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

## 🧩 Building the FAISS Index

This app loads an existing index rather than creating one. A minimal script to build one from your scheme PDFs/text:

```python
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

docs = PyPDFDirectoryLoader("source_documents/").load()
chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(docs)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = FAISS.from_documents(chunks, embeddings)
vector_db.save_local(folder_path="faiss_index", index_name="index")
```

Adjust `chunk_size` to match what you configure later in the Settings page.

## ⚙️ Configuration

Editable at runtime from the **⚙ Settings** page (held in `st.session_state`, not persisted across restarts):

- Embedding model name
- LLM model name
- Chunk size
- Top-K retrieval count
- Temperature

Changing the embedding model, LLM model, or Top-K re-initializes the RAG pipeline (`@st.cache_resource`) on next interaction.

## 🗺️ Roadmap / Known Placeholders

The following are stubbed in the UI with inline notes in `app.py` (`# NOTE:` comments) so they're easy to find and wire up:

- **Document indexing on upload** — currently only tracks filenames; needs parsing + chunking + `vector_db.add_documents(...)`
- **Eligibility Checker** — uses a simple placeholder rule, not a real eligibility engine
- **Scheme Explorer / News / Dashboard data** — static/randomized; replace with a real database or scraped feed
- **Confidence score** — randomized; replace with a real retrieval-similarity or LLM-based score
- **Voice input** — needs a mic-capture component (e.g. `streamlit-mic-recorder`)
- **Text-to-speech** — needs a TTS engine (e.g. `gTTS`, browser `SpeechSynthesis`)
- **Tamil (தமிழ்) language** — toggle exists; UI strings aren't translated yet

## 📜 License

Jesika shree Sivakumar

---

Built with ❤️ using Python • Streamlit • LangChain • RAG • FAISS • LLM
