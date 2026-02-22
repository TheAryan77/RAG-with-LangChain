# 📚 RAG Pipeline - Retrieval-Augmented Generation System

A production-ready RAG (Retrieval-Augmented Generation) pipeline for intelligent document question-answering using LangChain, ChromaDB, and Groq LLM.

## 🌟 Features

- **PDF Document Processing**: Automatically load and process PDF files from directories
- **Intelligent Text Chunking**: Split documents into optimal chunks for better retrieval
- **Vector Storage**: ChromaDB-powered persistent vector database with cosine similarity
- **Semantic Search**: Sentence-Transformers embeddings for accurate document retrieval
- **Multiple RAG Pipelines**:
  - Simple RAG: Basic question-answering
  - Enhanced RAG: With confidence scores and source citations
  - Advanced RAG: Streaming, query history, and answer summarization
- **LLM Integration**: Groq API for fast, high-quality answer generation

## 🏗️ Project Structure

```
RAG/
├── data/                      # Data storage
│   ├── text_files/           # Text documents
│   └── vector_store/         # ChromaDB persisted vectors
├── notebooks/                 # Jupyter notebooks for experimentation
│   ├── document.ipynb        # Document loading examples
│   ├── pdf_loader.ipynb      # PDF processing
│   └── rag_pipeline.ipynb    # Complete RAG pipeline
├── src/                       # Source code modules
│   ├── data_loader.py        # Document loading utilities
│   ├── embeddings.py         # Embedding generation
│   ├── vectorstore.py        # Vector database management
│   └── search.py             # Retrieval and search logic
├── main.py                    # Main application entry point
├── requirements.txt           # Python dependencies
├── pyproject.toml            # Project configuration
└── .env                       # Environment variables (not tracked)
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Git
- Groq API Key ([Get one here](https://console.groq.com))

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd RAG
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   
   Or using `uv` (faster):
   ```bash
   uv pip install -r requirements.txt
   ```

4. **Set up environment variables**
   
   Create a `.env` file in the root directory:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

### Usage

#### 1. Basic Python Script

```python
from src.data_loader import process_all_pdfs
from src.embeddings import EmbeddingManager
from src.vectorstore import VectorStore
from src.search import RAGRetriever

# Load documents
documents = process_all_pdfs("./data")

# Generate embeddings
embedding_manager = EmbeddingManager()
embeddings = embedding_manager.generate_embeddings([doc.page_content for doc in documents])

# Store in vector database
vectorstore = VectorStore()
vectorstore.add_documents(documents, embeddings)

# Query
retriever = RAGRetriever(vectorstore, embedding_manager)
results = retriever.retrieve("What is ISO?", top_k=3)
```

#### 2. Using Jupyter Notebooks

Launch Jupyter and explore the notebooks:
```bash
jupyter notebook notebooks/rag_pipeline.ipynb
```

The notebook contains:
- Complete RAG pipeline setup
- PDF processing examples
- Simple, Enhanced, and Advanced RAG implementations
- Interactive query testing

## 📋 Components

### Data Loader
Processes PDF files from directories and extracts text with metadata.

### Embedding Manager
Uses `sentence-transformers` (all-MiniLM-L6-v2) to generate 384-dimensional embeddings.

### Vector Store
ChromaDB-based persistent storage with cosine similarity for efficient retrieval.

### RAG Retriever
Handles query processing and returns ranked, relevant document chunks.

### RAG Pipelines

1. **Simple RAG**: Basic retrieval + LLM generation
2. **Enhanced RAG**: Adds confidence scores and source attribution
3. **Advanced RAG**: Query history, streaming, and summarization

## 🔧 Configuration

### Embedding Model
Change the embedding model in `EmbeddingManager`:
```python
embedding_manager = EmbeddingManager(model_name="all-mpnet-base-v2")
```

### Chunking Strategy
Adjust chunk size and overlap in text splitting:
```python
split_documents(documents, chunk_size=1000, chunk_overlap=200)
```

### LLM Model
Configure Groq model:
```python
llm = ChatGroq(
    groq_api_key=api_key,
    model_name="openai/gpt-oss-120b",  # or other Groq models
    temperature=0.1,
    max_tokens=1024
)
```

## 📊 Example Output

```python
query = "What is ISO?"
result = rag_advanced(query, rag_retriever, llm, top_k=3)

# Output:
{
    'answer': 'ISO stands for International Organization for Standardization...',
    'sources': [
        {'source': 'document.pdf', 'page': 5, 'score': 0.89},
        {'source': 'guide.pdf', 'page': 12, 'score': 0.85}
    ],
    'confidence': 0.89
}
```

## 🛠️ Development

### Running Tests
```bash
pytest tests/
```

### Code Formatting
```bash
black src/
ruff check src/
```

## 📦 Dependencies

- **LangChain**: Document processing and RAG orchestration
- **ChromaDB**: Vector database for embeddings
- **Sentence Transformers**: Embedding generation
- **PyPDF/PyMuPDF**: PDF parsing
- **Groq**: LLM API for answer generation
- **Python-dotenv**: Environment variable management

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- LangChain for the RAG framework
- ChromaDB for vector storage
- Groq for fast LLM inference
- Sentence Transformers for embeddings

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Built with ❤️ for intelligent document search and question-answering**