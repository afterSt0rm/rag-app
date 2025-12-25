# Retrieval-Augmented Generation (RAG) Using LangChain

## Overview

This application is a full-featured Retrieval-Augmented Generation (RAG) system built using LangChain. It enables users to ingest documents into a vector database, perform semantic searches, and generate contextually relevant answers to queries by leveraging the power of large language models (LLMs) combined with document retrieval.

The system uses **Ollama** for local LLM inference with models like DeepSeek-R1, **ChromaDB** as the vector store for document embeddings, and provides both a **FastAPI** backend and a **Streamlit** frontend for easy interaction. For RAG evaluation, the application integrates with the **Cerebras API** using an LLM-as-a-judge approach (gpt-oss-120b) to assess response quality.

## Features

- **Document Ingestion Pipeline**: Upload and process documents into a vector database with background task management
- **Semantic Search**: Perform intelligent searches across your document collection using vector embeddings
- **RAG Query Engine**: Get AI-generated answers grounded in your document context
- **Collection Management**: Create and manage multiple document collections
- **Evaluation Metrics**: Assess RAG response quality using RAGAS metrics (Faithfulness, Answer Relevancy, Context Precision) with LLM-as-a-judge via Cerebras API
- **Batch Evaluation**: Evaluate multiple RAG responses simultaneously
- **Real-time Task Monitoring**: Track ingestion task status and progress
- **LangFuse Integration**: Observability and tracing for LLM interactions
- **Streamlit UI**: User-friendly web interface for interacting with the RAG system
- **RESTful API**: Comprehensive API endpoints for programmatic access
- **Local LLM Support**: Run document ingestion, search, and RAG queries locally using Ollama
- **Cerebras Integration**: Cloud-based LLM-as-a-judge evaluation using gpt-oss-120b for accurate response assessment

## RAG System API Endpoints

![RAG System API](RAG_System_API.png)

- `/collections`: lists all the collections
- `/collections/{collection_name}`: creates a new collection
- `/ingest`: ingestion pipeline to ingest documents into a vector database
- `/ingest/tasks`: lists all the ingestion tasks status
- `/ingest/tasks/{task_id}`: get, cancel, update status of a particular ingestion task
- `/ingest/tasks/stats`: get detailed status of a particular ingestion task
- `/query`: query a particular collection
- `/collection/current`: set a current collection
- `/collection/current/info`: get current collection info
- `/search`: performs semantic search on documents in the vector database
- `/health`: performs health check of API
- `/evaluate`: evaluate a RAG response using RAGAS metrics.
- `/evalaute/batch`: evaluate multiple RAG responses in batch.
- `/evaluate/query`: evaluate a live RAG query

## Evaluation Metrics

1. **[Faithfulness](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/)**: Measures factual consistency between answer and context
2. **[Answer Relevancy](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/answer_relevance/)**: Measures how relevant the answer is to the question
3. **[Context Precision](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_precision/)**: Measures if relevant contexts are ranked higher

## How to Run
### Prerequites
- Ollama
- Python 3.10+
- Python virtual environment (recommended)
- LangFuse and Cerebras API Keys

### 1. Install Ollama

#### 1.1. Go to [Ollama](https://ollama.com/download/linux)

![Download_Ollama](Download_Ollama.png)

#### 1.2. Install Ollama

> Install ollama based on your system.

#### 1.3. Pull `deepseek-r1` and `qwen3-embedding` model

```bash
ollama pull deepseek-r1:8b
ollama pull qwen3-embedding:0.6b
```

#### 1.4. Run ollama server

```bash
ollama serve
```


### 2. Clone github repo

```bash
git clone https://github.com/afterSt0rm/rag-app.git
```

### 3. Create .env file

```bash
OLLAMA_BASE_URL=http://localhost:11434/
OLLAMA_LLM_MODEL=deepseek-r1
EMBEDDING_MODEL=qwen3-embedding:0.6b
CHROMA_PERSIST_DIR=./vector_store/chroma_db
API_BASE_URL=http://localhost:8000
LANGFUSE_SECRET_KEY=yourkey............
LANGFUSE_PUBLIC_KEY=yourkey............
LANGFUSE_BASE_URL=https://cloud.langfuse.com
CEREBRAS_API_KEY=yourkey............
CEREBRAS_LLM_MODEL=gpt-oss-120b
```

⚠️ Notes:

- Create a .env file in root directory after cloning the repository
- You need to create [Langfuse](https://cloud.langfuse.com/) account in order to create an API KEY
- You also need to create [Cerebras](https://cloud.cerebras.ai/) account and generate an API KEY
- Make sure to paste the generated langfuse api key in `LANGFUSE_SECRET_KEY` and `LANGFUSE_PUBLIC_KEY` in .env
- Make sure to paste the generated cerebras api key in `CEREBRAS_API_KEY` in .env


### 4. Create a python virtual environment

```bash
python -m venv .venv
```

### 5. Activate virtual environment

#### 5.1. On Windows:

```bash
.venv\Scripts\activate 
```

#### 5.2. On Linux/macOS:

```bash
source .venv/bin/activate 
```

### 6. Install requirements

```bash
pip install -r requirements.txt
```

### 7. Run FastAPI

```bash
uvicorn api.main:app --reload
```

### 8. Run Streamlit UI

```bash
streamlit run frontend/streamlit_app.py
```
