# Retrieval-Augemented Generation (RAG) Using LangChain

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
git clone git@github.com:afterSt0rm/rag-app.git
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
