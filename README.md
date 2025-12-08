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
- Ollama (for embedding model)
- Python 3.10+
- Python virtual environment (recommended)
- .env file

### 1. Install Ollama

#### 1.1. Go to [Ollama](https://ollama.com/download/linux)

![Download_Ollama](Download_Ollama.png)

#### 1.2. Install Ollama

#### 1.3. Pull `qwen3-embedding` model

```bash
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

### 2. Create .env file

```bash
GOOGLE_API_KEY=yourkey...
```

### 3. Create a python virtual environment

```bash
python -m venv .venv
```

### 4. Activate virtual environment

#### 4.1. On Windows:

```bash
.venv\Scripts\activate 
```

#### 4.2. On Linux/macOS:

```bash
source .venv\bin\activate 
```

### 5. Install requirements

```bash
pip install -r requirements.txt
```

### 6. Run FastAPI

```bash
uvicorn api.main:app --reload
```

### 7. Run Streamlit UI

```bash
streamlit run frontend/streamlit_app.py
```
