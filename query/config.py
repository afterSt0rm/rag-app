import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


@dataclass
class QueryConfig:
    # Model configurations
    llm_model: str = "gemini-2.5-flash"
    embedding_model: str = "qwen3-embedding:0.6b"
    temperature: float = 0.1

    # Retrieval configurations
    top_k: int = 4

    # Reranking configurations
    enable_reranking: bool = False
    rerank_top_n: int = 3

    # Vector store configurations
    base_vector_store_path: str = "./vector_store/chroma_db"

    # Current collection
    current_collection: Optional[str] = None
