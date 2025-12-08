from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class IngestionConfig:
    # Base paths
    base_data_dir: Path = Path("./data/collections")
    base_vector_store_path: Path = Path("./vector_store/chroma_db")

    # Default text splitting configurations
    default_chunk_size: int = 1000
    default_chunk_overlap: int = 200

    # Embedding model
    embedding_model: str = "qwen3-embedding:0.6b"

    # Supported file extensions
    supported_extensions: tuple = (".pdf", ".txt", ".md", ".docx", ".pptx", ".csv")

    # Batch processing
    batch_size: int = 100
