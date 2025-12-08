import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_community.document_loaders import (
    CSVLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ingestion.config import IngestionConfig


class CollectionManager:
    """Manages collections in the data directory"""

    def __init__(self, base_data_dir: Path):
        self.base_data_dir = base_data_dir
        self.base_data_dir.mkdir(parents=True, exist_ok=True)

    def list_collections(self) -> List[str]:
        """List all existing collections"""
        if not self.base_data_dir.exists():
            return []

        collections = []
        for item in self.base_data_dir.iterdir():
            if item.is_dir():
                collections.append(item.name)

        return sorted(collections)

    def create_collection(self, collection_name: str) -> Path:
        """Create a new collection directory"""
        collection_path = self.base_data_dir / collection_name
        collection_path.mkdir(parents=True, exist_ok=True)
        return collection_path

    def get_collection_path(self, collection_name: str) -> Path:
        """Get path for a collection"""
        return self.base_data_dir / collection_name

    def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection and its data"""
        collection_path = self.get_collection_path(collection_name)
        if collection_path.exists():
            shutil.rmtree(collection_path)
            return True
        return False


class IngestionPipeline:
    def __init__(self, config: Optional[IngestionConfig] = None):
        self.config = config or IngestionConfig()
        self.embeddings = OllamaEmbeddings(model=self.config.embedding_model)
        self.collection_manager = CollectionManager(self.config.base_data_dir)

    def get_loader(self, file_path: Path):
        """Get appropriate loader based on file extension"""
        suffix = file_path.suffix.lower()

        loader_map = {
            ".pdf": PyPDFLoader,
            ".txt": TextLoader,
            ".md": UnstructuredMarkdownLoader,
            ".docx": UnstructuredWordDocumentLoader,
            ".pptx": UnstructuredPowerPointLoader,
            ".csv": CSVLoader,
        }

        if suffix in loader_map:
            if suffix == ".csv":
                return loader_map[suffix](str(file_path), encoding="utf-8")
            return loader_map[suffix](str(file_path))
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

    def load_documents(self, file_paths: List[Path]) -> List:
        """Load documents from multiple files"""
        all_docs = []

        for file_path in file_paths:
            try:
                loader = self.get_loader(file_path)
                documents = loader.load()

                # Add metadata about source file
                for doc in documents:
                    doc.metadata["source"] = str(file_path)
                    doc.metadata["filename"] = file_path.name
                    doc.metadata["file_type"] = file_path.suffix

                all_docs.extend(documents)
                print(f"✓ Loaded {len(documents)} documents from {file_path.name}")

            except Exception as e:
                print(f"✗ Failed to load {file_path}: {e}")

        return all_docs

    def split_documents(
        self, documents: List, chunk_size: int, chunk_overlap: int
    ) -> List:
        """Split documents into chunks with configurable parameters"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )

        chunks = text_splitter.split_documents(documents)
        print(f"Split {len(documents)} documents into {len(chunks)} chunks")
        return chunks

    def create_vector_store(
        self, chunks: List, collection_name: str, chunk_size: int, chunk_overlap: int
    ) -> Chroma:
        """Create and persist vector store for a specific collection"""

        # Collection-specific vector store path
        collection_vector_path = self.config.base_vector_store_path / collection_name
        collection_vector_path.mkdir(parents=True, exist_ok=True)

        # Add collection metadata to all chunks
        for chunk in chunks:
            chunk.metadata["collection"] = collection_name
            chunk.metadata["chunk_size"] = chunk_size
            chunk.metadata["chunk_overlap"] = chunk_overlap

        # Create vector store
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=str(collection_vector_path),
            collection_name=collection_name,
        )

        # Persist to disk
        vector_store.persist()
        print(f"✓ Vector store created for collection '{collection_name}'")
        print(f"  Chunks: {len(chunks)}, Size: {chunk_size}, Overlap: {chunk_overlap}")
        print(f"  Saved to: {collection_vector_path}")

        return vector_store

    def ingest_to_collection(
        self,
        collection_name: str,
        file_paths: List[Path],
        chunk_size: int = None,
        chunk_overlap: int = None,
    ) -> Dict[str, Any]:
        """Ingest files to a specific collection"""

        if not file_paths:
            return {
                "success": False,
                "error": "No files provided",
                "message": "No files provided for ingestion",  # Add message field
                "collection": collection_name,
            }

        # Use provided parameters or defaults
        chunk_size = chunk_size or self.config.default_chunk_size
        chunk_overlap = chunk_overlap or self.config.default_chunk_overlap

        # Save files to collection directory
        collection_path = self.collection_manager.get_collection_path(collection_name)
        saved_files = []

        for file_path in file_paths:
            dest_path = collection_path / file_path.name
            shutil.copy2(file_path, dest_path)
            saved_files.append(str(dest_path))

        # Process files
        try:
            documents = self.load_documents([Path(f) for f in saved_files])
            chunks = self.split_documents(documents, chunk_size, chunk_overlap)
            vector_store = self.create_vector_store(
                chunks, collection_name, chunk_size, chunk_overlap
            )

            return {
                "success": True,
                "message": f"Successfully ingested {len(saved_files)} files into collection '{collection_name}'",
                "collection": collection_name,
                "files_processed": len(saved_files),
                "chunks_created": len(chunks),
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "collection_path": str(collection_path),
                "vector_store_path": str(
                    self.config.base_vector_store_path / collection_name
                ),
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "collection": collection_name,
                "message": f"Ingestion failed: {str(e)}",  # Add message field
            }

    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """Get statistics for a collection"""
        collection_path = self.collection_manager.get_collection_path(collection_name)

        if not collection_path.exists():
            return {"exists": False}

        # Count files
        file_count = 0
        supported_exts = set(self.config.supported_extensions)

        for ext in supported_exts:
            file_count += len(list(collection_path.glob(f"*{ext}")))

        # Check vector store
        vector_store_path = self.config.base_vector_store_path / collection_name

        vector_exists = vector_store_path.exists()
        vector_count = 0

        if vector_exists:
            try:
                vector_store = Chroma(
                    persist_directory=str(vector_store_path),
                    embedding_function=self.embeddings,
                    collection_name=collection_name,
                )
                vector_count = vector_store._collection.count()
            except:
                vector_count = 0

        return {
            "exists": True,
            "name": collection_name,
            "file_count": file_count,
            "vector_exists": vector_exists,
            "vector_count": vector_count,
            "data_path": str(collection_path),
            "vector_path": str(vector_store_path),
        }

    def list_all_collection_stats(self) -> List[Dict[str, Any]]:
        """Get statistics for all collections"""
        collections = self.collection_manager.list_collections()
        stats = []

        for collection in collections:
            stats.append(self.get_collection_stats(collection))

        return stats


# Singleton instance
_ingestion_pipeline = None


def get_ingestion_pipeline(
    config: Optional[IngestionConfig] = None,
) -> IngestionPipeline:
    """Get or create ingestion pipeline instance"""
    global _ingestion_pipeline
    if _ingestion_pipeline is None:
        _ingestion_pipeline = IngestionPipeline(config)
    return _ingestion_pipeline
