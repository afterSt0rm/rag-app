import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain_classic.schema import Document
from langchain_classic.schema.output_parser import StrOutputParser
from langchain_classic.schema.runnable import RunnablePassthrough
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import OllamaEmbeddings
from langfuse import Langfuse, get_client, observe
from langfuse.langchain import CallbackHandler

from query.config import QueryConfig

load_dotenv()

Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_BASE_URL"),
)

# Get the configured client instance
langfuse = get_client()

# Initialize the Langfuse handler
langfuse_handler = CallbackHandler()


class QueryPipeline:
    def __init__(self, config: Optional[QueryConfig] = None):
        self.config = config or QueryConfig()

        # Initialize embeddings and LLM
        self.embeddings = OllamaEmbeddings(model=self.config.embedding_model)
        self.llm = ChatGoogleGenerativeAI(
            model=self.config.llm_model, temperature=self.config.temperature
        )

        # Initialize vector store (will be set when collection is selected)
        self.vector_store = None
        self.retriever = None
        self.chain = None

        # Load available collections
        self.available_collections = self._load_available_collections()

        # If a collection is specified in config, load it
        if self.config.current_collection:
            self.switch_collection(self.config.current_collection)

    def _load_available_collections(self) -> List[str]:
        """Load all available collections from vector store directory"""
        base_path = Path(self.config.base_vector_store_path)

        if not base_path.exists():
            return []

        collections = []
        for item in base_path.iterdir():
            if item.is_dir():
                # Check if it's a valid Chroma collection
                chroma_file = item / "chroma.sqlite3"
                if chroma_file.exists():
                    collections.append(item.name)

        return sorted(collections)

    def set_top_k(self, top_k: int):
        """Set the number of documents to retrieve"""
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer")

        self.config.top_k = top_k

        if self.vector_store:
            # Update retriever with new top_k
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": top_k},
            )
            # Rebuild chain with new retriever
            self.chain = self._build_chain()
            print(f"✓ Updated top_k to: {top_k}")

    def switch_collection(self, collection_name: str) -> bool:
        """Switch to a different collection"""
        try:
            if collection_name not in self.available_collections:
                # Try to refresh available collections
                self.available_collections = self._load_available_collections()

                if collection_name not in self.available_collections:
                    return False

            # Build collection-specific vector store path
            collection_path = Path(self.config.base_vector_store_path) / collection_name

            # Initialize vector store for this collection
            self.vector_store = Chroma(
                persist_directory=str(collection_path),
                embedding_function=self.embeddings,
                collection_name=collection_name,
            )

            # Update retriever with current top_k
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={
                    "k": self.config.top_k,
                },
            )

            # Rebuild chain
            self.chain = self._build_chain()

            # Update config
            self.config.current_collection = collection_name

            print(
                f"✓ Switched to collection: {collection_name} (top_k={self.config.top_k})"
            )
            return True

        except Exception as e:
            print(f"✗ Failed to switch collection: {e}")
            return False

    def _build_chain(self):
        """Build the RAG chain for the current collection"""

        def format_docs(docs: List[Document]) -> str:
            """Format documents for context"""
            if not docs:
                return "No relevant documents found."

            formatted = []
            for i, doc in enumerate(docs, 1):
                source = doc.metadata.get(
                    "filename", doc.metadata.get("source", "Unknown")
                )
                collection = doc.metadata.get(
                    "collection", self.config.current_collection or "Unknown"
                )
                content = doc.page_content
                formatted.append(
                    f"[Document {i} | Collection: {collection} | Source: {source}]\n{content}\n"
                )
            return "\n".join(formatted)

        # Create prompt template
        template = """You are a helpful assistant. Use the following context to answer the user's question.

        Context from documents (Collection: {collection}):
        {context}

        Question: {question}

        Answer the question based on the context above. If the context doesn't contain relevant information, say "I don't have enough information to answer this question based on the provided documents."

        If the question is conversational continue normally.

        Provide a comprehensive answer with citations to the source documents when possible. Mention which collection the information comes from.

        Answer:"""

        prompt = ChatPromptTemplate.from_messages(
            [("system", template), ("human", "{question}")]
        )

        # Create the chain
        chain = (
            {
                "context": self.retriever | format_docs,
                "collection": lambda x: self.config.current_collection or "Unknown",
                "question": RunnablePassthrough(),
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

        return chain

    @observe()
    def query(
        self,
        question: str,
        collection_name: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Execute a query, optionally specifying a collection and number of results"""
        # Switch collection if specified
        if collection_name and collection_name != self.config.current_collection:
            if not self.switch_collection(collection_name):
                return {
                    "question": question,
                    "answer": f"Error: Collection '{collection_name}' not found or cannot be loaded.",
                    "sources": [],
                    "doc_count": 0,
                    "collection": collection_name,
                    "error": f"Collection '{collection_name}' not available",
                }

        if not self.vector_store or not self.chain:
            return {
                "question": question,
                "answer": "Error: No collection loaded. Please select a collection first.",
                "sources": [],
                "doc_count": 0,
                "error": "No collection loaded",
            }

        # Handle temporary top_k override
        original_top_k = None
        original_retriever = None
        original_chain = None

        if top_k is not None and top_k != self.config.top_k:
            original_top_k = self.config.top_k
            original_retriever = self.retriever
            original_chain = self.chain

            try:
                # Temporarily update top_k
                self.retriever = self.vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": top_k},
                )
                self.chain = self._build_chain()
                print(f" temporarily using top_k={top_k}")
            except Exception as e:
                # Restore original state if something goes wrong
                self.config.top_k = original_top_k
                self.retriever = original_retriever
                self.chain = original_chain
                return {
                    "question": question,
                    "answer": f"Error setting top_k: {str(e)}",
                    "sources": [],
                    "doc_count": 0,
                    "collection": self.config.current_collection,
                    "error": str(e),
                }

        try:
            # Trace the document retrieval
            with langfuse.start_as_current_observation(
                as_type="retriever",
                name="retrieve_documents",
                input=question,
            ) as span:
                docs = self.retriever.invoke(question)
                span.update(output=docs)

            # # Get relevant documents
            # docs = self.retriever.invoke(question)

            # Generate answer
            answer = self.chain.invoke(
                question, config={"callbacks": [langfuse_handler]}
            )

            # Prepare sources
            sources = []
            for doc in docs:
                sources.append(
                    {
                        "content": doc.page_content[:300] + "...",
                        "source": doc.metadata.get(
                            "filename", doc.metadata.get("source", "Unknown")
                        ),
                        "collection": doc.metadata.get(
                            "collection", self.config.current_collection or "Unknown"
                        ),
                        "score": doc.metadata.get("score", 0.0),
                    }
                )

            result = {
                "question": question,
                "answer": answer,
                "sources": sources,
                "doc_count": len(docs),
                "collection": self.config.current_collection,
                "top_k_used": top_k if top_k is not None else self.config.top_k,
            }

            return result

        except Exception as e:
            return {
                "question": question,
                "answer": f"Error processing query: {str(e)}",
                "sources": [],
                "doc_count": 0,
                "collection": self.config.current_collection,
                "error": str(e),
            }
        finally:
            # Restore original state if we temporarily changed top_k
            if original_top_k is not None:
                self.config.top_k = original_top_k
                self.retriever = original_retriever
                self.chain = original_chain
                print(f" restored top_k to {original_top_k}")

    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the current collection"""
        if not self.vector_store:
            return {
                "current_collection": None,
                "loaded": False,
                "available_collections": self.available_collections,
                "current_top_k": self.config.top_k
                if hasattr(self.config, "top_k")
                else None,
            }

        try:
            collection = self.vector_store._collection
            return {
                "current_collection": self.config.current_collection,
                "loaded": True,
                "document_count": collection.count(),
                "available_collections": self.available_collections,
                "current_top_k": self.config.top_k,
            }
        except:
            return {
                "current_collection": self.config.current_collection,
                "loaded": False,
                "available_collections": self.available_collections,
                "current_top_k": self.config.top_k,
            }

    def similarity_search(self, query: str, k: Optional[int] = None) -> List[Document]:
        """Perform similarity search without LLM generation"""
        if not self.vector_store:
            raise ValueError("No collection loaded")

        # Use provided k or default to config top_k
        search_k = k if k is not None else self.config.top_k
        return self.vector_store.similarity_search(query, k=search_k)


# Singleton instance with dynamic configuration
_query_pipelines = {}


def get_query_pipeline(
    collection_name: Optional[str] = None, config: Optional[QueryConfig] = None
) -> QueryPipeline:
    """Get or create query pipeline instance for a specific collection"""
    global _query_pipelines

    # Create a key for this configuration
    config_key = collection_name or "default"

    if config_key not in _query_pipelines:
        if config is None:
            config = QueryConfig()

        if collection_name:
            config.current_collection = collection_name

        _query_pipelines[config_key] = QueryPipeline(config)

    return _query_pipelines[config_key]


def query_rag(
    question: str, collection_name: Optional[str] = None, top_k: Optional[int] = None
) -> Dict[str, Any]:
    """Convenience function for querying with collection selection and top_k parameter"""
    pipeline = get_query_pipeline(collection_name)
    return pipeline.query(question, collection_name, top_k)
