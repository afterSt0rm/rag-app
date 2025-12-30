import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.tools import tool
from langchain_ollama import OllamaEmbeddings

load_dotenv()

# Configuration from environment
RAG_API_URL = os.getenv("RAG_API_URL", "http://localhost:8000")
VECTOR_STORE_BASE_PATH = os.getenv("VECTOR_STORE_PATH", "./data/vector_store")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "qwen3-embedding")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
API_TIMEOUT = int(os.getenv("API_TIMEOUT", "60"))


@tool
def query_rag_endpoint(
    question: str,
    collection_name: str,
    top_k: int = 4,
) -> Dict[str, Any]:
    """
    Query the full RAG pipeline via FastAPI endpoint for comprehensive answers.

    This tool calls the existing /query endpoint which performs retrieval
    and LLM generation internally, returning a synthesized answer with citations.

    Use this tool when:
    - You need a comprehensive, synthesized answer from documents
    - The question requires complex reasoning over multiple document sections
    - You want a complete answer with source citations
    - You're querying a single collection and want the RAG system to handle generation

    Args:
        question: The question to ask the RAG system. Be specific and clear.
        collection_name: Name of the document collection to query (e.g., "research_papers", "deepseek").
        top_k: Number of documents to retrieve (default: 4, max: 20).

    Returns:
        Dictionary containing:
        - answer: The generated answer from the RAG system
        - sources: List of source documents with content and metadata
        - doc_count: Number of documents retrieved
        - collection: Name of the collection queried
        - error: Error message if the query failed
    """
    try:
        # Validate top_k
        top_k = max(1, min(top_k, 20))

        # Make request to RAG API
        response = requests.post(
            f"{RAG_API_URL}/query",
            json={
                "question": question,
                "collection_name": collection_name,
                "top_k": top_k,
            },
            timeout=API_TIMEOUT,
        )
        response.raise_for_status()
        result = response.json()

        return {
            "answer": result.get("answer", ""),
            "sources": result.get("sources", []),
            "doc_count": result.get("doc_count", 0),
            "collection": result.get("collection", collection_name),
            "processing_time": result.get("processing_time"),
            "success": True,
        }

    except requests.exceptions.Timeout:
        return {
            "error": f"RAG query timed out after {API_TIMEOUT} seconds. Try a simpler question or reduce top_k.",
            "success": False,
        }
    except requests.exceptions.ConnectionError:
        return {
            "error": f"Could not connect to RAG API at {RAG_API_URL}. Ensure the server is running.",
            "success": False,
        }
    except requests.exceptions.HTTPError as e:
        return {
            "error": f"RAG API returned error: {e.response.status_code} - {e.response.text}",
            "success": False,
        }
    except Exception as e:
        return {
            "error": f"RAG query failed: {str(e)}",
            "success": False,
        }


@tool
def similarity_search_vectordb(
    query: str,
    collection_names: List[str],
    k: int = 5,
) -> Dict[str, Any]:
    """
    Perform direct similarity search on vector database collections.

    This tool searches the vector database directly and returns raw document chunks.
    Unlike query_rag_endpoint, this tool does NOT generate an answer - YOU (the agent)
    must synthesize the answer from the retrieved documents.

    Use this tool when:
    - You need to search MULTIPLE collections simultaneously
    - You want fine-grained control over how the response is generated
    - You're doing a simple document lookup ("Find documents about X")
    - You're comparing information across different document collections
    - You need to see the raw source content before synthesizing an answer

    Args:
        query: Search query to find similar documents. Be specific for better results.
        collection_names: List of collection names to search. Can be one or many
                         (e.g., ["deepseek"] or ["deepseek", "qwen", "openai"]).
        k: Number of documents to retrieve PER collection (default: 5).

    Returns:
        Dictionary containing:
        - documents: List of retrieved documents with content, source, collection, and metadata
        - query: The original search query
        - collections_searched: List of collections that were searched
        - total_docs: Total number of documents retrieved
        - error: Error message if the search failed
    """
    try:
        # Validate k
        k = max(1, min(k, 20))

        # Initialize embeddings
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

        all_documents = []
        collections_found = []
        collections_not_found = []

        for collection_name in collection_names:
            collection_path = Path(VECTOR_STORE_BASE_PATH) / collection_name

            # Check if collection exists
            if not collection_path.exists():
                collections_not_found.append(collection_name)
                continue

            # Check for valid Chroma database
            chroma_file = collection_path / "chroma.sqlite3"
            if not chroma_file.exists():
                collections_not_found.append(collection_name)
                continue

            try:
                # Initialize vector store for this collection
                vector_store = Chroma(
                    persist_directory=str(collection_path),
                    embedding_function=embeddings,
                    collection_name=collection_name,
                )

                # Perform similarity search
                docs = vector_store.similarity_search(query, k=k)
                collections_found.append(collection_name)

                # Format documents
                for doc in docs:
                    all_documents.append(
                        {
                            "content": doc.page_content,
                            "source": doc.metadata.get(
                                "filename", doc.metadata.get("source", "Unknown")
                            ),
                            "collection": collection_name,
                            "metadata": doc.metadata,
                        }
                    )

            except Exception as e:
                # Log but continue with other collections
                collections_not_found.append(f"{collection_name} (error: {str(e)})")
                continue

        # Build response
        result = {
            "query": query,
            "documents": all_documents,
            "total_docs": len(all_documents),
            "collections_searched": collections_found,
            "success": True,
        }

        if collections_not_found:
            result["collections_not_found"] = collections_not_found
            result["warning"] = (
                f"Some collections were not found or had errors: {collections_not_found}"
            )

        if not all_documents:
            result["message"] = (
                "No documents found matching the query in the specified collections."
            )

        return result

    except Exception as e:
        return {
            "error": f"Similarity search failed: {str(e)}",
            "query": query,
            "collections_searched": [],
            "documents": [],
            "total_docs": 0,
            "success": False,
        }


@tool
def web_search_tavily(
    query: str,
    max_results: int = 5,
) -> Dict[str, Any]:
    """
    Search the web for current information using Tavily API.

    This tool searches the internet and returns relevant web results.
    Use it for information that is likely NOT in the document collections.

    Use this tool when:
    - The question asks about current events or recent news
    - The question asks about real-time data (stock prices, weather, live scores)
    - The information is unlikely to be in the document collections
    - The user explicitly asks to search the web/internet
    - Document search returned no relevant results and web info might help
    - You need to verify or supplement document information with external sources

    Args:
        query: Search query for web search. Be specific for better results.
        max_results: Maximum number of results to return (default: 5, max: 10).

    Returns:
        Dictionary containing:
        - results: List of web search results with title, url, and content
        - query: The original search query
        - source: "web_search" to indicate the source
        - error: Error message if the search failed
    """
    try:
        # Check if Tavily is configured
        if not TAVILY_API_KEY:
            return {
                "error": "Web search is not configured. TAVILY_API_KEY environment variable is not set.",
                "query": query,
                "results": [],
                "success": False,
                "suggestion": "You can still search the document collections using similarity_search_vectordb or query_rag_endpoint.",
            }

        # Validate max_results
        max_results = max(1, min(max_results, 10))

        # Import Tavily (lazy import to avoid errors if not installed)
        try:
            from langchain_community.tools.tavily_search import TavilySearchResults
        except ImportError:
            return {
                "error": "Tavily search is not installed. Install with: pip install tavily-python",
                "query": query,
                "results": [],
                "success": False,
            }

        # Initialize Tavily search
        tavily = TavilySearchResults(
            max_results=max_results,
            api_key=TAVILY_API_KEY,
        )

        # Perform search
        results = tavily.invoke({"query": query})

        # Format results
        formatted_results = []
        if isinstance(results, list):
            for result in results:
                if isinstance(result, dict):
                    formatted_results.append(
                        {
                            "title": result.get("title", "No title"),
                            "url": result.get("url", ""),
                            "content": result.get("content", ""),
                        }
                    )
                else:
                    formatted_results.append({"content": str(result)})

        return {
            "query": query,
            "results": formatted_results,
            "total_results": len(formatted_results),
            "source": "web_search",
            "success": True,
        }

    except Exception as e:
        error_msg = str(e)
        if "api key" in error_msg.lower() or "unauthorized" in error_msg.lower():
            return {
                "error": "Invalid Tavily API key. Please check your TAVILY_API_KEY.",
                "query": query,
                "results": [],
                "success": False,
            }
        return {
            "error": f"Web search failed: {error_msg}",
            "query": query,
            "results": [],
            "success": False,
        }


@tool
def list_available_collections() -> Dict[str, Any]:
    """
    List all available document collections in the RAG system.

    This tool returns information about what document collections exist
    and are available for querying.

    Use this tool when:
    - You need to know what collections exist before searching
    - The user asks what documents or collections are available
    - You're unsure which collection name to use for a query
    - You want to provide the user with options for what to search

    Returns:
        Dictionary containing:
        - collections: List of collection names
        - collection_details: List of detailed info about each collection
        - count: Total number of collections
        - error: Error message if the query failed
    """
    try:
        # Try to get from API first (has more details)
        try:
            response = requests.get(
                f"{RAG_API_URL}/collections",
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()

            collections = []
            collection_details = []

            for col in data.get("collections", []):
                name = col.get("name", "")
                collections.append(name)
                collection_details.append(
                    {
                        "name": name,
                        "file_count": col.get("file_count", 0),
                        "vector_count": col.get("vector_count", 0),
                        "vector_exists": col.get("vector_exists", False),
                    }
                )

            return {
                "collections": collections,
                "collection_details": collection_details,
                "count": len(collections),
                "current_collection": data.get("current_collection"),
                "success": True,
            }

        except (requests.exceptions.RequestException, Exception):
            # Fall back to filesystem check
            pass

        # Fallback: Check filesystem directly
        base_path = Path(VECTOR_STORE_BASE_PATH)

        if not base_path.exists():
            return {
                "collections": [],
                "collection_details": [],
                "count": 0,
                "message": "Vector store directory does not exist. No collections available.",
                "success": True,
            }

        collections = []
        collection_details = []

        for item in base_path.iterdir():
            if item.is_dir():
                # Check if it's a valid Chroma collection
                chroma_file = item / "chroma.sqlite3"
                if chroma_file.exists():
                    collections.append(item.name)
                    collection_details.append(
                        {
                            "name": item.name,
                            "path": str(item),
                            "has_vector_store": True,
                        }
                    )

        return {
            "collections": sorted(collections),
            "collection_details": sorted(collection_details, key=lambda x: x["name"]),
            "count": len(collections),
            "success": True,
        }

    except Exception as e:
        return {
            "error": f"Failed to list collections: {str(e)}",
            "collections": [],
            "collection_details": [],
            "count": 0,
            "success": False,
        }


def get_all_tools() -> List:
    """
    Get all available tools for the agent.

    Returns:
        List of tool functions that can be bound to the LLM.
    """
    return [
        query_rag_endpoint,
        similarity_search_vectordb,
        web_search_tavily,
        list_available_collections,
    ]


def get_tools_by_name() -> Dict[str, Any]:
    """
    Get a dictionary mapping tool names to tool functions.

    Returns:
        Dictionary with tool names as keys and tool functions as values.
    """
    return {
        "query_rag_endpoint": query_rag_endpoint,
        "similarity_search_vectordb": similarity_search_vectordb,
        "web_search_tavily": web_search_tavily,
        "list_available_collections": list_available_collections,
    }


def get_enabled_tools(
    enable_rag: bool = True,
    enable_similarity_search: bool = True,
    enable_web_search: bool = True,
    enable_list_collections: bool = True,
) -> List:
    """
    Get tools based on which are enabled.

    This allows selective enabling/disabling of tools based on configuration.

    Args:
        enable_rag: Enable RAG query tool
        enable_similarity_search: Enable similarity search tool
        enable_web_search: Enable web search tool (also requires TAVILY_API_KEY)
        enable_list_collections: Enable list collections tool

    Returns:
        List of enabled tool functions.
    """
    tools = []

    if enable_rag:
        tools.append(query_rag_endpoint)

    if enable_similarity_search:
        tools.append(similarity_search_vectordb)

    if enable_web_search and TAVILY_API_KEY:
        tools.append(web_search_tavily)

    if enable_list_collections:
        tools.append(list_available_collections)

    return tools
