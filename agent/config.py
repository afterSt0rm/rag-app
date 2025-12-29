import os
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


@dataclass
class AgentConfig:
    """Configuration for the Agentic RAG system."""

    # API Configuration
    api_base_url: str = field(
        default_factory=lambda: os.getenv("API_BASE_URL", "http://localhost:8000")
    )

    # LLM Configuration
    llm_model: str = field(
        default_factory=lambda: os.getenv("OLLAMA_AGENT_LLM_MODEL", "ministral-3")
    )
    temperature: float = 0.0  # Use 0 for deterministic intent classification

    # Tavily Configuration
    tavily_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("TAVILY_API_KEY")
    )
    tavily_max_results: int = 3

    # Default collection
    default_collection: Optional[str] = None

    # Retrieval settings
    default_top_k: int = 5

    def __post_init__(self):
        if not self.tavily_api_key:
            print(
                "⚠️  Warning: TAVILY_API_KEY not set. Web search tool will not be available."
            )
