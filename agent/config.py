import os
from typing import Optional

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

load_dotenv()


class AgentConfig(BaseSettings):
    """Configuration settings for the ReAct agent."""

    # LLM Settings
    llm_model: str = Field(
        default_factory=lambda: os.getenv("OLLAMA_AGENT_LLM_MODEL", "ministral-3"),
        description="Ollama model to use for agent reasoning",
    )
    embedding_model: str = Field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "qwen3-embedding"),
        description="Embedding model for similarity search",
    )
    temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Temperature for LLM responses (lower = more deterministic)",
    )

    # Agent Behavior Settings
    max_reasoning_steps: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum number of reasoning steps before forcing termination",
    )
    default_top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Default number of documents to retrieve",
    )

    # API Settings
    rag_api_url: str = Field(
        default="http://localhost:8000",
        description="Base URL for the RAG FastAPI server",
    )
    api_timeout: int = Field(
        default=60,
        ge=10,
        le=300,
        description="Timeout in seconds for API calls",
    )

    # Vector Store Settings
    vector_store_base_path: str = Field(
        default_factory=lambda: os.getenv("VECTOR_STORE_PATH", "./data/vector_store"),
        description="Base path for vector store collections",
    )

    # Web Search Settings (Tavily)
    tavily_api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("TAVILY_API_KEY"),
        description="Tavily API key for web search",
    )
    web_search_max_results: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of web search results",
    )
    web_search_enabled: bool = Field(
        default=True,
        description="Whether web search tool is enabled",
    )

    # Observability Settings
    langfuse_enabled: bool = Field(
        default_factory=lambda: bool(os.getenv("LANGFUSE_PUBLIC_KEY")),
        description="Whether LangFuse tracing is enabled",
    )
    langfuse_public_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("LANGFUSE_PUBLIC_KEY"),
        description="LangFuse public key",
    )
    langfuse_secret_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("LANGFUSE_SECRET_KEY"),
        description="LangFuse secret key",
    )
    langfuse_base_url: Optional[str] = Field(
        default_factory=lambda: os.getenv(
            "LANGFUSE_BASE_URL", "https://cloud.langfuse.com"
        ),
        description="LangFuse base URL",
    )

    # Memory Settings
    enable_memory: bool = Field(
        default=True,
        description="Whether to enable conversation memory",
    )
    memory_max_messages: int = Field(
        default=20,
        ge=5,
        le=100,
        description="Maximum number of messages to keep in memory",
    )

    # Debug Settings
    debug_mode: bool = Field(
        default=False,
        description="Enable debug logging for agent reasoning",
    )
    log_tool_calls: bool = Field(
        default=True,
        description="Log tool calls and responses",
    )

    class Config:
        env_prefix = "AGENT_"
        env_file = ".env"
        extra = "ignore"

    def is_web_search_available(self) -> bool:
        """Check if web search is configured and available."""
        return self.web_search_enabled and self.tavily_api_key is not None

    def is_langfuse_available(self) -> bool:
        """Check if LangFuse is configured and available."""
        return (
            self.langfuse_enabled
            and self.langfuse_public_key is not None
            and self.langfuse_secret_key is not None
        )


# Singleton instance
_agent_config: Optional[AgentConfig] = None


def get_agent_config() -> AgentConfig:
    """Get or create the agent configuration singleton."""
    global _agent_config
    if _agent_config is None:
        _agent_config = AgentConfig()
    return _agent_config


def reset_agent_config() -> None:
    """Reset the agent configuration (useful for testing)."""
    global _agent_config
    _agent_config = None
