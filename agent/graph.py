import logging
import uuid
from typing import Any, Dict, List, Optional

from agent.config import AgentConfig, get_agent_config
from agent.nodes import (
    create_agent_node,
    create_input_node,
    format_response,
    should_continue,
)
from agent.prompts import get_system_prompt
from agent.state import AgentState, create_initial_state
from agent.tools import get_all_tools, get_enabled_tools
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

load_dotenv()


# ===================
# Langfuse Integration
# ===================


def get_langfuse_handler(
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    trace_name: Optional[str] = None,
    tags: Optional[List[str]] = None,
):
    """
    Get a Langfuse callback handler for tracing agent execution.

    Returns None if Langfuse is not configured or available.

    Args:
        session_id: Optional session ID to group related traces
        user_id: Optional user ID to associate with the trace
        trace_name: Optional custom name for the trace
        tags: Optional list of tags for the trace

    Returns:
        CallbackHandler if Langfuse is configured, None otherwise
    """
    config = get_agent_config()

    if not config.is_langfuse_available():
        logger.debug("Langfuse not configured, skipping tracing")
        return None

    try:
        from langfuse.langchain import CallbackHandler

        # Build metadata for trace attributes
        metadata = {}
        if user_id:
            metadata["langfuse_user_id"] = user_id
        if session_id:
            metadata["langfuse_session_id"] = session_id
        if tags:
            metadata["langfuse_tags"] = tags

        handler = CallbackHandler()

        logger.debug(f"Langfuse handler created for trace: {trace_name}")
        return handler

    except ImportError:
        logger.warning("Langfuse package not installed. Run: pip install langfuse")
        return None
    except Exception as e:
        logger.warning(f"Failed to initialize Langfuse handler: {e}")
        return None


def get_langfuse_config(
    thread_id: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    trace_name: Optional[str] = None,
    tags: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Build a LangGraph config dict with Langfuse callback handler included.

    This is a convenience function that combines the thread_id configurable
    with Langfuse tracing callbacks.

    Args:
        thread_id: Thread ID for conversation memory
        user_id: Optional user ID for Langfuse trace
        session_id: Optional session ID for Langfuse trace
        trace_name: Optional custom trace name
        tags: Optional tags for the trace

    Returns:
        Config dict ready to pass to agent.invoke() or agent.ainvoke()
    """
    config: Dict[str, Any] = {
        "configurable": {"thread_id": thread_id or str(uuid.uuid4())}
    }

    # Get Langfuse handler
    handler = get_langfuse_handler(
        session_id=session_id,
        user_id=user_id,
        trace_name=trace_name,
        tags=tags,
    )

    if handler:
        config["callbacks"] = [handler]

        # Set run_name for trace name in Langfuse
        config["run_name"] = trace_name or "agent"

        # Add tags if provided
        if tags:
            config["tags"] = tags

        # Add metadata for trace attributes
        metadata = {}
        if user_id:
            metadata["langfuse_user_id"] = user_id
        if session_id:
            metadata["langfuse_session_id"] = session_id

        if metadata:
            config["metadata"] = metadata

    return config


def flush_langfuse():
    """
    Flush any pending Langfuse events.

    Call this in short-lived applications or before shutdown
    to ensure all traces are sent to Langfuse.
    """
    config = get_agent_config()

    if not config.is_langfuse_available():
        return

    try:
        from langfuse import get_client

        client = get_client()
        client.flush()
        logger.debug("Langfuse events flushed")
    except Exception as e:
        logger.warning(f"Failed to flush Langfuse events: {e}")


def shutdown_langfuse():
    """
    Shutdown Langfuse client and flush remaining events.

    Call this when your application is shutting down.
    """
    config = get_agent_config()

    if not config.is_langfuse_available():
        return

    try:
        from langfuse import get_client

        client = get_client()
        client.shutdown()
        logger.debug("Langfuse client shutdown complete")
    except Exception as e:
        logger.warning(f"Failed to shutdown Langfuse client: {e}")


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_react_agent(
    config: Optional[AgentConfig] = None,
    enable_memory: bool = True,
):
    """
    Create a ReAct agent using LangGraph.

    This function builds the complete agent workflow with:
    - Input processing node
    - Agent reasoning node (LLM with tools)
    - Tool execution node
    - Conditional routing (ReAct loop)

    Args:
        config: Optional agent configuration. Uses default if not provided.
        enable_memory: Whether to enable conversation memory persistence.

    Returns:
        Compiled LangGraph workflow that can be invoked with queries.
    """
    if config is None:
        config = get_agent_config()

    # Initialize LLM
    llm = ChatOllama(
        model=config.llm_model,
        temperature=config.temperature,
    )

    # Get tools based on configuration
    tools = get_enabled_tools(
        enable_rag=True,
        enable_similarity_search=True,
        enable_web_search=config.is_web_search_available(),
        enable_list_collections=True,
    )

    if config.debug_mode:
        logger.info(f"Enabled tools: {[t.name for t in tools]}")

    # Bind tools to LLM
    llm_with_tools = llm.bind_tools(tools)

    # Get system prompt
    system_prompt = get_system_prompt(concise=False)

    # ===================
    # Create Node Functions from nodes.py
    # ===================

    input_node = create_input_node(system_prompt=system_prompt)

    agent_node = create_agent_node(
        llm_with_tools=llm_with_tools,
        max_reasoning_steps=config.max_reasoning_steps,
        debug_mode=config.debug_mode,
        log_tool_calls=config.log_tool_calls,
    )

    # should_continue and format_response are imported directly from nodes.py

    # ===================
    # Build the Graph
    # ===================

    # Create the state graph
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("input", input_node)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(tools))
    workflow.add_node("format_response", format_response)

    # Set entry point
    workflow.set_entry_point("input")

    # Add edges
    workflow.add_edge("input", "agent")

    # Add conditional edge for ReAct loop
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end": "format_response",
        },
    )

    # Tools always go back to agent (for observation and reasoning)
    workflow.add_edge("tools", "agent")

    # Format response goes to END
    workflow.add_edge("format_response", END)

    # ===================
    # Compile the Graph
    # ===================

    if enable_memory:
        memory = MemorySaver()
        graph = workflow.compile(checkpointer=memory)
    else:
        graph = workflow.compile()

    return graph


# ===================
# Singleton Management
# ===================

_agent_instance = None
_agent_config_hash = None


def get_agent(force_recreate: bool = False) -> Any:
    """
    Get or create the singleton ReAct agent instance.

    Args:
        force_recreate: Force recreation of the agent even if one exists.

    Returns:
        Compiled LangGraph agent workflow.
    """
    global _agent_instance, _agent_config_hash

    config = get_agent_config()
    current_config_hash = hash(
        (config.llm_model, config.temperature, config.max_reasoning_steps)
    )

    # Recreate if config changed or forced
    if (
        _agent_instance is None
        or force_recreate
        or _agent_config_hash != current_config_hash
    ):
        logger.info("Creating new ReAct agent instance")
        _agent_instance = create_react_agent(config)
        _agent_config_hash = current_config_hash

    return _agent_instance


def reset_agent() -> None:
    """Reset the singleton agent instance."""
    global _agent_instance, _agent_config_hash
    _agent_instance = None
    _agent_config_hash = None
    logger.info("Agent instance reset")


# ===================
# High-Level API
# ===================


async def run_agent(
    query: str,
    collection_names: Optional[List[str]] = None,
    thread_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run the ReAct agent with a query.

    This is the main entry point for using the agent. It handles:
    - State initialization
    - Agent execution
    - Result extraction

    Args:
        query: The user's question or request.
        collection_names: Optional list of collections to search.
                         If None, agent will discover and decide.
        thread_id: Conversation thread ID for memory persistence.
                  If None, a new UUID is generated.

    Returns:
        Dictionary containing:
        - query: Original query
        - response: Agent's response
        - reasoning_steps: Number of reasoning steps taken
        - tool_used: Primary tool used (first tool called)
        - tools_used: List of all tools used
        - error: Error message (if any)

    Example:
        >>> result = await run_agent(
        ...     query="What are the key findings about transformers?",
        ...     collection_names=["research_papers"]
        ... )
        >>> print(result["response"])
    """
    agent = get_agent()
    config = get_agent_config()

    # Generate thread ID if not provided
    if thread_id is None:
        thread_id = str(uuid.uuid4())

    # Create initial state
    initial_state = create_initial_state(
        query=query,
        collection_names=collection_names,
        thread_id=thread_id,
    )

    # Configure the run with Langfuse tracing if available
    run_config = get_langfuse_config(
        thread_id=thread_id,
        trace_name="agent",
        tags=["rag", "async"]
        + (
            ["multi-collection"]
            if collection_names and len(collection_names) > 1
            else []
        ),
    )

    try:
        if config.debug_mode:
            logger.info(f"Running agent with query: {query[:100]}...")
            logger.info(f"Collections: {collection_names}")
            logger.info(f"Thread ID: {thread_id}")
            logger.info(
                f"Langfuse tracing: {'enabled' if 'callbacks' in run_config else 'disabled'}"
            )

        # Run the agent (async)
        result = await agent.ainvoke(initial_state, run_config)

        # Extract results
        response = result.get("response", "No response generated")
        reasoning_steps = result.get("reasoning_steps", 0)
        tools_used = result.get("tools_used", []) or []
        tool_used = result.get("tool_used") or (tools_used[0] if tools_used else None)
        error = result.get("error")

        if config.debug_mode:
            logger.info(f"Agent completed in {reasoning_steps} steps")
            logger.info(f"Tools used: {tools_used}")

        return {
            "query": query,
            "response": response,
            "reasoning_steps": reasoning_steps,
            "tool_used": tool_used,
            "tools_used": tools_used,
            "thread_id": thread_id,
            "error": error,
        }

    except Exception as e:
        logger.error(f"Agent execution error: {e}")
        return {
            "query": query,
            "response": f"I encountered an error while processing your request: {str(e)}",
            "reasoning_steps": 0,
            "tool_used": None,
            "tools_used": [],
            "thread_id": thread_id,
            "error": str(e),
        }


def run_agent_sync(
    query: str,
    collection_names: Optional[List[str]] = None,
    thread_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Synchronous version of run_agent.

    Use this when you're not in an async context.

    Args:
        query: The user's question or request.
        collection_names: Optional list of collections to search.
        thread_id: Conversation thread ID for memory persistence.

    Returns:
        Dictionary with response and metadata.
    """
    agent = get_agent()

    # Generate thread ID if not provided
    if thread_id is None:
        thread_id = str(uuid.uuid4())

    # Create initial state
    initial_state = create_initial_state(
        query=query,
        collection_names=collection_names,
        thread_id=thread_id,
    )

    # Configure the run with Langfuse tracing if available
    run_config = get_langfuse_config(
        thread_id=thread_id,
        trace_name="agent",
        tags=["rag", "sync"]
        + (
            ["multi-collection"]
            if collection_names and len(collection_names) > 1
            else []
        ),
    )

    try:
        # Run the agent (sync)
        result = agent.invoke(initial_state, run_config)

        # Extract results
        tools_used = result.get("tools_used", []) or []
        tool_used = result.get("tool_used") or (tools_used[0] if tools_used else None)

        return {
            "query": query,
            "response": result.get("response", "No response generated"),
            "reasoning_steps": result.get("reasoning_steps", 0),
            "tool_used": tool_used,
            "tools_used": tools_used,
            "thread_id": thread_id,
            "error": result.get("error"),
        }

    except Exception as e:
        logger.error(f"Agent execution error: {e}")
        return {
            "query": query,
            "response": f"I encountered an error: {str(e)}",
            "reasoning_steps": 0,
            "tool_used": None,
            "tools_used": [],
            "thread_id": thread_id,
            "error": str(e),
        }


# ===================
# Streaming Support
# ===================


async def stream_agent(
    query: str,
    collection_names: Optional[List[str]] = None,
    thread_id: Optional[str] = None,
):
    """
    Stream the agent's execution for real-time updates.

    Yields state updates as the agent progresses through its
    reasoning and tool execution steps.

    Args:
        query: The user's question or request.
        collection_names: Optional list of collections to search.
        thread_id: Conversation thread ID for memory persistence.

    Yields:
        Dictionary with current state information including:
        - node: Current node being executed
        - messages: Updated messages
        - tool_calls: Any pending tool calls
        - tools_used: Tools used so far
    """
    agent = get_agent()

    if thread_id is None:
        thread_id = str(uuid.uuid4())

    initial_state = create_initial_state(
        query=query,
        collection_names=collection_names,
        thread_id=thread_id,
    )

    # Configure with Langfuse tracing if available
    run_config = get_langfuse_config(
        thread_id=thread_id,
        trace_name="agent",
        tags=["rag", "stream"]
        + (
            ["multi-collection"]
            if collection_names and len(collection_names) > 1
            else []
        ),
    )
    tools_used = []

    try:
        async for event in agent.astream(initial_state, run_config):
            # event is a dict with node name as key
            for node_name, node_output in event.items():
                # Track tools used from the output
                if "tools_used" in node_output:
                    tools_used = node_output["tools_used"]

                yield {
                    "node": node_name,
                    "output": node_output,
                    "thread_id": thread_id,
                    "tools_used": tools_used,
                }

    except Exception as e:
        logger.error(f"Streaming error: {e}")
        yield {
            "node": "error",
            "output": {"error": str(e)},
            "thread_id": thread_id,
            "tools_used": tools_used,
        }


# ===================
# Utility Functions
# ===================


def get_agent_info() -> Dict[str, Any]:
    """
    Get information about the current agent configuration.

    Returns:
        Dictionary with agent configuration and status.
    """
    config = get_agent_config()

    return {
        "llm_model": config.llm_model,
        "temperature": config.temperature,
        "max_reasoning_steps": config.max_reasoning_steps,
        "web_search_enabled": config.is_web_search_available(),
        "langfuse_enabled": config.is_langfuse_available(),
        "memory_enabled": config.enable_memory,
        "debug_mode": config.debug_mode,
        "tools_available": [t.name for t in get_all_tools()],
    }
