import json
import logging
import time
from typing import Any, Dict, List, Optional

from agent.config import get_agent_config
from agent.graph import (
    get_agent,
    get_agent_info,
    reset_agent,
    run_agent,
    run_agent_sync,
    stream_agent,
)
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# Setup logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/agent", tags=["Agent"])


# ===================
# Request/Response Models
# ===================


class AgentQueryRequest(BaseModel):
    """Request model for agent queries."""

    query: str = Field(
        ...,
        description="The question or request to send to the agent",
        min_length=1,
        max_length=10000,
        examples=["What are the key findings about transformers?"],
    )
    collection_names: Optional[List[str]] = Field(
        None,
        description="Optional list of collection names to search. If not provided, agent will discover available collections.",
        examples=[["research_papers", "deepseek"]],
    )
    thread_id: Optional[str] = Field(
        None,
        description="Conversation thread ID for memory persistence. If not provided, a new thread is created.",
        examples=["user_123_session_456"],
    )


class SourceInfo(BaseModel):
    """Information about a source document."""

    content: str = Field(description="Content snippet from the source")
    source: str = Field(description="Source file or document name")
    collection: str = Field(description="Collection the document belongs to")


class AgentQueryResponse(BaseModel):
    """Response model for agent queries."""

    query: str = Field(description="The original query")
    response: str = Field(description="The agent's response")
    reasoning_steps: int = Field(
        description="Number of reasoning steps taken by the agent"
    )
    tools_used: List[str] = Field(
        default_factory=list,
        description="List of all tools used by the agent during execution",
    )
    thread_id: Optional[str] = Field(
        None, description="Thread ID for conversation continuity"
    )
    processing_time: Optional[float] = Field(
        None, description="Total processing time in seconds"
    )
    error: Optional[str] = Field(None, description="Error message if query failed")


class AgentInfoResponse(BaseModel):
    """Response model for agent information."""

    llm_model: str = Field(description="LLM model being used")
    temperature: float = Field(description="Temperature setting for LLM")
    max_reasoning_steps: int = Field(description="Maximum reasoning steps allowed")
    web_search_enabled: bool = Field(description="Whether web search is available")
    langfuse_enabled: bool = Field(description="Whether LangFuse tracing is enabled")
    memory_enabled: bool = Field(description="Whether conversation memory is enabled")
    debug_mode: bool = Field(description="Whether debug mode is enabled")
    tools_available: List[str] = Field(description="List of available tools")


class AgentHealthResponse(BaseModel):
    """Response model for agent health check."""

    status: str = Field(description="Health status (healthy/unhealthy)")
    agent_loaded: bool = Field(description="Whether the agent is loaded")
    config_valid: bool = Field(description="Whether configuration is valid")
    tools_count: int = Field(description="Number of tools available")
    message: Optional[str] = Field(None, description="Additional status message")


class StreamChunk(BaseModel):
    """Model for streaming response chunks."""

    node: str = Field(description="Current node being executed")
    content: Optional[str] = Field(None, description="Content if available")
    tool_call: Optional[Dict[str, Any]] = Field(
        None, description="Tool call information if applicable"
    )
    done: bool = Field(default=False, description="Whether streaming is complete")


# ===================
# API Endpoints
# ===================


@router.post("/query", response_model=AgentQueryResponse)
async def agent_query(request: AgentQueryRequest) -> AgentQueryResponse:
    """
    Query the ReAct agent with a question.

    The agent will:
    1. Analyze your question
    2. Decide which tool(s) to use (RAG, similarity search, web search)
    3. Execute the selected tool(s)
    4. Synthesize and return a comprehensive response

    **Example requests:**

    - Simple document query:
      ```json
      {
        "query": "What are the main findings in the research paper?",
        "collection_names": ["research_papers"]
      }
      ```

    - Multi-collection comparison:
      ```json
      {
        "query": "Compare the approaches used in Deepseek and Qwen models",
        "collection_names": ["deepseek", "qwen"]
      }
      ```

    - Web search (for current events):
      ```json
      {
        "query": "What are the latest developments in AI regulation?"
      }
      ```
    """
    start_time = time.time()

    try:
        # Run the agent
        result = await run_agent(
            query=request.query,
            collection_names=request.collection_names,
            thread_id=request.thread_id,
        )

        processing_time = time.time() - start_time

        return AgentQueryResponse(
            query=result["query"],
            response=result["response"],
            reasoning_steps=result["reasoning_steps"],
            tools_used=result.get("tools_used", []),
            thread_id=result.get("thread_id"),
            processing_time=processing_time,
            error=result.get("error"),
        )

    except Exception as e:
        logger.error(f"Agent query error: {e}")
        processing_time = time.time() - start_time
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "processing_time": processing_time,
            },
        )


@router.post("/query/sync", response_model=AgentQueryResponse)
def agent_query_sync(request: AgentQueryRequest) -> AgentQueryResponse:
    """
    Synchronous version of the agent query endpoint.

    Use this endpoint if you need a synchronous request/response
    pattern without async/await.
    """
    start_time = time.time()

    try:
        # Run the agent synchronously
        result = run_agent_sync(
            query=request.query,
            collection_names=request.collection_names,
            thread_id=request.thread_id,
        )

        processing_time = time.time() - start_time

        return AgentQueryResponse(
            query=result["query"],
            response=result["response"],
            reasoning_steps=result["reasoning_steps"],
            tools_used=result.get("tools_used", []),
            thread_id=result.get("thread_id"),
            processing_time=processing_time,
            error=result.get("error"),
        )

    except Exception as e:
        logger.error(f"Agent sync query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query/stream")
async def agent_query_stream(request: AgentQueryRequest):
    """
    Stream the agent's response for real-time UI updates.

    This endpoint uses Server-Sent Events (SSE) to stream
    the agent's progress as it reasons and executes tools.

    **Response format:**
    Each chunk is a JSON object with:
    - `node`: Current execution node (input, agent, tools, format_response)
    - `content`: Any content generated
    - `tool_call`: Tool call information (if applicable)
    - `done`: Boolean indicating if streaming is complete
    """

    async def generate():
        last_content = None
        tools_used = []
        reasoning_steps = 0

        try:
            async for chunk in stream_agent(
                query=request.query,
                collection_names=request.collection_names,
                thread_id=request.thread_id,
            ):
                # Format as SSE
                node = chunk.get("node", "unknown")
                output = chunk.get("output", {})

                # Track tools used from chunk
                if chunk.get("tools_used"):
                    tools_used = chunk.get("tools_used")

                # Extract relevant information
                stream_data = {
                    "node": node,
                    "content": None,
                    "tool_call": None,
                    "tool_result": None,
                    "done": False,
                    "tools_used": tools_used,
                    "reasoning_steps": reasoning_steps,
                }

                # Check for message content
                if "messages" in output:
                    messages = output["messages"]
                    if messages:
                        last_msg = (
                            messages[-1] if isinstance(messages, list) else messages
                        )
                        if hasattr(last_msg, "content") and last_msg.content:
                            stream_data["content"] = last_msg.content
                            last_content = last_msg.content
                        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                            # Extract tool call info
                            tc = last_msg.tool_calls[0]
                            stream_data["tool_call"] = {
                                "name": tc.get("name", "unknown"),
                                "args": tc.get("args", {}),
                            }
                        # Check if this is a ToolMessage (tool result)
                        if hasattr(last_msg, "name") and hasattr(last_msg, "content"):
                            # This is likely a tool result
                            stream_data["tool_result"] = {
                                "tool_name": getattr(last_msg, "name", "unknown"),
                                "result_preview": str(last_msg.content)[:500]
                                if last_msg.content
                                else None,
                            }

                # Track reasoning steps
                if "reasoning_steps" in output:
                    reasoning_steps = output["reasoning_steps"]
                    stream_data["reasoning_steps"] = reasoning_steps

                # Check for final response
                if "response" in output:
                    stream_data["content"] = output["response"]
                    stream_data["done"] = True
                    last_content = output["response"]

                # Check for error in output
                if "error" in output:
                    stream_data["error"] = output["error"]

                yield f"data: {json.dumps(stream_data)}\n\n"

            # Send final done signal with last content as fallback
            final_data = {
                "node": "complete",
                "done": True,
                "content": last_content,
                "tools_used": tools_used,
                "reasoning_steps": reasoning_steps,
            }
            yield f"data: {json.dumps(final_data)}\n\n"

        except Exception as e:
            logger.error(f"Streaming error: {e}")
            error_data = {
                "node": "error",
                "error": str(e),
                "done": True,
                "content": last_content,  # Include last content even on error
                "tools_used": tools_used,
            }
            yield f"data: {json.dumps(error_data)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/info", response_model=AgentInfoResponse)
async def get_agent_info_endpoint() -> AgentInfoResponse:
    """
    Get information about the current agent configuration.

    Returns details about:
    - LLM model and settings
    - Available tools
    - Feature flags (web search, memory, etc.)
    """
    try:
        info = get_agent_info()
        return AgentInfoResponse(**info)
    except Exception as e:
        logger.error(f"Error getting agent info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", response_model=AgentHealthResponse)
async def agent_health_check() -> AgentHealthResponse:
    """
    Health check endpoint for the agent service.

    Verifies that:
    - Agent can be instantiated
    - Configuration is valid
    - Tools are available
    """
    try:
        config = get_agent_config()
        agent = get_agent()
        info = get_agent_info()

        return AgentHealthResponse(
            status="healthy",
            agent_loaded=agent is not None,
            config_valid=True,
            tools_count=len(info.get("tools_available", [])),
            message="Agent is ready to accept queries",
        )

    except Exception as e:
        logger.error(f"Agent health check failed: {e}")
        return AgentHealthResponse(
            status="unhealthy",
            agent_loaded=False,
            config_valid=False,
            tools_count=0,
            message=f"Agent initialization failed: {str(e)}",
        )


@router.post("/reset")
async def reset_agent_endpoint():
    """
    Reset the agent instance.

    This clears the current agent and forces recreation on next use.
    Useful for:
    - Applying configuration changes
    - Clearing any cached state
    - Debugging purposes
    """
    try:
        reset_agent()
        return {
            "success": True,
            "message": "Agent has been reset. It will be recreated on next query.",
        }
    except Exception as e:
        logger.error(f"Error resetting agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tools")
async def list_agent_tools():
    """
    List all tools available to the agent.

    Returns detailed information about each tool including:
    - Name
    - Description
    - Parameters
    """
    try:
        from agent.tools import get_all_tools

        tools = get_all_tools()
        tool_info = []

        for tool in tools:
            info = {
                "name": tool.name,
                "description": tool.description,
            }

            # Get parameter info if available
            if hasattr(tool, "args_schema") and tool.args_schema:
                schema = tool.args_schema.schema()
                info["parameters"] = schema.get("properties", {})
                info["required"] = schema.get("required", [])

            tool_info.append(info)

        return {
            "tools": tool_info,
            "count": len(tool_info),
        }

    except Exception as e:
        logger.error(f"Error listing tools: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ===================
# Conversation Management
# ===================


@router.get("/conversations/{thread_id}/history")
async def get_conversation_history(
    thread_id: str,
    limit: int = Query(
        default=50, ge=1, le=200, description="Maximum messages to return"
    ),
):
    """
    Get conversation history for a thread.

    Note: This requires memory to be enabled in the agent configuration.
    """
    try:
        agent = get_agent()

        # Get state from checkpointer
        config = {"configurable": {"thread_id": thread_id}}

        # Try to get the current state
        try:
            state = agent.get_state(config)
            if state and state.values:
                messages = state.values.get("messages", [])

                # Format messages for response
                formatted_messages = []
                for msg in messages[-limit:]:
                    formatted_messages.append(
                        {
                            "type": type(msg).__name__,
                            "content": msg.content
                            if hasattr(msg, "content")
                            else str(msg),
                        }
                    )

                return {
                    "thread_id": thread_id,
                    "messages": formatted_messages,
                    "count": len(formatted_messages),
                }
        except Exception:
            pass

        return {
            "thread_id": thread_id,
            "messages": [],
            "count": 0,
            "message": "No conversation history found for this thread",
        }

    except Exception as e:
        logger.error(f"Error getting conversation history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ===================
# Export Router
# ===================

__all__ = ["router"]
