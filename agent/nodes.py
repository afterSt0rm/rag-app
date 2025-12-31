"""
Node functions for the ReAct agent workflow.

This module contains all node functions used in the LangGraph workflow.
Functions are designed to accept dependencies as parameters for flexibility
and testability.
"""

import logging
from typing import Any, Callable, Dict, Literal, Union

from agent.state import AgentState
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import Runnable

# Setup logging
logger = logging.getLogger(__name__)


def create_input_node(
    system_prompt: str,
) -> Callable[[AgentState], Dict[str, Any]]:
    """
    Create an input node function with the given system prompt.

    Args:
        system_prompt: The system prompt to use for the conversation.

    Returns:
        Input node function that can be used in the graph.
    """

    def input_node(state: AgentState) -> Dict[str, Any]:
        """
        Process initial input and prepare the conversation.

        Adds the system prompt and formats the user query with any
        collection context that was provided.

        Args:
            state: Initial state with query.

        Returns:
            State update with prepared messages.
        """
        query = state.get("query", "")
        collection_names = state.get("collection_names")
        messages = list(state.get("messages", []))

        # Add system prompt if not already present
        if not messages or not isinstance(messages[0], SystemMessage):
            messages.insert(0, SystemMessage(content=system_prompt))

        # Build user message with context
        user_message = query
        if collection_names:
            user_message += f"\n\n[Available collections for this query: {', '.join(collection_names)}]"

        # Add user message
        messages.append(HumanMessage(content=user_message))

        return {
            "messages": messages,
            "reasoning_steps": 0,
            "max_steps_reached": False,
            "tools_used": [],  # Initialize empty list for tracking tools
        }

    return input_node


def create_agent_node(
    llm_with_tools: Union[Runnable, Any],
    max_reasoning_steps: int = 10,
    debug_mode: bool = False,
    log_tool_calls: bool = False,
) -> Callable[[AgentState], Dict[str, Any]]:
    """
    Create the agent node function with LLM and tools bound.

    Args:
        llm_with_tools: LLM instance with tools already bound.
        max_reasoning_steps: Maximum number of reasoning steps allowed.
        debug_mode: Whether to enable debug logging.
        log_tool_calls: Whether to log tool calls.

    Returns:
        Agent node function that can be used in the graph.
    """

    def agent_node(state: AgentState) -> Dict[str, Any]:
        """
        The main agent reasoning node.

        This node invokes the LLM with the current conversation to:
        1. Reason about what action to take
        2. Generate tool calls OR a final response

        Args:
            state: Current agent state with messages and context.

        Returns:
            Updated state with new message from LLM.
        """
        messages = state.get("messages", [])
        reasoning_steps = state.get("reasoning_steps", 0)
        # Preserve existing tools_used list
        tools_used = list(state.get("tools_used", []) or [])

        # Check max steps
        if reasoning_steps >= max_reasoning_steps:
            logger.warning(f"Max reasoning steps ({max_reasoning_steps}) reached")
            return {
                "messages": [
                    AIMessage(
                        content="I've reached the maximum number of reasoning steps. "
                        "Let me provide the best answer I can based on the information gathered so far."
                    )
                ],
                "reasoning_steps": reasoning_steps,
                "max_steps_reached": True,
                "tools_used": tools_used,
            }

        try:
            # Invoke LLM with tools
            response = llm_with_tools.invoke(messages)

            # Track which tool(s) were selected and add to cumulative list
            if hasattr(response, "tool_calls") and response.tool_calls:
                for tc in response.tool_calls:
                    tool_name = tc.get("name", "unknown")
                    if tool_name not in tools_used:
                        tools_used.append(tool_name)

                    # Log tool selection for debugging
                    if debug_mode or log_tool_calls:
                        tool_args = tc.get("args", {})
                        logger.info(f"Tool call: {tool_name}({tool_args})")

            return {
                "messages": [response],
                "reasoning_steps": reasoning_steps + 1,
                "tools_used": tools_used,
            }

        except Exception as e:
            logger.error(f"Agent node error: {e}")
            return {
                "messages": [
                    AIMessage(
                        content=f"I encountered an error: {str(e)}. Let me try a different approach."
                    )
                ],
                "reasoning_steps": reasoning_steps + 1,
                "tools_used": tools_used,
                "error": str(e),
            }

    return agent_node


def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """
    Routing function for the ReAct loop.

    Determines whether to:
    - Execute tools (if there are pending tool calls)
    - End (if no tool calls or max steps reached)

    Args:
        state: Current agent state.

    Returns:
        "tools" if there are tool calls to execute, "end" otherwise.
    """
    messages = state.get("messages", [])

    if not messages:
        return "end"

    # Check if max steps reached
    if state.get("max_steps_reached", False):
        return "end"

    last_message = messages[-1]

    # Check for tool calls in the last message
    if (
        isinstance(last_message, AIMessage)
        and hasattr(last_message, "tool_calls")
        and last_message.tool_calls
    ):
        return "tools"

    return "end"


def format_response(state: AgentState) -> Dict[str, Any]:
    """
    Extract and format the final response from the conversation.

    Also extracts tool usage information from the conversation history.

    Args:
        state: Final agent state.

    Returns:
        State update with formatted response and tool usage info.
    """
    messages = state.get("messages", [])
    tools_used = list(state.get("tools_used", []) or [])

    # If tools_used is empty, try to extract from message history
    if not tools_used:
        for msg in messages:
            # Check AI messages for tool calls
            if (
                isinstance(msg, AIMessage)
                and hasattr(msg, "tool_calls")
                and msg.tool_calls
            ):
                for tc in msg.tool_calls:
                    tool_name = tc.get("name", "unknown")
                    if tool_name not in tools_used:
                        tools_used.append(tool_name)
            # Check ToolMessages for tool names
            elif isinstance(msg, ToolMessage) and hasattr(msg, "name"):
                if msg.name and msg.name not in tools_used:
                    tools_used.append(msg.name)

    if not messages:
        return {
            "response": "I wasn't able to generate a response. Please try again.",
            "tools_used": tools_used,
            "tool_used": tools_used[0] if tools_used else None,
        }

    # Find the last AI message (the final response)
    response_content = None
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            content = msg.content
            if isinstance(content, str) and content.strip():
                response_content = content.strip()
                break
            elif content:
                response_content = str(content)
                break

    if not response_content:
        response_content = "I wasn't able to generate a response. Please try again."

    return {
        "response": response_content,
        "tools_used": tools_used,
        "tool_used": tools_used[0] if tools_used else None,  # Primary tool (first used)
    }


def error_node(state: AgentState) -> Dict[str, Any]:
    """
    Handle errors gracefully.

    This node is called when an error occurs in the workflow.
    It provides a user-friendly error message.

    Args:
        state: Current agent state with error information.

    Returns:
        State update with error response.
    """
    error = state.get("error", "Unknown error occurred")

    logger.error(f"Error in agent workflow: {error}")

    return {
        "response": f"I encountered an error while processing your request: {error}. "
        f"You might want to try rephrasing your question or check if the required "
        f"services are available.",
        "error": error,
    }


def extract_tool_results(state: AgentState) -> Dict[str, Any]:
    """
    Extract and store tool results for observability.

    This node processes tool messages and extracts results
    for tracking and debugging purposes.

    Args:
        state: Current state with tool messages.

    Returns:
        State update with extracted tool results.
    """
    messages = state.get("messages", [])
    existing_results = state.get("tool_results", []) or []
    tool_results = list(existing_results)

    # Find tool messages
    for msg in messages:
        if isinstance(msg, ToolMessage):
            tool_name = msg.name if hasattr(msg, "name") else "unknown"
            tool_output = str(msg.content) if msg.content else ""

            # Create a ToolResult compatible dict
            result = {
                "tool_name": tool_name or "unknown",
                "tool_input": {},  # Not available from ToolMessage
                "tool_output": tool_output,
                "success": True,  # Assume success if we got a message
                "error": None,
            }
            tool_results.append(result)  # type: ignore

    return {"tool_results": tool_results}


# Export node factory functions and routing functions
__all__ = [
    "create_input_node",
    "create_agent_node",
    "should_continue",
    "format_response",
    "error_node",
    "extract_tool_results",
]
