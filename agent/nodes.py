import logging
from typing import Any, Dict, Literal

from agent.config import get_agent_config
from agent.prompts import get_system_prompt
from agent.state import AgentState
from agent.tools import get_all_tools
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_ollama import ChatOllama

# Setup logging
logger = logging.getLogger(__name__)


def create_agent_node():
    """
    Create the agent node function with LLM and tools bound.

    Returns:
        Agent node function that can be used in the graph.
    """
    config = get_agent_config()

    # Initialize LLM
    llm = ChatOllama(
        model=config.llm_model,
        temperature=config.temperature,
    )

    # Bind tools to LLM
    tools = get_all_tools()
    llm_with_tools = llm.bind_tools(tools)

    def agent_node(state: AgentState) -> Dict[str, Any]:
        """
        The main agent reasoning node.

        This node:
        1. Takes the current state (messages, context)
        2. Invokes the LLM with tools to decide next action
        3. Returns updated state with the LLM's response

        The LLM will either:
        - Generate tool calls (to be executed by tool_node)
        - Generate a final response (ending the loop)

        Args:
            state: Current agent state with messages and context.

        Returns:
            Updated state with new message from LLM.
        """
        messages = list(state.get("messages", []))
        reasoning_steps = state.get("reasoning_steps", 0)
        config = get_agent_config()

        # Add system prompt if not present
        if not messages or not isinstance(messages[0], SystemMessage):
            system_prompt = get_system_prompt(concise=False)
            messages.insert(0, SystemMessage(content=system_prompt))

        # Check if we've hit max reasoning steps
        if reasoning_steps >= config.max_reasoning_steps:
            logger.warning(
                f"Agent reached maximum reasoning steps ({config.max_reasoning_steps})"
            )
            return {
                "messages": [
                    AIMessage(
                        content="I've reached the maximum number of reasoning steps. "
                        "Based on the information gathered so far, here's what I found: "
                        f"[Summary of {reasoning_steps} steps of reasoning]"
                    )
                ],
                "reasoning_steps": reasoning_steps,
                "max_steps_reached": True,
            }

        try:
            # Invoke LLM with tools
            response = llm_with_tools.invoke(messages)

            # Track tool usage for observability
            tool_used = None
            if hasattr(response, "tool_calls") and response.tool_calls:
                tool_used = (
                    response.tool_calls[0].get("name") if response.tool_calls else None
                )
                if config.log_tool_calls:
                    logger.info(f"Agent selected tool: {tool_used}")
                    for tc in response.tool_calls:
                        logger.debug(f"Tool call: {tc}")

            return {
                "messages": [response],
                "reasoning_steps": reasoning_steps + 1,
                "tool_used": tool_used,
            }

        except Exception as e:
            logger.error(f"Agent node error: {str(e)}")
            return {
                "messages": [
                    AIMessage(
                        content=f"I encountered an error while processing: {str(e)}. "
                        "Let me try a different approach."
                    )
                ],
                "reasoning_steps": reasoning_steps + 1,
                "error": str(e),
            }

    return agent_node


def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """
    Determine if the agent should continue to tools or end.

    This is the routing function that implements the ReAct loop control.
    It checks the last message to see if there are tool calls to execute.

    Args:
        state: Current agent state.

    Returns:
        "tools" if there are tool calls to execute, "end" otherwise.
    """
    messages = state.get("messages", [])

    if not messages:
        return "end"

    last_message = messages[-1]

    # Check if max steps reached
    if state.get("max_steps_reached", False):
        return "end"

    # Check for tool calls
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    # No tool calls means we're done
    return "end"


def response_node(state: AgentState) -> Dict[str, Any]:
    """
    Format the final response from the agent.

    This node extracts the final answer from the conversation
    and formats it for output.

    Args:
        state: Final agent state.

    Returns:
        State update with formatted response.
    """
    messages = state.get("messages", [])

    if not messages:
        return {
            "response": "I wasn't able to generate a response. Please try again.",
            "error": "No messages in state",
        }

    # Get the last AI message (the final response)
    last_message = messages[-1]

    if isinstance(last_message, AIMessage):
        response_content = last_message.content

        # Clean up the response if needed
        if isinstance(response_content, str):
            response = response_content.strip()
        else:
            response = str(response_content)

        return {"response": response}

    # If the last message isn't from the AI, something went wrong
    return {
        "response": "I encountered an issue generating the response.",
        "error": f"Last message was not from AI: {type(last_message).__name__}",
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
    query = state.get("query", "")

    logger.error(f"Error in agent workflow: {error}")

    return {
        "response": f"I encountered an error while processing your request: {error}. "
        f"You might want to try rephrasing your question or check if the required "
        f"services are available.",
        "error": error,
    }


def input_node(state: AgentState) -> Dict[str, Any]:
    """
    Process initial input and prepare state.

    This node takes the initial query and prepares it for the agent.

    Args:
        state: Initial state with query.

    Returns:
        State update with prepared messages.
    """
    query = state.get("query", "")
    collection_names = state.get("collection_names", [])

    # Build the user message with context
    user_message = query

    # Add collection context if provided
    if collection_names:
        collection_info = f"\n\n[Context: The user has specified these collections to search: {', '.join(collection_names)}]"
        user_message += collection_info

    return {
        "messages": [HumanMessage(content=user_message)],
        "reasoning_steps": 0,
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
    tool_results = state.get("tool_results", []) or []

    # Find tool messages
    for msg in messages:
        if isinstance(msg, ToolMessage):
            tool_results.append(
                {
                    "tool_name": msg.name if hasattr(msg, "name") else "unknown",
                    "tool_call_id": msg.tool_call_id
                    if hasattr(msg, "tool_call_id")
                    else None,
                    "content": msg.content,
                }
            )

    return {"tool_results": tool_results}


# Export node factory functions and routing functions
__all__ = [
    "create_agent_node",
    "should_continue",
    "response_node",
    "error_node",
    "input_node",
    "extract_tool_results",
]
