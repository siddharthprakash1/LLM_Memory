"""
Agent State for LangGraph Framework.

This defines the shared state passed between nodes in the graph.
"""

from typing import List, Dict, Any, Optional, Annotated
from typing_extensions import TypedDict
import operator
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    """
    The state of the agent workflow.
    
    Attributes:
        messages: The conversation history (list of messages)
        user_id: The ID of the user interacting with the agent
        memory_context: Context retrieved from long-term memory
        next_step: The next step in the plan (if any)
        scratchpad: Temporary storage for intermediate reasoning
    """
    # Append-only list of messages
    messages: Annotated[List[BaseMessage], operator.add]
    
    # Static context
    user_id: str
    
    # Dynamic state
    memory_context: Optional[str]
    scratchpad: Dict[str, Any]
