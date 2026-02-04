"""
Agent State for V5.
"""

from typing import Annotated, TypedDict, List, Optional
from langchain_core.messages import BaseMessage
import operator


class AgentStateV5(TypedDict):
    """State for the V5 agent."""
    messages: Annotated[List[BaseMessage], operator.add]
    user_id: str
    session_id: Optional[str]
    memory_context: str
    graph_context: str
    reasoning_log: List[str]
