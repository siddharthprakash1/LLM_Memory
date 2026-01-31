"""
State Management for Multi-Agent System.

Defines the shared state that flows through all agents in the graph.
"""

from typing import Annotated, Any, Literal, TypedDict
from dataclasses import dataclass, field
from datetime import datetime
import operator


@dataclass
class Message:
    """Represents a message in the conversation."""
    role: Literal["user", "assistant", "system", "tool"]
    content: str
    name: str | None = None
    tool_call_id: str | None = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert message to dictionary format for LLM."""
        msg = {"role": self.role, "content": self.content}
        if self.name:
            msg["name"] = self.name
        if self.tool_call_id:
            msg["tool_call_id"] = self.tool_call_id
        return msg


def add_messages(left: list[Message], right: list[Message] | Message) -> list[Message]:
    """Reducer function to append messages to the conversation history."""
    if isinstance(right, Message):
        return left + [right]
    return left + right


def merge_artifacts(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """Reducer function to merge artifacts from different agents."""
    merged = left.copy()
    merged.update(right)
    return merged


class AgentState(TypedDict):
    """
    Shared state across all agents in the graph.
    
    Attributes:
        messages: Full conversation history with reducer to append
        current_agent: Which agent is currently processing
        next_agent: Which agent should process next (set by supervisor)
        task: The current task being worked on
        artifacts: Shared outputs/results from agents
        iteration: Current iteration count
        max_iterations: Maximum allowed iterations
        human_feedback: Optional human input during execution
        needs_human_approval: Flag to pause for human review
        error: Any error that occurred during processing
        final_response: The compiled final response to user
        agent_scratchpad: Temporary working memory for agents
        subtasks: List of planned subtasks
        completed_subtasks: List of completed subtasks
    """
    # Core conversation state
    messages: Annotated[list[Message], add_messages]
    
    # Routing state
    current_agent: str
    next_agent: str | None
    
    # Task state
    task: str
    subtasks: list[str]
    completed_subtasks: list[str]
    
    # Shared artifacts between agents
    artifacts: Annotated[dict[str, Any], merge_artifacts]
    
    # Control flow
    iteration: int
    max_iterations: int
    
    # Human-in-the-loop
    human_feedback: str | None
    needs_human_approval: bool
    
    # Results
    error: str | None
    final_response: str | None
    
    # Working memory
    agent_scratchpad: dict[str, Any]


def create_initial_state(
    task: str,
    max_iterations: int = 10,
) -> AgentState:
    """Create the initial state for a new conversation."""
    return AgentState(
        messages=[
            Message(
                role="user",
                content=task,
                name="user",
            )
        ],
        current_agent="supervisor",
        next_agent=None,
        task=task,
        subtasks=[],
        completed_subtasks=[],
        artifacts={},
        iteration=0,
        max_iterations=max_iterations,
        human_feedback=None,
        needs_human_approval=False,
        error=None,
        final_response=None,
        agent_scratchpad={},
    )


class SupervisorDecision(TypedDict):
    """The supervisor's routing decision."""
    next_agent: Literal["research", "code", "analysis", "writer", "FINISH"]
    reasoning: str
    subtask: str | None
