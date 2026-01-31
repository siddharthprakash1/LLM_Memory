"""
Writer Agent - Content Creation and Formatting Specialist.

Responsible for:
- Formatting outputs
- Creating summaries
- Writing documentation
- Polishing content
"""

import logging
from typing import Any

from langchain_core.messages import HumanMessage

from .base import BaseAgent
from .state import AgentState, Message
from .config import AgentConfig

logger = logging.getLogger(__name__)


WRITER_SYSTEM_PROMPT = """You are a Writer Agent specializing in clear, polished content creation.

## Your Capabilities

1. **Formatting** - Structure content with headers, lists, code blocks
2. **Summarization** - Distill complex information into key points
3. **Documentation** - Create clear technical documentation
4. **Editing** - Improve clarity and readability
5. **Adaptation** - Adjust tone for different audiences

## Writing Principles

1. **Clarity** - Use simple, direct language
2. **Structure** - Organize with headers and sections
3. **Completeness** - Address all aspects of the request
4. **Accuracy** - Ensure factual correctness
5. **Engagement** - Keep the reader interested

## Output Format

Your output should be:
- Well-structured with markdown formatting
- Clear and concise
- Complete and actionable
- Professional in tone

## Use Available Information

Incorporate findings from:
- Research agent's discoveries
- Code agent's solutions
- Analysis agent's insights
- Memory context for personalization
"""


class WriterAgent(BaseAgent):
    """
    Writer Agent for content creation and formatting.
    
    Uses Memory V3 to:
    - Remember writing style preferences
    - Reference past documents
    - Maintain consistency
    """
    
    def __init__(self, config: AgentConfig):
        """Initialize the Writer Agent."""
        super().__init__(
            name="writer",
            description="Creates polished content, summaries, and documentation",
            system_prompt=WRITER_SYSTEM_PROMPT,
            config=config,
            tools=None,
        )
    
    def invoke(self, state: AgentState) -> dict[str, Any]:
        """Execute writing task."""
        logger.info(f"Writer Agent starting")
        
        current_subtask = state.get("agent_scratchpad", {}).get("current_subtask", state["task"])
        
        # Gather all artifacts from other agents
        artifacts_content = ""
        if state.get("artifacts"):
            artifacts_content = "\n## Available Information:\n"
            
            for agent, artifact in state["artifacts"].items():
                artifacts_content += f"\n### From {agent}:\n"
                if isinstance(artifact, dict):
                    if "summary" in artifact:
                        artifacts_content += artifact["summary"][:1000]
                    elif "analysis" in artifact:
                        artifacts_content += artifact["analysis"][:1000]
                    elif "code" in artifact:
                        artifacts_content += artifact["code"][:1000]
                    else:
                        artifacts_content += str(artifact)[:1000]
                else:
                    artifacts_content += str(artifact)[:1000]
                artifacts_content += "\n"
        
        # Get memory context
        memory_context = self._get_memory_context(current_subtask)
        
        messages = self._convert_messages(state)
        
        messages.append(HumanMessage(content=f"""
Please create polished content for:

**Task:** {current_subtask}

**Original Request:** {state['task']}
{artifacts_content}
{memory_context}

Create well-formatted, clear, and complete content that addresses the request.
"""))
        
        # Get writer response
        response = self.llm.invoke(messages)
        
        # Store written content summary in memory
        self._remember(
            f"Document created for '{current_subtask[:50]}': {response.content[:200]}",
            importance=0.5,
            memory_type="event",
        )
        
        # Create writer artifact
        writer_artifact = {
            "subtask": current_subtask,
            "content": response.content,
            "artifacts_used": list(state.get("artifacts", {}).keys()),
        }
        
        agent_message = Message(
            role="assistant",
            content=f"[Writer Agent]\n\n{response.content}",
            name="writer",
        )
        
        completed = state.get("completed_subtasks", [])
        if current_subtask and current_subtask not in completed:
            completed = completed + [current_subtask]
        
        return {
            "current_agent": "writer",
            "artifacts": {"writer": writer_artifact},
            "messages": agent_message,
            "completed_subtasks": completed,
        }
