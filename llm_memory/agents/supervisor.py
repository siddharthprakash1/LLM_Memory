"""
Supervisor Agent - The Orchestrator.

Responsible for:
- Analyzing user requests
- Breaking down into subtasks
- Routing to appropriate specialist agents
- Compiling final responses
"""

import json
import logging
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from .base import BaseAgent
from .state import AgentState, Message
from .config import AgentConfig

logger = logging.getLogger(__name__)


class RouteDecision(BaseModel):
    """Structured output for routing decisions."""
    next_agent: Literal["research", "code", "analysis", "writer", "FINISH"] = Field(
        description="The next agent to route to, or FINISH if task is complete"
    )
    reasoning: str = Field(
        description="Brief explanation of why this agent was chosen"
    )
    subtask: str | None = Field(
        default=None,
        description="Specific subtask for the chosen agent"
    )


SUPERVISOR_SYSTEM_PROMPT = """You are a Supervisor Agent orchestrating specialized AI agents. Be DECISIVE and EFFICIENT.

## Your Team

1. **research** - Web/news search, Wikipedia. Use for: current events, facts, external info
2. **code** - Python execution, debugging. Use for: programming, calculations, data processing
3. **analysis** - Reasoning, comparisons. Use for: decisions, evaluations, complex reasoning
4. **writer** - Formatting, summaries. Use for: polished final outputs, documentation

## CRITICAL RULES

1. **Be decisive** - Pick ONE agent and commit
2. **Don't over-delegate** - Simple tasks need 1-2 agents max
3. **FINISH quickly** - If any agent has provided useful results, consider finishing
4. **Don't loop** - Never route to the same agent twice for the same subtask
5. **Use memory** - Check memory context for relevant past information

## Quick Decision Guide

- User wants NEWS/CURRENT INFO → research (once), then FINISH
- User wants CODE → code (once), then FINISH
- User wants ANALYSIS/COMPARISON → analysis (once), then FINISH
- User wants FORMATTED OUTPUT → writer (once), then FINISH
- Complex task → research → analysis or code → FINISH

## Response Format

You must respond with EXACTLY this JSON format:
```json
{
    "next_agent": "research" | "code" | "analysis" | "writer" | "FINISH",
    "reasoning": "One sentence why",
    "subtask": "Specific task (null if FINISH)"
}
```

## IMPORTANT

- After research returns results, usually FINISH (don't re-research)
- After 2-3 iterations, strongly prefer FINISH
- Trust the agents' work - don't second-guess
"""


class SupervisorAgent(BaseAgent):
    """
    The Supervisor Agent orchestrates the multi-agent workflow.
    
    Uses Memory V3 to:
    - Remember past tasks and their outcomes
    - Use knowledge graph for context
    - Make informed routing decisions
    """
    
    def __init__(self, config: AgentConfig):
        """Initialize the Supervisor Agent."""
        super().__init__(
            name="supervisor",
            description="Orchestrates the multi-agent workflow by routing tasks to specialists",
            system_prompt=SUPERVISOR_SYSTEM_PROMPT,
            config=config,
            tools=None,
        )
        
        # Try to set up structured output
        try:
            self.structured_llm = self.llm.with_structured_output(RouteDecision)
        except Exception:
            self.structured_llm = None
            logger.warning("Structured output not available, using regex parsing")
    
    def _build_context(self, state: AgentState) -> str:
        """Build context from current state for decision making."""
        context_parts = []
        
        # Original task
        context_parts.append(f"## Original Task\n{state['task']}")
        
        # Subtasks status
        if state.get("subtasks"):
            context_parts.append("\n## Planned Subtasks")
            for i, subtask in enumerate(state["subtasks"], 1):
                status = "✓" if subtask in state.get("completed_subtasks", []) else "○"
                context_parts.append(f"{status} {i}. {subtask}")
        
        # Artifacts from other agents
        if state.get("artifacts"):
            context_parts.append("\n## Completed Work")
            for agent, result in state["artifacts"].items():
                context_parts.append(f"\n### From {agent}:")
                if isinstance(result, dict):
                    summary = result.get("summary", str(result))[:800]
                    context_parts.append(summary)
                else:
                    context_parts.append(str(result)[:800])
        
        # Iteration info
        context_parts.append(f"\n## Progress")
        context_parts.append(f"Iteration: {state.get('iteration', 0)} / {state.get('max_iterations', 10)}")
        
        # Human feedback if any
        if state.get("human_feedback"):
            context_parts.append(f"\n## Human Feedback\n{state['human_feedback']}")
        
        return "\n".join(context_parts)
    
    def invoke(self, state: AgentState) -> dict[str, Any]:
        """
        Analyze state and decide which agent should handle the next subtask.
        """
        logger.info(f"Supervisor analyzing task (iteration {state.get('iteration', 0)})")
        
        # Build context for decision
        context = self._build_context(state)
        
        # Add memory context
        memory_context = self._get_memory_context(state["task"])
        if memory_context:
            context = memory_context + "\n" + context
        
        # Create message for structured decision
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"""Based on the following context, decide the next action.
Respond with ONLY a JSON object in the exact format specified.

{context}""")
        ]
        
        try:
            if self.structured_llm:
                decision: RouteDecision = self.structured_llm.invoke(messages)
            else:
                # Fallback to parsing
                response = self.llm.invoke(messages)
                decision = self._parse_decision(response.content)
            
            logger.info(f"Supervisor decision: {decision.next_agent} - {decision.reasoning}")
            
            # Remember this decision
            self._remember(
                f"Task routing: '{state['task'][:100]}' -> {decision.next_agent}",
                importance=0.3,
                memory_type="event",
            )
            
            # Create supervisor message
            supervisor_message = Message(
                role="assistant",
                content=f"[Supervisor] Routing to {decision.next_agent}: {decision.reasoning}",
                name="supervisor",
                metadata={
                    "decision": decision.model_dump() if hasattr(decision, 'model_dump') else str(decision),
                    "iteration": state.get("iteration", 0)
                }
            )
            
            # Build state updates
            updates: dict[str, Any] = {
                "next_agent": decision.next_agent.lower(),
                "current_agent": "supervisor",
                "messages": supervisor_message,
                "iteration": state.get("iteration", 0) + 1,
            }
            
            # Add subtask to list if new
            if decision.subtask and decision.subtask not in state.get("subtasks", []):
                updates["subtasks"] = state.get("subtasks", []) + [decision.subtask]
            
            # Update scratchpad with current subtask
            updates["agent_scratchpad"] = {
                **state.get("agent_scratchpad", {}),
                "current_subtask": decision.subtask,
            }
            
            return updates
            
        except Exception as e:
            logger.error(f"Supervisor error: {e}")
            
            return {
                "next_agent": "FINISH",
                "current_agent": "supervisor",
                "error": f"Supervisor error: {str(e)}",
                "messages": Message(
                    role="assistant",
                    content=f"[Supervisor] Error occurred: {str(e)}. Finishing task.",
                    name="supervisor"
                )
            }
    
    def _parse_decision(self, content: str) -> RouteDecision:
        """Parse decision from LLM response text."""
        import re
        
        # Try to find JSON
        json_match = re.search(r'\{[^}]+\}', content, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                return RouteDecision(**data)
            except Exception:
                pass
        
        # Fallback: look for keywords
        content_lower = content.lower()
        
        if "finish" in content_lower:
            return RouteDecision(next_agent="FINISH", reasoning="Task appears complete")
        elif "research" in content_lower:
            return RouteDecision(next_agent="research", reasoning="Need to gather information")
        elif "code" in content_lower:
            return RouteDecision(next_agent="code", reasoning="Need to write/execute code")
        elif "analysis" in content_lower:
            return RouteDecision(next_agent="analysis", reasoning="Need to analyze")
        elif "writer" in content_lower:
            return RouteDecision(next_agent="writer", reasoning="Need to format output")
        else:
            return RouteDecision(next_agent="FINISH", reasoning="Could not determine next step")
    
    def compile_final_response(self, state: AgentState) -> str:
        """Compile all agent outputs into a final response."""
        logger.info("Compiling final response")
        
        compilation_prompt = f"""You are compiling the final response for the user.

## Original Request
{state['task']}

## Work Completed by Agents
{json.dumps(state.get('artifacts', {}), indent=2, default=str)}

## Instructions
Create a comprehensive, well-formatted response that:
1. Directly addresses the user's original request
2. Incorporates all relevant findings from the agents
3. Is clear, organized, and actionable
4. Uses appropriate formatting (headers, lists, code blocks)

Provide the final response:"""

        messages = [
            SystemMessage(content="You are a helpful assistant compiling a final response."),
            HumanMessage(content=compilation_prompt)
        ]
        
        try:
            response = self.llm.invoke(messages)
            final_response = response.content
            
            # Remember the completion
            self._remember(
                f"Completed task: {state['task'][:200]}",
                importance=0.6,
                memory_type="event",
            )
            
            return final_response
            
        except Exception as e:
            logger.error(f"Error compiling response: {e}")
            
            return f"""## Task Completed

### Original Request
{state['task']}

### Results
{json.dumps(state.get('artifacts', {}), indent=2, default=str)}

Note: Auto-compiled due to error in final formatting.
"""
