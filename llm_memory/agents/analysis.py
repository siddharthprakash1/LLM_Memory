"""
Analysis Agent - Reasoning and Comparison Specialist.

Responsible for:
- Complex reasoning
- Comparisons and evaluations
- Decision making
- Data analysis
"""

import logging
from typing import Any

from langchain_core.messages import HumanMessage

from .base import BaseAgent
from .state import AgentState, Message
from .config import AgentConfig

logger = logging.getLogger(__name__)


ANALYSIS_SYSTEM_PROMPT = """You are an Analysis Agent specializing in reasoning and evaluation.

## Your Capabilities

1. **Comparative Analysis** - Compare options, pros/cons
2. **Root Cause Analysis** - Identify underlying issues
3. **Decision Making** - Evaluate choices systematically
4. **Data Interpretation** - Extract insights from data
5. **Multi-hop Reasoning** - Connect multiple pieces of information

## Framework for Analysis

1. **Understand** - What exactly needs to be analyzed?
2. **Gather** - What information is available?
3. **Analyze** - Apply relevant frameworks
4. **Conclude** - What are the key insights?
5. **Recommend** - What actions should be taken?

## Output Format

Structure your analysis as:
1. **Summary** - Brief overview of findings
2. **Analysis** - Detailed reasoning
3. **Conclusion** - Key takeaways
4. **Recommendations** - Suggested next steps

## Memory Integration

Use knowledge from memory to:
- Draw on past analyses
- Connect related concepts
- Apply learned patterns
"""


class AnalysisAgent(BaseAgent):
    """
    Analysis Agent for complex reasoning tasks.
    
    Uses Memory V3's knowledge graph for:
    - Multi-hop reasoning
    - Connecting related concepts
    - Building on past analyses
    """
    
    def __init__(self, config: AgentConfig):
        """Initialize the Analysis Agent."""
        super().__init__(
            name="analysis",
            description="Performs complex reasoning, comparisons, and evaluations",
            system_prompt=ANALYSIS_SYSTEM_PROMPT,
            config=config,
            tools=None,
        )
    
    def invoke(self, state: AgentState) -> dict[str, Any]:
        """Execute analysis task."""
        logger.info(f"Analysis Agent starting")
        
        current_subtask = state.get("agent_scratchpad", {}).get("current_subtask", state["task"])
        
        # Get memory context with knowledge graph
        memory_context = self._get_memory_context(current_subtask)
        
        # Also get specific KG facts if entities are mentioned
        kg_facts = []
        if self.memory:
            entities = self.memory.entity_extractor.extract(current_subtask)
            for etype, elist in entities.items():
                for entity in elist[:3]:
                    facts = self._get_knowledge_graph_facts(entity)
                    kg_facts.extend(facts[:5])
        
        kg_section = ""
        if kg_facts:
            kg_section = "\n\nKnowledge Graph Facts:\n" + "\n".join(f"- {f}" for f in kg_facts[:10])
        
        messages = self._convert_messages(state)
        
        # Include previous agent artifacts
        artifacts_context = ""
        if state.get("artifacts"):
            artifacts_context = "\n\n## Information from Other Agents:\n"
            for agent, artifact in state["artifacts"].items():
                if isinstance(artifact, dict):
                    artifacts_context += f"\n### {agent}:\n{artifact.get('summary', str(artifact))[:500]}\n"
        
        messages.append(HumanMessage(content=f"""
Please analyze the following:

**Task:** {current_subtask}

**Original Context:** {state['task']}
{memory_context}
{kg_section}
{artifacts_context}

Provide a thorough analysis with clear reasoning and actionable conclusions.
"""))
        
        # Get analysis response
        response = self.llm.invoke(messages)
        
        # Store analysis in memory
        self._remember(
            f"Analysis of '{current_subtask[:100]}': {response.content[:300]}",
            importance=0.7,
            memory_type="fact",
        )
        
        # Create analysis artifact
        analysis_artifact = {
            "subtask": current_subtask,
            "analysis": response.content,
            "kg_facts_used": len(kg_facts),
            "memory_context_used": bool(memory_context),
        }
        
        agent_message = Message(
            role="assistant",
            content=f"[Analysis Agent]\n\n{response.content}",
            name="analysis",
        )
        
        completed = state.get("completed_subtasks", [])
        if current_subtask and current_subtask not in completed:
            completed = completed + [current_subtask]
        
        return {
            "current_agent": "analysis",
            "artifacts": {"analysis": analysis_artifact},
            "messages": agent_message,
            "completed_subtasks": completed,
        }
