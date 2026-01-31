"""
Research Agent - Information Gathering Specialist.

Responsible for:
- Web searches
- Wikipedia lookups
- Fact verification
- External information gathering
"""

import logging
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from .base import BaseAgent
from .state import AgentState, Message
from .config import AgentConfig

logger = logging.getLogger(__name__)


# ============================================
# Research Tools
# ============================================

@tool
def web_search(query: str, num_results: int = 5) -> str:
    """
    Search the web for information.
    
    Args:
        query: Search query
        num_results: Number of results to return
    
    Returns:
        Search results as formatted text
    """
    try:
        from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
        search = DuckDuckGoSearchAPIWrapper(max_results=num_results)
        results = search.run(query)
        return f"Web search results for '{query}':\n{results}"
    except Exception as e:
        return f"Web search error: {e}. Using simulated results."


@tool
def wikipedia_search(query: str, sentences: int = 5) -> str:
    """
    Search Wikipedia for information.
    
    Args:
        query: Topic to search
        sentences: Number of sentences to return
        
    Returns:
        Wikipedia summary
    """
    try:
        from langchain_community.utilities import WikipediaAPIWrapper
        wiki = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=2000)
        return wiki.run(query)
    except Exception as e:
        return f"Wikipedia search error: {e}"


RESEARCH_TOOLS = [web_search, wikipedia_search]


RESEARCH_SYSTEM_PROMPT = """You are a Research Agent specializing in fast, efficient information gathering.

## Your Tools

1. **web_search** - Search the web (query, num_results=5)
2. **wikipedia_search** - Look up topics on Wikipedia (query, sentences=5)

## IMPORTANT: Be Efficient!

- Use ONLY 1-2 tool calls per request
- Pick the most relevant tool for the task
- For factual/encyclopedic info → use wikipedia_search
- For current events/general web → use web_search
- DO NOT call multiple search tools for the same query

## Output Format

After ONE search, immediately provide:
1. **Key Findings** - What you found (bullet points)
2. **Source** - Where the info came from

Keep it brief. Don't over-research - one good search is enough.

## Memory Integration

You have access to past conversation memory. Check if relevant information
already exists in memory before searching externally.
"""


class ResearchAgent(BaseAgent):
    """
    Research Agent for information gathering.
    
    Uses Memory V3 to:
    - Check past research results
    - Store new findings for future use
    - Build knowledge graph from research
    """
    
    def __init__(self, config: AgentConfig):
        """Initialize the Research Agent."""
        super().__init__(
            name="research",
            description="Gathers information through web search and Wikipedia",
            system_prompt=RESEARCH_SYSTEM_PROMPT,
            config=config,
            tools=RESEARCH_TOOLS,
        )
    
    def invoke(self, state: AgentState) -> dict[str, Any]:
        """Execute research based on the current subtask."""
        logger.info(f"Research Agent starting")
        
        # Get current subtask
        current_subtask = state.get("agent_scratchpad", {}).get("current_subtask", state["task"])
        
        # First, check memory for relevant info
        memory_results = self._search_memory(current_subtask, top_k=3)
        memory_context = ""
        if memory_results:
            memory_context = "\n\nRelevant information from memory:\n"
            for item, score in memory_results[:3]:
                memory_context += f"- {item.content[:200]}\n"
        
        # Build messages for LLM
        messages = self._convert_messages(state)
        
        messages.append(HumanMessage(content=f"""
Please research the following:

**Task:** {current_subtask}

**Original Context:** {state['task']}
{memory_context}
Use your tools to gather information. After researching, provide a structured summary.
"""))
        
        tool_results = []
        max_tool_iterations = 2
        
        for i in range(max_tool_iterations):
            response = self.llm_with_tools.invoke(messages)
            
            if hasattr(response, 'tool_calls') and response.tool_calls:
                results = self._handle_tool_calls(response)
                tool_results.extend(results)
                
                messages.append(response)
                for result in results:
                    messages.append(HumanMessage(
                        content=f"Tool Result ({result['tool']}): {result['output']}"
                    ))
            else:
                break
        
        # Get final response
        final_response = self.llm.invoke(messages)
        
        # Store findings in memory
        self._remember(
            f"Research: {current_subtask[:100]} -> {final_response.content[:300]}",
            importance=0.6,
            memory_type="fact",
        )
        
        # Create findings artifact
        findings = {
            "subtask": current_subtask,
            "summary": final_response.content,
            "tool_results": tool_results,
            "sources": [r.get("tool") for r in tool_results],
        }
        
        # Create message
        agent_message = Message(
            role="assistant",
            content=f"[Research Agent]\n\n{final_response.content}",
            name="research",
            metadata={"tool_calls": len(tool_results)}
        )
        
        # Update completed subtasks
        completed = state.get("completed_subtasks", [])
        if current_subtask and current_subtask not in completed:
            completed = completed + [current_subtask]
        
        return {
            "current_agent": "research",
            "artifacts": {"research": findings},
            "messages": agent_message,
            "completed_subtasks": completed,
        }
