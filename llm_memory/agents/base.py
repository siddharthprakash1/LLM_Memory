"""
Base Agent Class for Multi-Agent System.

Provides common functionality for all agents including:
- LLM initialization
- Tool binding
- Memory integration (V3 with Knowledge Graph)
- Message handling
"""

import logging
from abc import ABC, abstractmethod
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from .state import AgentState, Message
from .config import AgentConfig, ModelConfig
from ..memory_v3 import HierarchicalMemoryV3

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the multi-agent system.
    
    Provides common functionality:
    - LLM initialization based on config
    - Tool binding
    - Memory V3 integration with Knowledge Graph
    - Standardized invoke pattern
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        system_prompt: str,
        config: AgentConfig,
        tools: list[BaseTool] | None = None,
    ):
        """
        Initialize the base agent.
        
        Args:
            name: Unique name for this agent
            description: What this agent does (used by supervisor for routing)
            system_prompt: System prompt that defines agent behavior
            config: Configuration object
            tools: List of tools available to this agent
        """
        self.name = name
        self.description = description
        self.system_prompt = system_prompt
        self.config = config
        self.tools = tools or []
        
        # Memory V3 with Knowledge Graph
        self.memory: HierarchicalMemoryV3 | None = None
        
        # Initialize the LLM
        self.llm = self._create_llm()
        
        # Bind tools to LLM if any
        if self.tools:
            self.llm_with_tools = self.llm.bind_tools(self.tools)
        else:
            self.llm_with_tools = self.llm
        
        logger.info(f"Initialized agent: {self.name}")
    
    def set_memory(self, memory: HierarchicalMemoryV3):
        """Attach a Memory V3 system to this agent."""
        self.memory = memory
        logger.info(f"Memory attached to agent: {self.name}")
    
    def _get_memory_context(self, query: str) -> str:
        """
        Get relevant memory context using V3's advanced retrieval.
        
        Uses:
        - Query decomposition for multi-hop
        - Knowledge graph lookup
        - Re-ranking
        """
        if not self.memory:
            return ""
        
        context = self.memory.get_context(query, max_chars=2000, include_kg=True)
        if context:
            return f"\n\n## Memory Context (from Knowledge Graph)\n{context}\n"
        return ""
    
    def _remember(
        self,
        content: str,
        importance: float = 0.5,
        memory_type: str = "fact",
    ):
        """
        Store something in memory with entity extraction.
        
        Memory V3 automatically:
        - Extracts entities
        - Builds knowledge graph
        - Creates searchable embeddings
        """
        if self.memory and content:
            self.memory.add(content, memory_type=memory_type, importance=importance)
    
    def _search_memory(self, query: str, top_k: int = 5) -> list:
        """
        Advanced search with multi-hop reasoning.
        
        Returns:
            List of (MemoryItem, score) tuples
        """
        if not self.memory:
            return []
        return self.memory.search(query, top_k=top_k, use_decomposition=True, use_reranking=True)
    
    def _get_knowledge_graph_facts(self, entity: str) -> list[str]:
        """Get facts from knowledge graph about an entity."""
        if not self.memory:
            return []
        return self.memory.knowledge_graph.get_facts_about(entity, as_text=True)
    
    def _create_llm(self) -> BaseChatModel:
        """Create the LLM instance based on configuration."""
        model_config = self.config.get_model_for_agent(self.name)
        
        if model_config.provider == "ollama":
            from langchain_ollama import ChatOllama
            return ChatOllama(
                model=model_config.model_name,
                temperature=model_config.temperature,
                base_url=model_config.base_url or "http://localhost:11434",
            )
        
        elif model_config.provider == "google":
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model=model_config.model_name,
                temperature=model_config.temperature,
                max_output_tokens=model_config.max_tokens,
                google_api_key=model_config.get_api_key() or None,
            )
        
        elif model_config.provider == "openai":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=model_config.model_name,
                temperature=model_config.temperature,
                max_tokens=model_config.max_tokens,
                api_key=model_config.get_api_key() or None,
            )
        
        elif model_config.provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(
                model=model_config.model_name,
                temperature=model_config.temperature,
                max_tokens=model_config.max_tokens,
                api_key=model_config.get_api_key() or None,
            )
        
        else:
            raise ValueError(f"Unsupported provider: {model_config.provider}")
    
    def _convert_messages(self, state: AgentState) -> list:
        """Convert state messages to LangChain message format with memory context."""
        # Get memory context for the current task
        task = state.get("task", "")
        memory_context = self._get_memory_context(task)
        
        # Build system prompt with memory
        full_system_prompt = self.system_prompt
        if memory_context:
            full_system_prompt += memory_context
        
        messages = [SystemMessage(content=full_system_prompt)]
        
        for msg in state["messages"]:
            if isinstance(msg, Message):
                if msg.role == "user":
                    messages.append(HumanMessage(content=msg.content))
                elif msg.role == "assistant":
                    messages.append(AIMessage(content=msg.content))
                elif msg.role == "system":
                    messages.append(SystemMessage(content=msg.content))
            elif isinstance(msg, dict):
                if msg.get("role") == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg.get("role") == "assistant":
                    messages.append(AIMessage(content=msg["content"]))
        
        return messages
    
    def _handle_tool_calls(self, ai_message: AIMessage) -> list[dict]:
        """Execute tool calls and return results."""
        results = []
        
        if not hasattr(ai_message, 'tool_calls') or not ai_message.tool_calls:
            return results
        
        for tool_call in ai_message.tool_calls:
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("args", {})
            
            tool = next((t for t in self.tools if t.name == tool_name), None)
            
            if tool:
                try:
                    output = tool.invoke(tool_args)
                    results.append({
                        "tool": tool_name,
                        "args": tool_args,
                        "output": output,
                        "success": True
                    })
                    logger.info(f"Tool {tool_name} executed successfully")
                except Exception as e:
                    results.append({
                        "tool": tool_name,
                        "args": tool_args,
                        "output": f"Error: {str(e)}",
                        "success": False
                    })
                    logger.error(f"Tool {tool_name} failed: {e}")
            else:
                results.append({
                    "tool": tool_name,
                    "output": f"Tool not found: {tool_name}",
                    "success": False
                })
        
        return results
    
    @abstractmethod
    def invoke(self, state: AgentState) -> dict[str, Any]:
        """
        Process the current state and return updates.
        
        Args:
            state: Current graph state
            
        Returns:
            Dictionary of state updates
        """
        pass
    
    async def ainvoke(self, state: AgentState) -> dict[str, Any]:
        """Async version of invoke."""
        return self.invoke(state)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, tools={len(self.tools)})"
