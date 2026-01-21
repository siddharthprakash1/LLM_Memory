"""
LangChain integration for LLM Memory.

Provides:
- Memory class compatible with LangChain
- Chat history management
- Memory retriever for RAG
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from llm_memory.models.base import MemoryType, MemorySource
from llm_memory.models.short_term import ShortTermMemory, STMRole
from llm_memory.models.semantic import SemanticMemory
from llm_memory.api.memory_api import MemoryAPI, StoreRequest, SearchRequest


def _utcnow() -> datetime:
    """Get current UTC time (naive)."""
    return datetime.utcnow()


class Message(BaseModel):
    """A chat message."""

    role: str  # "human", "ai", "system"
    content: str
    timestamp: datetime = Field(default_factory=_utcnow)
    metadata: dict = Field(default_factory=dict)


class LangChainMemory:
    """
    LangChain-compatible memory class.
    
    Can be used as a drop-in replacement for LangChain memory classes.
    
    Usage:
        from llm_memory.api.integrations.langchain import LangChainMemory
        
        memory = LangChainMemory()
        
        # With LangChain
        from langchain.chains import ConversationChain
        chain = ConversationChain(llm=llm, memory=memory)
    """

    memory_key: str = "history"
    input_key: str = "input"
    output_key: str = "output"
    human_prefix: str = "Human"
    ai_prefix: str = "AI"
    return_messages: bool = False

    def __init__(
        self,
        memory_api: MemoryAPI | None = None,
        session_id: str | None = None,
        memory_key: str = "history",
        return_messages: bool = False,
    ):
        self.memory_api = memory_api or MemoryAPI()
        self.session_id = session_id or f"session_{_utcnow().timestamp()}"
        self.memory_key = memory_key
        self.return_messages = return_messages
        
        # Initialize STM for this session
        self._stm = ShortTermMemory(
            content=f"LangChain session {self.session_id}",
            session_id=self.session_id,
        )

    @property
    def memory_variables(self) -> List[str]:
        """Return memory variables."""
        return [self.memory_key]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Load memory variables for chain."""
        if self.return_messages:
            # Return as message objects
            messages = []
            for item in self._stm.items:
                msg = {
                    "role": item.role.value,
                    "content": item.content,
                }
                messages.append(msg)
            return {self.memory_key: messages}
        else:
            # Return as string
            history = self._format_history()
            return {self.memory_key: history}

    def _format_history(self) -> str:
        """Format chat history as string."""
        lines = []
        for item in self._stm.items:
            if item.role == STMRole.USER:
                lines.append(f"{self.human_prefix}: {item.content}")
            elif item.role == STMRole.ASSISTANT:
                lines.append(f"{self.ai_prefix}: {item.content}")
            else:
                lines.append(f"System: {item.content}")
        return "\n".join(lines)

    def save_context(
        self,
        inputs: Dict[str, Any],
        outputs: Dict[str, str],
    ) -> None:
        """Save context from chain run."""
        # Extract input and output
        input_str = inputs.get(self.input_key, "")
        output_str = outputs.get(self.output_key, "")
        
        if isinstance(output_str, dict):
            output_str = output_str.get("text", str(output_str))
        
        # Add to STM
        if input_str:
            self._stm.add_message(input_str, role=STMRole.USER)
        if output_str:
            self._stm.add_message(output_str, role=STMRole.ASSISTANT)

    def clear(self) -> None:
        """Clear memory contents."""
        self._stm.clear()

    def add_user_message(self, message: str) -> None:
        """Add a user message."""
        self._stm.add_message(message, role=STMRole.USER)

    def add_ai_message(self, message: str) -> None:
        """Add an AI message."""
        self._stm.add_message(message, role=STMRole.ASSISTANT)

    def add_message(self, message: Message) -> None:
        """Add a generic message."""
        role = STMRole.USER if message.role == "human" else STMRole.ASSISTANT
        self._stm.add_message(message.content, role=role)

    def get_messages(self) -> List[Message]:
        """Get all messages."""
        return [
            Message(
                role="human" if item.role == STMRole.USER else "ai",
                content=item.content,
                timestamp=item.created_at if hasattr(item, 'created_at') else _utcnow(),
            )
            for item in self._stm.items
        ]


class HierarchicalMemory(LangChainMemory):
    """
    Extended memory that leverages the full hierarchical memory system.
    
    Automatically retrieves relevant memories from episodic and semantic
    stores based on the current context.
    """

    def __init__(
        self,
        memory_api: MemoryAPI | None = None,
        session_id: str | None = None,
        memory_key: str = "history",
        context_key: str = "context",
        return_messages: bool = False,
        include_semantic: bool = True,
        include_episodic: bool = True,
        max_context_memories: int = 5,
    ):
        super().__init__(memory_api, session_id, memory_key, return_messages)
        self.context_key = context_key
        self.include_semantic = include_semantic
        self.include_episodic = include_episodic
        self.max_context_memories = max_context_memories

    @property
    def memory_variables(self) -> List[str]:
        """Return memory variables."""
        return [self.memory_key, self.context_key]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Load memory variables including relevant context."""
        # Get chat history
        result = super().load_memory_variables(inputs)
        
        # Get relevant context
        query = inputs.get(self.input_key, "")
        if query:
            context = self._retrieve_relevant_context(query)
            result[self.context_key] = context
        else:
            result[self.context_key] = ""
        
        return result

    def _retrieve_relevant_context(self, query: str) -> str:
        """Retrieve relevant memories for context."""
        memory_types = []
        if self.include_semantic:
            memory_types.append(MemoryType.SEMANTIC)
        if self.include_episodic:
            memory_types.append(MemoryType.EPISODIC)
        
        if not memory_types:
            return ""
        
        # Search for relevant memories
        request = SearchRequest(
            query=query,
            memory_types=memory_types,
            limit=self.max_context_memories,
        )
        
        results = self.memory_api.search(request)
        
        if not results.ranked_results:
            return ""
        
        # Format context
        context_lines = ["Relevant memories:"]
        for ranked in results.ranked_results[:self.max_context_memories]:
            content = ranked.result.content[:200]
            if len(ranked.result.content) > 200:
                content += "..."
            context_lines.append(f"- {content}")
        
        return "\n".join(context_lines)

    def save_context(
        self,
        inputs: Dict[str, Any],
        outputs: Dict[str, str],
    ) -> None:
        """Save context and potentially create long-term memories."""
        super().save_context(inputs, outputs)
        
        # Optionally save important exchanges to semantic memory
        output_str = outputs.get(self.output_key, "")
        if isinstance(output_str, dict):
            output_str = output_str.get("text", str(output_str))
        
        # Store if seems important (simple heuristic)
        if len(output_str) > 100 and any(
            keyword in output_str.lower()
            for keyword in ["prefer", "always", "never", "remember", "important"]
        ):
            self.memory_api.store(StoreRequest(
                content=output_str,
                memory_type=MemoryType.SEMANTIC,
                source=MemorySource.ASSISTANT_OUTPUT,
            ))


class MemoryRetriever:
    """
    Memory retriever for RAG (Retrieval-Augmented Generation).
    
    Can be used with LangChain's retrieval chains.
    
    Usage:
        retriever = MemoryRetriever(memory_api)
        docs = retriever.get_relevant_documents("What did we discuss?")
    """

    def __init__(
        self,
        memory_api: MemoryAPI,
        memory_types: List[MemoryType] | None = None,
        k: int = 5,
    ):
        self.memory_api = memory_api
        self.memory_types = memory_types or [
            MemoryType.SEMANTIC,
            MemoryType.EPISODIC,
        ]
        self.k = k

    def get_relevant_documents(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: The search query
            
        Returns:
            List of document dictionaries
        """
        request = SearchRequest(
            query=query,
            memory_types=self.memory_types,
            limit=self.k,
        )
        
        results = self.memory_api.search(request)
        
        documents = []
        for ranked in results.ranked_results:
            doc = {
                "page_content": ranked.result.content,
                "metadata": {
                    "memory_id": ranked.result.memory_id,
                    "memory_type": ranked.result.memory_type.value,
                    "similarity_score": ranked.result.similarity_score,
                    "relevance_score": ranked.result.relevance_score,
                },
            }
            documents.append(doc)
        
        return documents

    async def aget_relevant_documents(self, query: str) -> List[Dict[str, Any]]:
        """Async version of get_relevant_documents."""
        return self.get_relevant_documents(query)
