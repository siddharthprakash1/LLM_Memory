"""
Tests for API and Memory System.
"""

import pytest
from datetime import datetime

from llm_memory.models.base import MemoryType, MemorySource
from llm_memory.models.short_term import STMRole
from llm_memory.models.episodic import EventType
from llm_memory.models.semantic import FactType

from llm_memory.api.memory_api import (
    MemoryAPI,
    MemoryOperation,
    OperationResult,
    StoreRequest,
    SearchRequest,
)
from llm_memory.api.hooks import (
    HookEvent,
    HookContext,
    HookRegistry,
    get_hook_registry,
    on_event,
    trigger_memory_created,
)
from llm_memory.api.memory_system import (
    MemorySystem,
    MemorySystemConfig,
)
from llm_memory.api.integrations.langchain import (
    Message,
    LangChainMemory,
    HierarchicalMemory,
    MemoryRetriever,
)


class TestMemoryAPI:
    """Tests for MemoryAPI."""

    def test_init(self):
        """Should initialize with defaults."""
        api = MemoryAPI()
        assert api is not None

    def test_store_semantic(self):
        """Should store semantic memory."""
        api = MemoryAPI()
        
        request = StoreRequest(
            content="Python is great",
            memory_type=MemoryType.SEMANTIC,
        )
        
        result = api.store(request)
        
        assert result.success is True
        assert result.operation == MemoryOperation.STORE
        assert result.memory_type == MemoryType.SEMANTIC

    def test_store_episodic(self):
        """Should store episodic memory."""
        api = MemoryAPI()
        
        request = StoreRequest(
            content="Debug session",
            memory_type=MemoryType.EPISODIC,
            event_type=EventType.ERROR,
        )
        
        result = api.store(request)
        
        assert result.success is True
        assert result.memory_type == MemoryType.EPISODIC

    def test_store_stm(self):
        """Should store short-term memory."""
        api = MemoryAPI()
        
        request = StoreRequest(
            content="Hello",
            memory_type=MemoryType.SHORT_TERM,
            role=STMRole.USER,
        )
        
        result = api.store(request)
        
        assert result.success is True

    def test_get_memory(self):
        """Should retrieve memory by ID."""
        api = MemoryAPI()
        
        request = StoreRequest(content="Test content")
        store_result = api.store(request)
        
        memory = api.get(store_result.memory_id)
        
        assert memory is not None
        assert memory.content == "Test content"

    def test_get_nonexistent(self):
        """Should return None for nonexistent memory."""
        api = MemoryAPI()
        
        memory = api.get("nonexistent_id")
        
        assert memory is None

    def test_update_memory(self):
        """Should update memory."""
        api = MemoryAPI()
        
        request = StoreRequest(content="Original")
        store_result = api.store(request)
        
        # Verify memory was stored
        memory_before = api.get(store_result.memory_id)
        assert memory_before is not None
        assert memory_before.content == "Original"
        
        update_result = api.update(store_result.memory_id, content="Updated")
        
        assert update_result.success is True
        
        memory = api.get(store_result.memory_id)
        assert memory.content == "Updated"

    def test_update_nonexistent(self):
        """Should fail for nonexistent memory."""
        api = MemoryAPI()
        
        result = api.update("nonexistent_id", content="Test")
        
        assert result.success is False

    def test_delete_soft(self):
        """Should soft delete memory."""
        api = MemoryAPI()
        
        request = StoreRequest(content="To delete")
        store_result = api.store(request)
        
        delete_result = api.delete(store_result.memory_id, soft=True)
        
        assert delete_result.success is True
        
        memory = api.get(store_result.memory_id)
        assert memory.is_active is False

    def test_delete_hard(self):
        """Should hard delete memory."""
        api = MemoryAPI()
        
        request = StoreRequest(content="To delete")
        store_result = api.store(request)
        
        delete_result = api.delete(store_result.memory_id, soft=False)
        
        assert delete_result.success is True
        
        memory = api.get(store_result.memory_id)
        assert memory is None

    def test_search(self):
        """Should search memories."""
        api = MemoryAPI()
        
        # Store some memories
        api.store(StoreRequest(content="Python programming"))
        api.store(StoreRequest(content="JavaScript coding"))
        api.store(StoreRequest(content="Python data science"))
        
        request = SearchRequest(query="Python")
        results = api.search(request)
        
        assert len(results.ranked_results) >= 2

    def test_add_to_stm(self):
        """Should add messages to STM."""
        api = MemoryAPI()
        
        result = api.add_to_stm("session_1", "Hello", STMRole.USER)
        
        assert result.success is True
        assert result.data["session_id"] == "session_1"

    def test_get_stm(self):
        """Should get STM for session."""
        api = MemoryAPI()
        
        api.add_to_stm("session_1", "Hello", STMRole.USER)
        
        stm = api.get_stm("session_1")
        
        assert stm is not None
        assert len(stm.items) == 1

    def test_clear_stm(self):
        """Should clear STM."""
        api = MemoryAPI()
        
        api.add_to_stm("session_1", "Hello", STMRole.USER)
        api.clear_stm("session_1")
        
        stm = api.get_stm("session_1")
        assert len(stm.items) == 0

    def test_get_statistics(self):
        """Should return statistics."""
        api = MemoryAPI()
        
        api.store(StoreRequest(content="Test 1"))
        api.store(StoreRequest(content="Test 2"))
        
        stats = api.get_statistics()
        
        assert stats["total_memories"] == 2

    def test_list_memories(self):
        """Should list memories with pagination."""
        api = MemoryAPI()
        
        for i in range(5):
            api.store(StoreRequest(content=f"Memory {i}"))
        
        memories = api.list_memories(limit=3)
        
        assert len(memories) == 3

    def test_register_hook(self):
        """Should register and trigger hooks."""
        api = MemoryAPI()
        
        called = []
        def hook(mem):
            called.append(mem.id)
        
        api.register_hook("post_store", hook)
        
        request = StoreRequest(content="Test")
        api.store(request)
        
        assert len(called) == 1


class TestHookRegistry:
    """Tests for HookRegistry."""

    def test_register_hook(self):
        """Should register hooks."""
        registry = HookRegistry()
        
        def callback(ctx):
            pass
        
        registry.register(HookEvent.MEMORY_CREATED, callback)
        
        assert registry.get_hook_count(HookEvent.MEMORY_CREATED) == 1

    def test_trigger_hook(self):
        """Should trigger hooks."""
        registry = HookRegistry()
        
        results = []
        def callback(ctx):
            results.append(ctx.event)
        
        registry.register(HookEvent.MEMORY_CREATED, callback)
        
        ctx = HookContext(event=HookEvent.MEMORY_CREATED)
        registry.trigger(ctx)
        
        assert HookEvent.MEMORY_CREATED in results

    def test_unregister_hook(self):
        """Should unregister hooks."""
        registry = HookRegistry()
        
        def callback(ctx):
            pass
        
        registry.register(HookEvent.MEMORY_CREATED, callback)
        registry.unregister(HookEvent.MEMORY_CREATED, callback)
        
        assert registry.get_hook_count(HookEvent.MEMORY_CREATED) == 0

    def test_global_hook(self):
        """Should trigger global hooks for all events."""
        registry = HookRegistry()
        
        results = []
        def callback(ctx):
            results.append(ctx.event)
        
        registry.register_global(callback)
        
        ctx1 = HookContext(event=HookEvent.MEMORY_CREATED)
        ctx2 = HookContext(event=HookEvent.MEMORY_ACCESSED)
        registry.trigger(ctx1)
        registry.trigger(ctx2)
        
        assert len(results) == 2

    def test_disable_hooks(self):
        """Should disable hook triggering."""
        registry = HookRegistry()
        
        results = []
        def callback(ctx):
            results.append(ctx.event)
        
        registry.register(HookEvent.MEMORY_CREATED, callback)
        registry.disable()
        
        ctx = HookContext(event=HookEvent.MEMORY_CREATED)
        registry.trigger(ctx)
        
        assert len(results) == 0

    def test_clear_hooks(self):
        """Should clear all hooks."""
        registry = HookRegistry()
        
        def callback(ctx):
            pass
        
        registry.register(HookEvent.MEMORY_CREATED, callback)
        registry.register(HookEvent.MEMORY_ACCESSED, callback)
        registry.clear()
        
        assert registry.get_hook_count() == 0


class TestMemorySystem:
    """Tests for MemorySystem."""

    @pytest.mark.asyncio
    async def test_initialize(self):
        """Should initialize successfully."""
        system = MemorySystem()
        await system.initialize()
        
        stats = system.get_statistics()
        assert stats["initialized"] is True

    @pytest.mark.asyncio
    async def test_remember_semantic(self):
        """Should store semantic memory."""
        system = MemorySystem(system_config=MemorySystemConfig(
            enable_embeddings=False,
            enable_summarization=False,
        ))
        await system.initialize()
        
        memory = await system.remember(
            "Python is great for data science",
            memory_type=MemoryType.SEMANTIC,
        )
        
        assert memory is not None
        assert memory.memory_type == MemoryType.SEMANTIC

    @pytest.mark.asyncio
    async def test_remember_episodic(self):
        """Should store episodic memory."""
        system = MemorySystem(system_config=MemorySystemConfig(
            enable_embeddings=False,
            enable_summarization=False,
        ))
        await system.initialize()
        
        memory = await system.remember(
            "Debugged async issue",
            memory_type=MemoryType.EPISODIC,
            event_type=EventType.ERROR,
        )
        
        assert memory is not None
        assert memory.memory_type == MemoryType.EPISODIC

    @pytest.mark.asyncio
    async def test_recall(self):
        """Should recall relevant memories."""
        system = MemorySystem(system_config=MemorySystemConfig(
            enable_embeddings=False,
            enable_summarization=False,
        ))
        await system.initialize()
        
        await system.remember("Python is great", tags=["python"])
        await system.remember("JavaScript is popular", tags=["javascript"])
        
        results = await system.recall("Python programming")
        
        assert len(results.ranked_results) >= 1

    @pytest.mark.asyncio
    async def test_add_message(self):
        """Should add message to STM."""
        system = MemorySystem(system_config=MemorySystemConfig(
            enable_embeddings=False,
            enable_summarization=False,
        ))
        await system.initialize()
        
        stm = await system.add_message("Hello!", role="user", session_id="test")
        
        assert len(stm.items) == 1
        assert stm.items[0].role == STMRole.USER

    @pytest.mark.asyncio
    async def test_get_context(self):
        """Should get conversation context."""
        system = MemorySystem(system_config=MemorySystemConfig(
            enable_embeddings=False,
            enable_summarization=False,
        ))
        await system.initialize()
        
        await system.add_message("Hello", role="user", session_id="test")
        await system.add_message("Hi there!", role="assistant", session_id="test")
        
        context = await system.get_context(session_id="test")
        
        assert len(context["history"]) == 2

    @pytest.mark.asyncio
    async def test_consolidate(self):
        """Should run consolidation."""
        system = MemorySystem(system_config=MemorySystemConfig(
            enable_embeddings=False,
            enable_summarization=False,
            enable_consolidation=True,
        ))
        await system.initialize()
        
        stats = await system.consolidate()
        
        assert "stm_promoted" in stats
        assert "garbage_collected" in stats

    @pytest.mark.asyncio
    async def test_forget(self):
        """Should forget memory."""
        system = MemorySystem(system_config=MemorySystemConfig(
            enable_embeddings=False,
            enable_summarization=False,
        ))
        await system.initialize()
        
        memory = await system.remember("To forget")
        
        result = await system.forget(memory.id)
        
        assert result is True

    @pytest.mark.asyncio
    async def test_get_statistics(self):
        """Should return statistics."""
        system = MemorySystem(system_config=MemorySystemConfig(
            enable_embeddings=False,
            enable_summarization=False,
        ))
        await system.initialize()
        
        await system.remember("Test 1")
        await system.remember("Test 2")
        
        stats = system.get_statistics()
        
        assert stats["total_memories"] == 2


class TestLangChainIntegration:
    """Tests for LangChain integration."""

    def test_langchain_memory_init(self):
        """Should initialize LangChain memory."""
        memory = LangChainMemory()
        
        assert memory.memory_key == "history"
        assert memory.session_id is not None

    def test_save_context(self):
        """Should save conversation context."""
        memory = LangChainMemory()
        
        memory.save_context(
            {"input": "Hello"},
            {"output": "Hi there!"},
        )
        
        messages = memory.get_messages()
        assert len(messages) == 2

    def test_load_memory_variables(self):
        """Should load memory variables."""
        memory = LangChainMemory()
        
        memory.save_context({"input": "Hello"}, {"output": "Hi!"})
        
        variables = memory.load_memory_variables({})
        
        assert "history" in variables
        assert "Hello" in variables["history"]

    def test_clear(self):
        """Should clear memory."""
        memory = LangChainMemory()
        
        memory.save_context({"input": "Hello"}, {"output": "Hi!"})
        memory.clear()
        
        messages = memory.get_messages()
        assert len(messages) == 0

    def test_add_messages(self):
        """Should add messages directly."""
        memory = LangChainMemory()
        
        memory.add_user_message("Hello")
        memory.add_ai_message("Hi!")
        
        messages = memory.get_messages()
        assert len(messages) == 2

    def test_return_messages(self):
        """Should return as message objects."""
        memory = LangChainMemory(return_messages=True)
        
        memory.save_context({"input": "Hello"}, {"output": "Hi!"})
        
        variables = memory.load_memory_variables({})
        
        assert isinstance(variables["history"], list)
        assert variables["history"][0]["role"] == "user"

    def test_hierarchical_memory(self):
        """Should initialize hierarchical memory."""
        memory = HierarchicalMemory()
        
        assert memory.context_key == "context"

    def test_memory_retriever(self):
        """Should retrieve documents."""
        api = MemoryAPI()
        api.store(StoreRequest(content="Python programming"))
        api.store(StoreRequest(content="Data science with Python"))
        
        retriever = MemoryRetriever(api, k=2)
        
        docs = retriever.get_relevant_documents("Python")
        
        assert len(docs) >= 1
        assert "page_content" in docs[0]


class TestMessage:
    """Tests for Message model."""

    def test_create_message(self):
        """Should create message."""
        msg = Message(role="human", content="Hello")
        
        assert msg.role == "human"
        assert msg.content == "Hello"
        assert msg.timestamp is not None
