"""
Advanced usage example for LLM Memory.

Demonstrates:
- Conflict resolution between memories
- LangChain integration
- Event hooks
- Memory consolidation flow
"""

import asyncio
from llm_memory import (
    MemorySystem,
    MemorySystemConfig,
    MemoryType,
    MemorySource,
    EventType,
    FactType,
)
from llm_memory.api.hooks import HookEvent, HookContext, get_hook_registry
from llm_memory.api.integrations.langchain import LangChainMemory, MemoryRetriever


async def main():
    """Advanced example demonstrating all features."""
    print("=" * 60)
    print("LLM Memory - Advanced Usage Example")
    print("=" * 60)
    
    # =========================================
    # 1. Initialize with Event Hooks
    # =========================================
    print("\n" + "-" * 40)
    print("1. Setting Up Event Hooks")
    print("-" * 40)
    
    # Get global hook registry
    hooks = get_hook_registry()
    
    # Register hooks for monitoring
    created_memories = []
    
    def on_memory_created(context: HookContext):
        created_memories.append(context.memory_id)
        print(f"  [HOOK] Memory created: {context.memory_id[:8]}...")
    
    hooks.register(HookEvent.MEMORY_CREATED, on_memory_created)
    print("  ✓ Registered MEMORY_CREATED hook")
    
    # Initialize system
    config = MemorySystemConfig(
        enable_embeddings=False,
        enable_summarization=False,
        enable_conflict_resolution=True,
    )
    
    system = MemorySystem(system_config=config)
    await system.initialize()
    print("  ✓ Memory system initialized")
    
    # =========================================
    # 2. Demonstrate Conflict Detection
    # =========================================
    print("\n" + "-" * 40)
    print("2. Conflict Detection & Resolution")
    print("-" * 40)
    
    # Store initial preference
    pref1 = await system.remember(
        "User's favorite programming language is Python",
        memory_type=MemoryType.SEMANTIC,
        tags=["preference"],
    )
    print(f"  Stored: {pref1.content}")
    
    # Store conflicting preference (system will detect)
    pref2 = await system.remember(
        "User's favorite programming language is Rust",
        memory_type=MemoryType.SEMANTIC,
        tags=["preference"],
    )
    print(f"  Stored (conflicting): {pref2.content}")
    
    print(f"\n  Total memories after conflict: {system.get_statistics()['total_memories']}")
    print(f"  (Both kept - conflicts can be reviewed manually)")
    
    # =========================================
    # 3. LangChain Integration
    # =========================================
    print("\n" + "-" * 40)
    print("3. LangChain Integration")
    print("-" * 40)
    
    # Create LangChain-compatible memory
    langchain_memory = LangChainMemory(session_id="langchain_session")
    
    # Simulate a conversation using LangChain-style context saving
    langchain_memory.save_context(
        {"input": "What's the best way to handle async in Python?"},
        {"output": "Use asyncio with async/await syntax for best results."},
    )
    
    langchain_memory.save_context(
        {"input": "How about error handling?"},
        {"output": "Use try/except blocks with specific exception types."},
    )
    
    # Load memory variables (as LangChain would)
    variables = langchain_memory.load_memory_variables({})
    print(f"  Loaded history:\n{variables['history'][:100]}...")
    
    # =========================================
    # 4. Memory Retriever for RAG
    # =========================================
    print("\n" + "-" * 40)
    print("4. Memory Retriever (RAG)")
    print("-" * 40)
    
    # Add some knowledge to the system
    from llm_memory.api import MemoryAPI, StoreRequest
    
    api = MemoryAPI()
    api.store(StoreRequest(content="Python asyncio uses event loops"))
    api.store(StoreRequest(content="Rust has zero-cost abstractions"))
    api.store(StoreRequest(content="TypeScript adds types to JavaScript"))
    
    # Create retriever
    retriever = MemoryRetriever(api, k=3)
    
    # Retrieve relevant documents
    docs = retriever.get_relevant_documents("async programming")
    print(f"  Found {len(docs)} relevant documents for 'async programming'")
    for i, doc in enumerate(docs[:2], 1):
        print(f"    {i}. {doc['page_content'][:50]}...")
    
    # =========================================
    # 5. Multi-Session Conversations
    # =========================================
    print("\n" + "-" * 40)
    print("5. Multi-Session Conversations")
    print("-" * 40)
    
    # Session 1
    await system.add_message("Hello!", role="user", session_id="session_1")
    await system.add_message("Hi! How can I help?", role="assistant", session_id="session_1")
    
    # Session 2 (parallel conversation)
    await system.add_message("Debug my code", role="user", session_id="session_2")
    await system.add_message("Sure, show me the error", role="assistant", session_id="session_2")
    
    # Get contexts for both sessions
    ctx1 = await system.get_context("session_1")
    ctx2 = await system.get_context("session_2")
    
    print(f"  Session 1: {len(ctx1['history'])} messages")
    print(f"  Session 2: {len(ctx2['history'])} messages")
    
    # =========================================
    # 6. Memory Statistics Dashboard
    # =========================================
    print("\n" + "-" * 40)
    print("6. Memory Statistics Dashboard")
    print("-" * 40)
    
    stats = system.get_statistics()
    print(f"""
  ┌─────────────────────────────────────┐
  │       Memory System Stats           │
  ├─────────────────────────────────────┤
  │ Total Memories:    {stats['total_memories']:15} │
  │ Active Sessions:   {stats['active_sessions']:15} │
  │ Pending Conflicts: {stats['pending_conflicts']:15} │
  ├─────────────────────────────────────┤
  │ By Type:                            │
  │   Short-Term:      {stats['by_type']['short_term']:15} │
  │   Episodic:        {stats['by_type']['episodic']:15} │
  │   Semantic:        {stats['by_type']['semantic']:15} │
  └─────────────────────────────────────┘
""")
    
    # =========================================
    # 7. Demonstrate Forgetting
    # =========================================
    print("-" * 40)
    print("7. Memory Forgetting")
    print("-" * 40)
    
    # Store a temporary memory
    temp_mem = await system.remember(
        "Temporary note: delete this later",
        memory_type=MemoryType.SEMANTIC,
    )
    print(f"  Created: {temp_mem.id[:8]}...")
    
    # Soft delete (deactivate)
    await system.forget(temp_mem.id)
    print(f"  Soft deleted (deactivated)")
    
    # Check it's not in active queries anymore
    # (it still exists in storage for potential recovery)
    
    # =========================================
    # Cleanup
    # =========================================
    await system.close()
    hooks.clear()
    
    print("\n" + "=" * 60)
    print(f"Hooks captured {len(created_memories)} memory creation events")
    print("Advanced example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
