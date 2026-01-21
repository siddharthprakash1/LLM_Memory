"""
Basic usage example for LLM Memory.

Demonstrates:
- Initializing the memory system
- Storing different types of memories
- Querying with intent-aware retrieval
- Managing conversation context
- Working with the consolidation pipeline
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


async def main():
    """Main example function."""
    print("=" * 60)
    print("LLM Memory - Basic Usage Example")
    print("=" * 60)
    
    # Initialize the memory system
    config = MemorySystemConfig(
        enable_embeddings=False,  # Disable for this example
        enable_summarization=False,
        enable_consolidation=True,
        enable_conflict_resolution=True,
    )
    
    system = MemorySystem(system_config=config)
    await system.initialize()
    
    print("\n✓ Memory system initialized")
    
    # =========================================
    # 1. Store Semantic Memories (Facts/Knowledge)
    # =========================================
    print("\n" + "-" * 40)
    print("1. Storing Semantic Memories (Facts)")
    print("-" * 40)
    
    # Store user preferences
    pref1 = await system.remember(
        "User prefers Python for data science projects",
        memory_type=MemoryType.SEMANTIC,
        tags=["preference", "language"],
        fact_type=FactType.PREFERENCE,
    )
    print(f"  Stored: {pref1.content[:50]}...")
    
    pref2 = await system.remember(
        "User likes dark mode in their IDE",
        memory_type=MemoryType.SEMANTIC,
        tags=["preference", "ui"],
        fact_type=FactType.PREFERENCE,
    )
    print(f"  Stored: {pref2.content[:50]}...")
    
    # Store factual knowledge
    fact1 = await system.remember(
        "The project uses FastAPI for the backend API",
        memory_type=MemoryType.SEMANTIC,
        tags=["project", "tech-stack"],
        fact_type=FactType.DEFINITION,
    )
    print(f"  Stored: {fact1.content[:50]}...")
    
    # =========================================
    # 2. Store Episodic Memories (Experiences)
    # =========================================
    print("\n" + "-" * 40)
    print("2. Storing Episodic Memories (Experiences)")
    print("-" * 40)
    
    # Store a problem-solving session
    episode1 = await system.remember(
        "User encountered a bug with async database connections. "
        "We debugged it together and found the issue was missing await. "
        "Solution: Added await to the database query call.",
        memory_type=MemoryType.EPISODIC,
        tags=["debugging", "async", "database"],
        event_type=EventType.ERROR,
    )
    print(f"  Stored: {episode1.content[:50]}...")
    
    # Store a decision
    episode2 = await system.remember(
        "User decided to use SQLAlchemy instead of raw SQL "
        "for better maintainability and type safety.",
        memory_type=MemoryType.EPISODIC,
        tags=["decision", "database"],
        event_type=EventType.DECISION,
    )
    print(f"  Stored: {episode2.content[:50]}...")
    
    # =========================================
    # 3. Conversation Memory (Short-Term)
    # =========================================
    print("\n" + "-" * 40)
    print("3. Conversation Memory (Short-Term)")
    print("-" * 40)
    
    session_id = "example_session"
    
    # Simulate a conversation
    await system.add_message(
        "Can you help me with my Python project?",
        role="user",
        session_id=session_id,
    )
    print("  User: Can you help me with my Python project?")
    
    await system.add_message(
        "Of course! I'd be happy to help. What do you need?",
        role="assistant",
        session_id=session_id,
    )
    print("  Assistant: Of course! I'd be happy to help...")
    
    await system.add_message(
        "I need to add authentication to my FastAPI app",
        role="user",
        session_id=session_id,
    )
    print("  User: I need to add authentication...")
    
    # Get conversation context
    context = await system.get_context(
        session_id=session_id,
        include_relevant=True,
        query="authentication FastAPI",
    )
    print(f"\n  Conversation has {len(context['history'])} messages")
    print(f"  Found {len(context['relevant_memories'])} relevant memories")
    
    # =========================================
    # 4. Intent-Aware Retrieval
    # =========================================
    print("\n" + "-" * 40)
    print("4. Intent-Aware Retrieval")
    print("-" * 40)
    
    # Factual query
    print("\n  Query: 'What language does the user prefer?'")
    results = await system.recall("What language does the user prefer?")
    print(f"  Intent: {results.intent.primary_intent.value}")
    print(f"  Found: {len(results.ranked_results)} results")
    if results.ranked_results:
        print(f"  Top result: {results.ranked_results[0].result.content[:60]}...")
    
    # Episodic query
    print("\n  Query: 'What happened with the database bug?'")
    results = await system.recall("What happened with the database bug?")
    print(f"  Intent: {results.intent.primary_intent.value}")
    print(f"  Found: {len(results.ranked_results)} results")
    if results.ranked_results:
        print(f"  Top result: {results.ranked_results[0].result.content[:60]}...")
    
    # Procedural query
    print("\n  Query: 'How do I set up the project?'")
    results = await system.recall("How do I set up the project?")
    print(f"  Intent: {results.intent.primary_intent.value}")
    
    # =========================================
    # 5. System Statistics
    # =========================================
    print("\n" + "-" * 40)
    print("5. System Statistics")
    print("-" * 40)
    
    stats = system.get_statistics()
    print(f"  Total memories: {stats['total_memories']}")
    print(f"  Active sessions: {stats['active_sessions']}")
    print(f"  By type:")
    for mem_type, count in stats['by_type'].items():
        print(f"    - {mem_type}: {count}")
    
    # =========================================
    # 6. Consolidation
    # =========================================
    print("\n" + "-" * 40)
    print("6. Memory Consolidation")
    print("-" * 40)
    
    consolidation_stats = await system.consolidate()
    print(f"  STM promoted: {consolidation_stats['stm_promoted']}")
    print(f"  Garbage collected: {consolidation_stats['garbage_collected']}")
    
    # =========================================
    # Cleanup
    # =========================================
    await system.close()
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
