"""
Basic usage example for LLM Memory.

This example demonstrates:
1. Creating different memory types
2. Memory decay behavior
3. Basic storage operations
"""

import asyncio
from datetime import datetime, timedelta
from pathlib import Path

# Add parent to path for development
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_memory.config import MemoryConfig, StorageConfig
from llm_memory.models.short_term import ShortTermMemory, WorkingContext, STMRole
from llm_memory.models.episodic import EpisodicMemory, Episode, EventType
from llm_memory.models.semantic import SemanticMemory, Fact, FactType
from llm_memory.storage.sqlite import SQLiteStorage


async def demo_short_term_memory():
    """Demonstrate short-term memory usage."""
    print("\n" + "="*60)
    print("SHORT-TERM MEMORY DEMO")
    print("="*60)
    
    # Create a short-term memory buffer
    stm = ShortTermMemory(
        content="Conversation buffer",
        session_id="session_001",
    )
    
    # Add conversation messages
    stm.add_message("How do I implement authentication?", role=STMRole.USER)
    stm.add_message(
        "I recommend using JWT tokens with OAuth2...",
        role=STMRole.ASSISTANT
    )
    stm.add_message("Can you show me an example?", role=STMRole.USER)
    
    print(f"\nSTM Buffer: {len(stm)} items")
    print(f"Memory Type: {stm.memory_type.value}")
    print(f"Decay Rate: {stm.get_decay_rate()}")
    
    # Show messages
    print("\nConversation:")
    for item in stm.items:
        print(f"  [{item.role.value}]: {item.content[:50]}...")
    
    # Demonstrate decay
    print("\n--- Memory Decay ---")
    ctx = stm.items[0]
    print(f"Initial strength: {ctx.current_strength:.2f}")
    
    # Simulate time passing
    ctx.last_accessed_at = datetime.utcnow() - timedelta(hours=2)
    ctx.update_strength()
    print(f"After 2 hours: {ctx.current_strength:.2f}")
    
    # Rehearsal boost
    ctx.apply_rehearsal_boost(boost=0.3)
    print(f"After rehearsal: {ctx.current_strength:.2f}")


async def demo_episodic_memory():
    """Demonstrate episodic memory usage."""
    print("\n" + "="*60)
    print("EPISODIC MEMORY DEMO")
    print("="*60)
    
    # Create episodic memory from a debugging session
    ep = EpisodicMemory(content="Debugging authentication issue")
    
    # Add episodes
    ep.add_episode(Episode(
        event_id="ep_001",
        event_type=EventType.ERROR,
        description="User reported 401 errors on API calls",
        details={"error_code": 401, "endpoint": "/api/users"},
        outcome="Investigated token expiration",
        was_successful=False,
    ))
    
    ep.add_episode(Episode(
        event_id="ep_002",
        event_type=EventType.DISCOVERY,
        description="Found that refresh token logic was missing",
        details={"file": "auth.py", "line": 45},
        emotional_valence=0.6,  # Positive - found the issue!
    ))
    
    ep.add_episode(Episode(
        event_id="ep_003",
        event_type=EventType.TASK_COMPLETION,
        description="Implemented token refresh, tests passing",
        outcome="Fixed authentication",
        was_successful=True,
    ))
    
    # Add lessons learned
    ep.lessons_learned = [
        "Always implement token refresh logic",
        "Add tests for token expiration scenarios"
    ]
    
    print(f"\nEpisodic Memory: {len(ep)} episodes")
    print(f"Memory Type: {ep.memory_type.value}")
    print(f"Decay Rate: {ep.get_decay_rate()}")
    
    print("\nEpisodes:")
    for episode in ep.episodes:
        status = "✓" if episode.was_successful else "✗"
        print(f"  [{status}] {episode.event_type.value}: {episode.description[:50]}...")
    
    print("\nLessons Learned:")
    for lesson in ep.lessons_learned:
        print(f"  • {lesson}")


async def demo_semantic_memory():
    """Demonstrate semantic memory usage."""
    print("\n" + "="*60)
    print("SEMANTIC MEMORY DEMO")
    print("="*60)
    
    # Create semantic memory for user preferences
    sem = SemanticMemory(content="User preferences and knowledge")
    
    # Add facts
    sem.add_fact(Fact(
        fact_id="pref_001",
        fact_type=FactType.PREFERENCE,
        subject="User",
        predicate="prefers",
        object="Python for backend development",
        statement="User prefers Python for backend development",
        confidence=0.9,
    ))
    
    sem.add_fact(Fact(
        fact_id="pref_002",
        fact_type=FactType.PREFERENCE,
        subject="User",
        predicate="prefers",
        object="TypeScript for frontend",
        statement="User prefers TypeScript for frontend development",
        confidence=0.85,
    ))
    
    sem.add_fact(Fact(
        fact_id="cap_001",
        fact_type=FactType.CAPABILITY,
        subject="User",
        predicate="knows",
        object="PostgreSQL and Redis",
        statement="User knows PostgreSQL and Redis for data storage",
        confidence=0.8,
    ))
    
    sem.add_fact(Fact(
        fact_id="proc_001",
        fact_type=FactType.PROCEDURE,
        subject="Authentication",
        predicate="should use",
        object="JWT with refresh tokens",
        statement="Authentication should use JWT with refresh tokens",
        confidence=0.95,
    ))
    
    print(f"\nSemantic Memory: {len(sem.facts)} facts")
    print(f"Memory Type: {sem.memory_type.value}")
    print(f"Decay Rate: {sem.get_decay_rate()}")
    print(f"Overall Confidence: {sem.overall_confidence:.2f}")
    
    print("\nFacts by Type:")
    for fact_type in [FactType.PREFERENCE, FactType.CAPABILITY, FactType.PROCEDURE]:
        facts = sem.get_facts_by_type(fact_type)
        print(f"\n  {fact_type.value.upper()}:")
        for fact in facts:
            print(f"    • {fact.statement} (confidence: {fact.confidence:.0%})")
    
    # Strengthen a fact (simulating repeated observation)
    print("\n--- Strengthening Facts ---")
    pref = sem.get_fact("pref_001")
    print(f"Before: confidence={pref.confidence:.2f}, evidence={pref.evidence_count}")
    
    sem.strengthen_fact("pref_001", boost=0.05)
    
    pref = sem.get_fact("pref_001")
    print(f"After:  confidence={pref.confidence:.2f}, evidence={pref.evidence_count}")


async def demo_storage():
    """Demonstrate storage operations."""
    print("\n" + "="*60)
    print("STORAGE DEMO")
    print("="*60)
    
    # Configure storage with temp directory
    config = StorageConfig(
        sqlite_path=Path("./data/demo_memory.db")
    )
    
    async with SQLiteStorage(config) as storage:
        print(f"\nConnected to: {config.sqlite_path}")
        
        # Create different memory types
        stm = ShortTermMemory(content="Demo buffer")
        stm.add_message("Hello!", role=STMRole.USER)
        stm.metadata.user_id = "demo_user"
        stm.metadata.scope = "demo_project"
        
        ep = EpisodicMemory(content="Demo episode")
        ep.add_episode(Episode(
            event_id="demo_ep",
            event_type=EventType.INTERACTION,
            description="Demo interaction",
        ))
        ep.metadata.user_id = "demo_user"
        
        sem = SemanticMemory(content="Demo knowledge")
        sem.add_fact(Fact(
            fact_id="demo_fact",
            fact_type=FactType.DEFINITION,
            subject="Demo",
            predicate="is",
            object="example",
            statement="Demo is an example",
        ))
        sem.metadata.user_id = "demo_user"
        
        # Store all memories
        await storage.create(stm)
        await storage.create(ep)
        await storage.create(sem)
        
        print("\nCreated 3 memories")
        
        # Get stats
        stats = await storage.get_stats()
        print(f"\nStorage Statistics:")
        print(f"  Total memories: {stats['total_memories']}")
        for mem_type, count in stats['by_type'].items():
            print(f"  {mem_type}: {count}")
        
        # Query by user
        user_memories = await storage.query_by_user("demo_user")
        print(f"\nMemories for demo_user: {len(user_memories)}")
        
        # Read back a memory
        loaded = await storage.read(sem.id)
        print(f"\nLoaded semantic memory:")
        print(f"  Facts: {len(loaded.facts)}")
        print(f"  Statement: {loaded.facts[0].statement}")
    
    print("\n✓ Storage demo complete!")


async def main():
    """Run all demos."""
    print("\n" + "="*60)
    print("LLM MEMORY - BASIC USAGE EXAMPLES")
    print("="*60)
    
    await demo_short_term_memory()
    await demo_episodic_memory()
    await demo_semantic_memory()
    await demo_storage()
    
    print("\n" + "="*60)
    print("ALL DEMOS COMPLETE!")
    print("="*60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
