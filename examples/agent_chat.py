"""
Example: Chat with the Memory Agent.

This example demonstrates using the Memory Agent programmatically.
"""

import asyncio
from llm_memory.agent import MemoryAgent, AgentConfig


async def main():
    """Run an example conversation with the agent."""
    print("=" * 60)
    print("🧠 Memory Agent Example")
    print("=" * 60)
    
    # Configure the agent
    config = AgentConfig(
        ollama_model="llama3.2",  # Change to your preferred model
        temperature=0.7,
        session_id="example_session",
    )
    
    # Create and initialize the agent
    agent = MemoryAgent(config)
    await agent.initialize()
    
    # Example conversation
    conversations = [
        "Hi! My name is Alex and I'm a Python developer.",
        "I prefer using VS Code as my IDE and I love dark themes.",
        "What's my name?",
        "What editor do I use?",
        "Remember that my favorite food is sushi.",
        "What do you know about me?",
    ]
    
    print("\n" + "-" * 40)
    print("Starting conversation...")
    print("-" * 40 + "\n")
    
    for user_msg in conversations:
        print(f"👤 You: {user_msg}")
        
        try:
            response = await agent.chat(user_msg)
            print(f"🤖 Agent: {response}\n")
        except Exception as e:
            print(f"❌ Error: {e}\n")
        
        # Small delay between messages
        await asyncio.sleep(0.5)
    
    # Show memory stats
    print("-" * 40)
    print("📊 Memory Statistics:")
    print("-" * 40)
    stats = agent.get_memory_stats()
    print(f"  Total memories: {stats.get('total_memories', 0)}")
    print(f"  Active sessions: {stats.get('active_sessions', 0)}")
    for mem_type, count in stats.get('by_type', {}).items():
        print(f"    - {mem_type}: {count}")
    
    # Recall specific memories
    print("\n" + "-" * 40)
    print("🔍 Recalling memories about 'preferences':")
    print("-" * 40)
    memories = await agent.recall("preferences", limit=3)
    for mem in memories:
        print(f"  [{mem['type']}] {mem['content'][:60]}...")
    
    # Clean up
    await agent.close()
    
    print("\n" + "=" * 60)
    print("Example completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
