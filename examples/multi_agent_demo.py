#!/usr/bin/env python3
"""
Multi-Agent System Demo

This demonstrates the LangGraph-powered multi-agent system with Memory V3.

Features:
- Supervisor orchestration
- Specialized agents (Research, Code, Analysis, Writer)
- Memory V3 with Knowledge Graph
- Multi-hop reasoning
"""

import logging
from llm_memory.agents import (
    create_graph,
    run_graph,
    run_graph_stream,
    get_memory_stats,
    AgentConfig,
    ModelConfig,
)

# Set up logging
logging.basicConfig(level=logging.INFO)


def demo_basic_task():
    """Demo: Basic task execution."""
    print("\n" + "=" * 70)
    print("DEMO 1: Basic Task Execution")
    print("=" * 70)
    
    # Create graph with default config
    graph = create_graph(user_id="demo_user")
    
    # Simple research task
    task = "What are the benefits of meditation for mental health?"
    
    print(f"\n📝 Task: {task}")
    print("\n🔄 Running agents...")
    
    result = run_graph(graph, task=task)
    
    print("\n✅ Final Response:")
    print("-" * 60)
    print(result.get("final_response", "No response"))
    
    # Show memory stats
    stats = get_memory_stats()
    print("\n📊 Memory Stats:", stats)


def demo_streaming():
    """Demo: Streaming execution."""
    print("\n" + "=" * 70)
    print("DEMO 2: Streaming Execution")
    print("=" * 70)
    
    graph = create_graph(user_id="demo_user")
    
    task = "Write a Python function to calculate fibonacci numbers"
    
    print(f"\n📝 Task: {task}")
    print("\n🔄 Streaming agent execution...")
    
    for node_name, state_update in run_graph_stream(graph, task=task):
        if "messages" in state_update:
            msg = state_update["messages"]
            if hasattr(msg, 'content') and msg.content:
                preview = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
                print(f"\n  [{node_name}] {preview}")


def demo_custom_model():
    """Demo: Using a different model."""
    print("\n" + "=" * 70)
    print("DEMO 3: Custom Model (qwen2.5:7b)")
    print("=" * 70)
    
    # Configure for Qwen model
    config = AgentConfig(
        default_model=ModelConfig(
            provider="ollama",
            model_name="qwen2.5:7b",
            temperature=0.7,
        ),
        enable_memory=True,
    )
    
    graph = create_graph(config=config, user_id="demo_user")
    
    task = "Compare Python vs JavaScript for web development"
    
    print(f"\n📝 Task: {task}")
    print("\n🔄 Running with qwen2.5:7b...")
    
    result = run_graph(graph, task=task)
    
    print("\n✅ Final Response:")
    print("-" * 60)
    print(result.get("final_response", "No response")[:1000])


def demo_memory_persistence():
    """Demo: Memory persists across tasks."""
    print("\n" + "=" * 70)
    print("DEMO 4: Memory Persistence")
    print("=" * 70)
    
    graph = create_graph(user_id="memory_demo")
    
    # First task: Tell it something
    task1 = "Remember that my favorite programming language is Python and I work as a data scientist"
    
    print(f"\n📝 Task 1: {task1}")
    result1 = run_graph(graph, task=task1, thread_id="thread1")
    print(f"Response: {result1.get('final_response', '')[:200]}")
    
    # Second task: Ask about it
    task2 = "What is my favorite programming language and what do I do for work?"
    
    print(f"\n📝 Task 2: {task2}")
    result2 = run_graph(graph, task=task2, thread_id="thread2")
    print(f"\n✅ Response: {result2.get('final_response', '')}")
    
    # Show memory stats
    stats = get_memory_stats()
    print(f"\n📊 Memory now has: {stats.get('total_memories', 0)} items")


def main():
    """Run all demos."""
    print("""
╔═══════════════════════════════════════════════════════════════════╗
║         🤖 Multi-Agent System Demo                                 ║
║                                                                   ║
║   This demonstrates the LangGraph multi-agent system               ║
║   with Memory V3 (Knowledge Graph + Multi-hop)                     ║
╚═══════════════════════════════════════════════════════════════════╝
    """)
    
    try:
        # Run demos
        demo_basic_task()
        demo_memory_persistence()
        
        print("\n" + "=" * 70)
        print("✅ All demos completed!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
