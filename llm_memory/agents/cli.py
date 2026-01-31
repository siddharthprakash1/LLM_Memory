#!/usr/bin/env python3
"""
CLI Interface for Multi-Agent System.

Run with: python -m llm_memory.agents.cli

or if installed: memory-agents
"""

import asyncio
import sys
import logging
from datetime import datetime

from .config import DEFAULT_CONFIG, QWEN_CONFIG, GEMMA_CONFIG, AgentConfig, ModelConfig
from .graph import create_graph, run_graph, run_graph_stream, get_memory_stats


BANNER = """
╔═══════════════════════════════════════════════════════════════════╗
║         🤖 Multi-Agent System with Memory V3                       ║
║                                                                   ║
║   Agents: Supervisor | Research | Code | Analysis | Writer         ║
║   Memory: Knowledge Graph + Multi-hop Reasoning                    ║
╚═══════════════════════════════════════════════════════════════════╝
"""

HELP_TEXT = """
Commands:
  /help         Show this help message
  /stats        Show memory statistics
  /stream       Toggle streaming mode
  /model <name> Change model (llama3.2, qwen2.5:7b, gemma3:27b)
  /clear        Clear memory
  /quit         Exit the application

Just type your task and press Enter to have the agents work on it!
"""


def create_config_for_model(model_name: str) -> AgentConfig:
    """Create config for a specific model."""
    return AgentConfig(
        default_model=ModelConfig(
            provider="ollama",
            model_name=model_name,
            temperature=0.7,
        ),
        enable_memory=True,
    )


def main():
    """Main CLI entry point."""
    logging.basicConfig(
        level=logging.WARNING,  # Only show warnings and errors
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print(BANNER)
    print(f"Initializing agents with Ollama (llama3.2)...")
    
    # Default config
    config = DEFAULT_CONFIG
    model_name = "llama3.2"
    streaming = False
    user_id = f"user_{datetime.now().strftime('%Y%m%d')}"
    
    # Create graph
    try:
        graph = create_graph(config=config, user_id=user_id)
        print("✓ Agents initialized successfully!")
        print(HELP_TEXT)
    except Exception as e:
        print(f"❌ Failed to initialize agents: {e}")
        sys.exit(1)
    
    thread_id = "cli_session"
    
    while True:
        try:
            # Get user input
            user_input = input("\n🧑 You: ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.startswith("/"):
                cmd = user_input.lower()
                
                if cmd == "/quit" or cmd == "/exit":
                    print("\n👋 Goodbye!")
                    break
                
                elif cmd == "/help":
                    print(HELP_TEXT)
                    continue
                
                elif cmd == "/stats":
                    stats = get_memory_stats()
                    print("\n📊 Memory Statistics:")
                    for key, value in stats.items():
                        print(f"   {key}: {value}")
                    continue
                
                elif cmd == "/stream":
                    streaming = not streaming
                    print(f"📺 Streaming mode: {'ON' if streaming else 'OFF'}")
                    continue
                
                elif cmd.startswith("/model "):
                    new_model = cmd.split(" ", 1)[1].strip()
                    print(f"🔄 Switching to model: {new_model}")
                    config = create_config_for_model(new_model)
                    model_name = new_model
                    graph = create_graph(config=config, user_id=user_id)
                    print("✓ Model switched!")
                    continue
                
                elif cmd == "/clear":
                    from .graph import _graph_memory
                    if _graph_memory:
                        _graph_memory.clear()
                    print("🗑️  Memory cleared!")
                    continue
                
                else:
                    print(f"❓ Unknown command: {cmd}")
                    print("   Type /help for available commands")
                    continue
            
            # Process task with agents
            print(f"\n🤖 Processing with {model_name}...")
            print("-" * 60)
            
            if streaming:
                # Streaming mode
                for node_name, state_update in run_graph_stream(
                    graph,
                    task=user_input,
                    thread_id=thread_id,
                ):
                    if "messages" in state_update:
                        msg = state_update["messages"]
                        if hasattr(msg, 'content'):
                            content = msg.content
                            if len(content) > 100:
                                print(f"\n📝 [{node_name}]: {content[:200]}...")
                            else:
                                print(f"\n📝 [{node_name}]: {content}")
                
                # Get final response
                from .graph import get_graph_memory
                final = "Task completed. Check the output above."
            else:
                # Regular mode
                result = run_graph(
                    graph,
                    task=user_input,
                    thread_id=thread_id,
                )
                final = result.get("final_response", "No response generated")
            
            print("-" * 60)
            print(f"\n🤖 Response:\n{final}")
            
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            logging.exception("CLI Error")


if __name__ == "__main__":
    main()
