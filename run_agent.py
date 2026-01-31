#!/usr/bin/env python3
"""
Run the Memory V4 Agent interactively.
"""

import sys
import os
from pathlib import Path

# Add package to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_memory.agents_v4.graph import MemoryAgent

def main():
    print("Initializing Memory Agent...")
    agent = MemoryAgent(
        model_name="qwen2.5:32b",
        memory_path="./agent_memory_v4"
    )
    
    print("\n🧠 Memory Agent V4 Ready!")
    print("Type 'exit' or 'quit' to stop.")
    print("-" * 50)
    
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            if user_input.lower() in ['exit', 'quit']:
                break
            
            if not user_input:
                continue
                
            agent.chat(user_input)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()
