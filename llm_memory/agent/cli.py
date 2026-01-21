"""
Interactive CLI for the Memory Agent.

Provides a terminal interface to chat with the agent.
"""

import asyncio
import sys
from datetime import datetime

from llm_memory.agent.memory_agent import MemoryAgent, AgentConfig


BANNER = """
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║   🧠  LLM Memory Agent                                        ║
║   ─────────────────────────────────────────────────────────   ║
║   An AI assistant with long-term memory                       ║
║   Powered by Ollama + LangGraph + LLM Memory                  ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝

Commands:
  /help      - Show this help message
  /memory    - Show memory statistics
  /remember  - Manually remember something
  /recall    - Recall memories about a topic
  /session   - Show current session info
  /new       - Start a new conversation session
  /clear     - Clear screen
  /quit      - Exit the chat

Type your message and press Enter to chat.
"""

HELP_TEXT = """
Available Commands:
───────────────────
  /help              Show this help message
  /memory            Show memory statistics  
  /remember <text>   Manually store a memory
  /recall <query>    Search memories for a topic
  /session           Show current session info
  /new [name]        Start a new session (optionally with a name)
  /clear             Clear the screen
  /quit or /exit     Exit the agent

Tips:
─────
• The agent automatically remembers important information
• Ask questions to recall past information
• Share preferences, facts, and experiences freely
• Use /memory to see what's been stored
• Use /new to start fresh without losing long-term memories
"""


def print_colored(text: str, color: str = "white") -> None:
    """Print colored text to terminal."""
    colors = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "magenta": "\033[95m",
        "cyan": "\033[96m",
        "white": "\033[97m",
        "reset": "\033[0m",
        "bold": "\033[1m",
        "dim": "\033[2m",
    }
    print(f"{colors.get(color, '')}{text}{colors['reset']}")


def print_agent_response(response: str) -> None:
    """Print the agent's response with formatting."""
    print()
    print_colored("┌─ Assistant ─────────────────────────────────────", "cyan")
    print_colored("│", "cyan")
    
    # Wrap long lines
    for line in response.split("\n"):
        if len(line) > 70:
            words = line.split()
            current_line = ""
            for word in words:
                if len(current_line) + len(word) + 1 > 70:
                    print_colored(f"│ {current_line}", "cyan")
                    current_line = word
                else:
                    current_line = f"{current_line} {word}".strip()
            if current_line:
                print_colored(f"│ {current_line}", "cyan")
        else:
            print_colored(f"│ {line}", "cyan")
    
    print_colored("│", "cyan")
    print_colored("└──────────────────────────────────────────────────", "cyan")
    print()


async def handle_command(agent: MemoryAgent, command: str) -> bool:
    """
    Handle a slash command.
    
    Returns:
        True if should continue, False if should exit
    """
    cmd = command.lower().strip()
    parts = cmd.split(maxsplit=1)
    cmd_name = parts[0]
    cmd_arg = parts[1] if len(parts) > 1 else ""
    
    if cmd_name in ["/quit", "/exit", "/q"]:
        print_colored("\n👋 Goodbye! Your memories are saved.\n", "yellow")
        return False
    
    elif cmd_name == "/help":
        print(HELP_TEXT)
    
    elif cmd_name == "/memory":
        stats = agent.get_memory_stats()
        print_colored("\n📊 Memory Statistics", "green")
        print_colored("─" * 30, "green")
        print(f"  Total memories:  {stats.get('total_memories', 0)}")
        print(f"  Active sessions: {stats.get('active_sessions', 0)}")
        print("\n  By type:")
        for mem_type, count in stats.get('by_type', {}).items():
            print(f"    • {mem_type}: {count}")
        print()
    
    elif cmd_name == "/remember":
        if not cmd_arg:
            print_colored("Usage: /remember <what to remember>", "yellow")
        else:
            mem_id = await agent.remember(cmd_arg)
            print_colored(f"✓ Remembered! (ID: {mem_id[:8]}...)", "green")
    
    elif cmd_name == "/recall":
        if not cmd_arg:
            print_colored("Usage: /recall <what to search for>", "yellow")
        else:
            memories = await agent.recall(cmd_arg)
            if memories:
                print_colored(f"\n🔍 Found {len(memories)} memories:", "green")
                for i, mem in enumerate(memories, 1):
                    content = mem["content"][:80]
                    if len(mem["content"]) > 80:
                        content += "..."
                    print(f"  {i}. [{mem['type']}] {content}")
                    print(f"     Score: {mem['score']:.2f}")
            else:
                print_colored("No memories found.", "yellow")
            print()
    
    elif cmd_name == "/clear":
        print("\033[2J\033[H")  # Clear screen
        print(BANNER)
    
    elif cmd_name == "/session":
        print_colored(f"\n📍 Current Session: {agent.config.session_id}", "green")
        stats = agent.get_memory_stats()
        print(f"   Messages in session: {stats.get('active_sessions', 0)} active sessions")
        print(f"   Total memories: {stats.get('total_memories', 0)}")
        print()
    
    elif cmd_name == "/new":
        # Generate new session ID
        from datetime import datetime
        new_session = cmd_arg if cmd_arg else f"session_{datetime.now().strftime('%H%M%S')}"
        old_session = agent.config.session_id
        agent.config.session_id = new_session
        print_colored(f"\n🆕 New session started!", "green")
        print(f"   Previous: {old_session}")
        print(f"   Current:  {new_session}")
        print_colored("   (Long-term memories are preserved, conversation history reset)", "dim")
        print()
    
    else:
        print_colored(f"Unknown command: {cmd_name}. Type /help for commands.", "yellow")
    
    return True


async def main_loop(agent: MemoryAgent) -> None:
    """Main chat loop."""
    print(BANNER)
    
    print_colored("Initializing agent...", "dim")
    await agent.initialize()
    print_colored(f"✓ Ready! Session: {agent.config.session_id}\n", "green")
    
    while True:
        try:
            # Get user input
            print_colored("You: ", "yellow", )
            user_input = input().strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.startswith("/"):
                should_continue = await handle_command(agent, user_input)
                if not should_continue:
                    break
                continue
            
            # Chat with agent
            print_colored("Thinking...", "dim")
            
            try:
                response = await agent.chat(user_input)
                # Clear "Thinking..." line
                print("\033[F\033[K", end="")
                print_agent_response(response)
            except Exception as e:
                print_colored(f"\n⚠ Error: {e}", "red")
                print_colored("The agent encountered an error. Please try again.", "yellow")
                print()
        
        except KeyboardInterrupt:
            print_colored("\n\n👋 Interrupted. Goodbye!\n", "yellow")
            break
        except EOFError:
            print_colored("\n\n👋 Goodbye!\n", "yellow")
            break


async def run_cli(
    model: str = "llama3.2",
    session_id: str = "cli_session",
) -> None:
    """
    Run the interactive CLI.
    
    Args:
        model: Ollama model to use
        session_id: Session identifier
    """
    config = AgentConfig(
        ollama_model=model,
        session_id=session_id,
        temperature=0.7,
    )
    
    agent = MemoryAgent(config)
    
    try:
        await main_loop(agent)
    finally:
        await agent.close()


def main():
    """Entry point for CLI."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="LLM Memory Agent - AI assistant with long-term memory"
    )
    parser.add_argument(
        "--model", "-m",
        default="llama3.2",
        help="Ollama model to use (default: llama3.2)"
    )
    parser.add_argument(
        "--session", "-s",
        default="cli_session",
        help="Session ID for memory isolation (default: cli_session)"
    )
    
    args = parser.parse_args()
    
    asyncio.run(run_cli(model=args.model, session_id=args.session))


if __name__ == "__main__":
    main()
