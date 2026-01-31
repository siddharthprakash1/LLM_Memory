#!/usr/bin/env python3
"""
Web UI for Multi-Agent System with Memory V3.

Features:
- Chat interface with agent team
- Real-time agent activity visualization
- Memory browser with Knowledge Graph
- Dashboard with statistics

Usage:
    python -m llm_memory.agents.web_ui
    python -m llm_memory.agents.web_ui --share
    python -m llm_memory.agents.web_ui --port 8080
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
import json

# Add package to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import gradio as gr
except ImportError:
    print("❌ Gradio not installed. Install with: pip install gradio")
    sys.exit(1)

from llm_memory.agents.config import AgentConfig, ModelConfig, DEFAULT_CONFIG
from llm_memory.agents.graph import (
    create_graph,
    run_graph_stream,
    get_graph_memory,
    get_memory_stats,
)
from llm_memory.agents.state import Message


# Global state
_graph = None
_config = None
_user_id = "ui_user"


# CSS for modern styling
CUSTOM_CSS = """
.gradio-container {
    max-width: 1400px !important;
}

.agent-activity {
    font-family: 'Monaco', 'Menlo', monospace;
    font-size: 12px;
    background: #1a1a2e;
    color: #00ff88;
    padding: 10px;
    border-radius: 8px;
    max-height: 300px;
    overflow-y: auto;
}

.memory-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 15px;
    border-radius: 12px;
    margin: 8px 0;
}

.kg-fact {
    background: #f0f4f8;
    padding: 8px 12px;
    border-radius: 6px;
    margin: 4px 0;
    border-left: 3px solid #4a90d9;
}

.stat-card {
    background: white;
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    text-align: center;
}

.stat-value {
    font-size: 2em;
    font-weight: bold;
    color: #4a90d9;
}

.stat-label {
    color: #666;
    font-size: 0.9em;
}

footer { display: none !important; }
"""


def check_ollama():
    """Check if Ollama is running."""
    try:
        import httpx
        response = httpx.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False


def get_available_models():
    """Get available Ollama models."""
    try:
        import httpx
        response = httpx.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            return [m["name"] for m in models]
    except:
        pass
    return ["llama3.2:latest", "qwen2.5:3b", "gemma3:27b"]


def initialize_agents(model_name: str, user_id: str):
    """Initialize the agent graph."""
    global _graph, _config, _user_id
    
    _user_id = user_id
    _config = AgentConfig(
        default_model=ModelConfig(
            provider="ollama",
            model_name=model_name,
            temperature=0.7,
        ),
        enable_memory=True,
        memory_persist_path="./ui_agent_memory",
    )
    
    _graph = create_graph(config=_config, user_id=user_id)
    return f"✅ Agents initialized with {model_name}"


def process_message(message: str, history: list, model: str, max_iter: int):
    """Process a message through the multi-agent system."""
    global _graph
    
    if not message.strip():
        return history, "", "No message"
    
    # Check Ollama
    if not check_ollama():
        history.append((message, "❌ Ollama is not running! Start it with: `ollama serve`"))
        return history, "", "Ollama not running"
    
    # Initialize if needed
    if _graph is None:
        initialize_agents(model, _user_id)
    
    # Collect agent activity
    agent_log = []
    final_response = None
    
    try:
        session_id = f"ui-{datetime.now().strftime('%H%M%S')}"
        
        for node_name, update in run_graph_stream(
            _graph,
            message,
            thread_id=session_id,
            max_iterations=max_iter
        ):
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            if node_name == "supervisor":
                msgs = update.get("messages")
                if isinstance(msgs, Message):
                    routing = msgs.content.replace("[Supervisor]", "").strip()[:100]
                    agent_log.append(f"[{timestamp}] 📋 SUPERVISOR: {routing}")
            
            elif node_name in ["research", "code", "analysis", "writer"]:
                icon = {"research": "🔍", "code": "💻", "analysis": "📊", "writer": "✍️"}[node_name]
                agent_log.append(f"[{timestamp}] {icon} {node_name.upper()} working...")
            
            elif node_name == "compile_response":
                final_response = update.get("final_response", "")
                agent_log.append(f"[{timestamp}] ✅ Response compiled")
        
        # Format response
        if final_response:
            history.append((message, final_response))
        else:
            history.append((message, "⚠️ No response generated. Try again or increase iterations."))
    
    except Exception as e:
        history.append((message, f"❌ Error: {str(e)}"))
        agent_log.append(f"[ERROR] {str(e)}")
    
    # Format agent log as string
    log_text = "\n".join(agent_log) if agent_log else "No activity recorded"
    
    return history, "", log_text


def get_memories_display():
    """Get formatted memory display."""
    memory = get_graph_memory()
    if not memory:
        return "No memories yet. Start chatting!"
    
    items = list(memory.memories.values())[:20]
    
    if not items:
        return "No memories stored yet."
    
    html = ""
    for item in items:
        date_str = item.event_date or "No date"
        speaker_str = item.speaker or "Unknown"
        type_badge = {
            "fact": "🧠 Fact",
            "event": "📅 Event", 
            "conversation": "💬 Conversation",
            "preference": "❤️ Preference",
        }.get(item.memory_type, "📝 Memory")
        
        html += f"""
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            color: white; padding: 15px; border-radius: 12px; margin: 8px 0;">
    <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
        <span style="background: rgba(255,255,255,0.2); padding: 2px 8px; border-radius: 4px; font-size: 0.8em;">{type_badge}</span>
        <span style="opacity: 0.8; font-size: 0.8em;">{date_str}</span>
    </div>
    <div style="font-size: 0.95em; line-height: 1.4;">{item.content[:200]}{'...' if len(item.content) > 200 else ''}</div>
    <div style="margin-top: 8px; font-size: 0.8em; opacity: 0.7;">
        Speaker: {speaker_str} | Importance: {item.importance:.2f} | Entities: {len(item.entities)}
    </div>
</div>
"""
    
    return html


def get_kg_facts():
    """Get knowledge graph facts."""
    memory = get_graph_memory()
    if not memory:
        return "No knowledge graph yet."
    
    triples = memory.knowledge_graph.triples[:30]
    
    if not triples:
        return "No facts in knowledge graph yet."
    
    html = ""
    for t in triples:
        html += f"""
<div style="background: #f0f4f8; padding: 8px 12px; border-radius: 6px; 
            margin: 4px 0; border-left: 3px solid #4a90d9;">
    <strong>{t.subject}</strong> → <em>{t.predicate}</em> → <strong>{t.obj}</strong>
</div>
"""
    
    return html


def get_stats_display():
    """Get memory statistics."""
    stats = get_memory_stats()
    
    if not stats:
        return """
<div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px;">
    <div class="stat-card"><div class="stat-value">0</div><div class="stat-label">Memories</div></div>
    <div class="stat-card"><div class="stat-value">0</div><div class="stat-label">KG Facts</div></div>
    <div class="stat-card"><div class="stat-value">0</div><div class="stat-label">Entities</div></div>
    <div class="stat-card"><div class="stat-value">0</div><div class="stat-label">Sessions</div></div>
</div>
"""
    
    return f"""
<div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px;">
    <div style="background: white; border-radius: 12px; padding: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center;">
        <div style="font-size: 2em; font-weight: bold; color: #4a90d9;">{stats.get('total_memories', 0)}</div>
        <div style="color: #666; font-size: 0.9em;">Memories</div>
    </div>
    <div style="background: white; border-radius: 12px; padding: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center;">
        <div style="font-size: 2em; font-weight: bold; color: #9b59b6;">{stats.get('kg_triples', 0)}</div>
        <div style="color: #666; font-size: 0.9em;">KG Facts</div>
    </div>
    <div style="background: white; border-radius: 12px; padding: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center;">
        <div style="font-size: 2em; font-weight: bold; color: #27ae60;">{stats.get('indexed_entities', 0)}</div>
        <div style="color: #666; font-size: 0.9em;">Entities</div>
    </div>
    <div style="background: white; border-radius: 12px; padding: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center;">
        <div style="font-size: 2em; font-weight: bold; color: #e74c3c;">{stats.get('indexed_speakers', 0)}</div>
        <div style="color: #666; font-size: 0.9em;">Speakers</div>
    </div>
</div>
"""


def clear_chat():
    """Clear chat history."""
    return [], "", "Chat cleared"


def refresh_memory():
    """Refresh memory displays."""
    return get_memories_display(), get_kg_facts(), get_stats_display()


def create_ui():
    """Create the Gradio UI."""
    
    with gr.Blocks(
        title="Multi-Agent System",
    ) as demo:
        
        # Header
        gr.Markdown("""
# 🤖 Multi-Agent System with Memory V3

**Powered by LangGraph + Ollama + Knowledge Graph**

A team of AI agents collaborate to answer your questions:
- 🔍 **Research** - Information gathering
- 💻 **Code** - Programming tasks  
- 📊 **Analysis** - Complex reasoning
- ✍️ **Writer** - Content creation
        """)
        
        with gr.Tabs():
            # ============================================
            # Tab 1: Chat
            # ============================================
            with gr.Tab("💬 Chat"):
                with gr.Row():
                    with gr.Column(scale=3):
                        chatbot = gr.Chatbot(
                            label="Conversation",
                            height=500,
                        )
                        
                        with gr.Row():
                            msg = gr.Textbox(
                                placeholder="Ask me anything... (e.g., 'Research and compare Python vs Rust for systems programming')",
                                label="Message",
                                scale=4,
                                show_label=False,
                            )
                            submit = gr.Button("Send 🚀", variant="primary", scale=1)
                        
                        with gr.Row():
                            clear = gr.Button("🗑️ Clear Chat")
                            
                        gr.Examples(
                            examples=[
                                "What are the best practices for building REST APIs?",
                                "Write a Python function to parse JSON and explain it",
                                "Compare microservices vs monolithic architecture",
                                "Research the latest developments in AI agents",
                            ],
                            inputs=msg,
                            label="Example Prompts",
                        )
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### ⚙️ Settings")
                        
                        model = gr.Dropdown(
                            choices=get_available_models(),
                            value="llama3.2:latest",
                            label="Model",
                        )
                        
                        max_iter = gr.Slider(
                            minimum=3,
                            maximum=15,
                            value=8,
                            step=1,
                            label="Max Iterations",
                        )
                        
                        gr.Markdown("### 📋 Agent Activity")
                        activity_log = gr.Textbox(
                            label="",
                            lines=15,
                            max_lines=20,
                            interactive=False,
                            show_label=False,
                        )
            
            # ============================================
            # Tab 2: Memory Browser
            # ============================================
            with gr.Tab("🧠 Memory Browser"):
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("### 📚 Stored Memories")
                        memory_display = gr.HTML(
                            value=get_memories_display(),
                            label="Memories",
                        )
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### 🔗 Knowledge Graph Facts")
                        kg_display = gr.HTML(
                            value=get_kg_facts(),
                            label="KG Facts",
                        )
                
                refresh_btn = gr.Button("🔄 Refresh", variant="secondary")
            
            # ============================================
            # Tab 3: Dashboard
            # ============================================
            with gr.Tab("📊 Dashboard"):
                gr.Markdown("### 📈 Memory Statistics")
                stats_display = gr.HTML(
                    value=get_stats_display(),
                    label="Stats",
                )
                
                gr.Markdown("""
### 🏗️ System Architecture

```
                    ┌─────────────────┐
                    │   SUPERVISOR    │
                    │  (Orchestrator) │
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
   ┌────▼────┐         ┌────▼────┐         ┌────▼────┐
   │Research │         │  Code   │         │Analysis │
   │  Agent  │         │  Agent  │         │  Agent  │
   └─────────┘         └─────────┘         └─────────┘
        │                    │                    │
        └────────────────────┼────────────────────┘
                             │
                    ┌────────▼────────┐
                    │     WRITER      │
                    │     Agent       │
                    └─────────────────┘
```

### 🧠 Memory V3 Features

- **Knowledge Graph**: Entities and relationships for multi-hop reasoning
- **Query Decomposition**: Break complex questions into sub-queries
- **Re-ranking**: Cross-attention scoring for better retrieval
- **Date Normalization**: Convert relative dates to absolute
- **Answer Extraction**: Pattern-based answer extraction
                """)
                
                stats_refresh_btn = gr.Button("🔄 Refresh Stats", variant="secondary")
        
        # Event handlers
        submit.click(
            process_message,
            inputs=[msg, chatbot, model, max_iter],
            outputs=[chatbot, msg, activity_log],
        )
        
        msg.submit(
            process_message,
            inputs=[msg, chatbot, model, max_iter],
            outputs=[chatbot, msg, activity_log],
        )
        
        clear.click(
            clear_chat,
            outputs=[chatbot, msg, activity_log],
        )
        
        refresh_btn.click(
            refresh_memory,
            outputs=[memory_display, kg_display, stats_display],
        )
        
        stats_refresh_btn.click(
            refresh_memory,
            outputs=[memory_display, kg_display, stats_display],
        )
        
        gr.Markdown("""
---
**Tips:**
- Ollama runs locally - no rate limits!
- The agents collaborate automatically
- Memory persists across conversations
- Check the Memory Browser to see what's stored
        """)
    
    return demo


def main():
    """Main entry point."""
    import sys
    
    parser = argparse.ArgumentParser(description="Multi-Agent Web UI")
    parser.add_argument("--port", "-p", type=int, default=7860, help="Port (default: 7860)")
    parser.add_argument("--share", "-s", action="store_true", help="Create public URL")
    
    args = parser.parse_args()
    
    print("🚀 Starting Multi-Agent UI with Memory V3...", flush=True)
    print(f"   Port: {args.port}", flush=True)
    
    if not check_ollama():
        print("⚠️  Ollama not running. Start with: ollama serve", flush=True)
    else:
        models = get_available_models()
        print(f"   Available models: {', '.join(models[:5])}", flush=True)
    
    sys.stdout.flush()
    
    demo = create_ui()
    print(f"✅ UI created! Launching at http://localhost:{args.port}", flush=True)
    sys.stdout.flush()
    
    demo.launch(
        server_port=args.port,
        share=args.share,
        show_error=True,
    )


if __name__ == "__main__":
    main()
