"""
Web UI for the Memory Agent.

A beautiful Gradio-based interface similar to Claude's UI.
Features:
- Chat interface with message history
- Memory browser with visualization
- Stats dashboard
- Session management
"""

import asyncio
from datetime import datetime
from typing import Any

import gradio as gr
import pandas as pd

from llm_memory import MemorySystem, MemorySystemConfig, MemoryType
from llm_memory.agent.memory_agent import MemoryAgent, AgentConfig


# Global state
agent: MemoryAgent | None = None
current_session = "default"


def get_loop():
    """Get or create event loop."""
    try:
        return asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop


def run_async(coro):
    """Run async function from sync context."""
    loop = get_loop()
    return loop.run_until_complete(coro)


async def initialize_agent(model_name: str, session_id: str) -> str:
    """Initialize the agent with given model."""
    global agent, current_session
    
    if agent:
        await agent.close()
    
    current_session = session_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    config = AgentConfig(
        ollama_model=model_name,
        session_id=current_session,
        temperature=0.7,
    )
    
    agent = MemoryAgent(config)
    await agent.initialize()
    
    return f"✓ Agent initialized with {model_name}\n📍 Session: {current_session}"


def init_agent_sync(model_name: str, session_id: str) -> str:
    """Sync wrapper for initialize_agent."""
    return run_async(initialize_agent(model_name, session_id))


async def chat_async(message: str, history: list) -> tuple[str, list]:
    """Send a message and get response."""
    global agent
    
    if not agent:
        history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": "⚠️ Please initialize the agent first (Settings tab)"}
        ]
        return "", history
    
    try:
        response = await agent.chat(message)
        history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": response}
        ]
        return "", history
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": error_msg}
        ]
        return "", history


def chat_sync(message: str, history: list) -> tuple[str, list]:
    """Sync wrapper for chat."""
    if history is None:
        history = []
    return run_async(chat_async(message, history))


def get_memory_stats() -> dict:
    """Get memory statistics."""
    global agent
    if not agent:
        return {"error": "Agent not initialized"}
    return agent.get_memory_stats()


def format_stats_html() -> str:
    """Format stats as modern cards for the dashboard."""
    stats = get_memory_stats()
    
    if "error" in stats:
        return f"<div style='padding: 20px; color: #ff9800; background: #fff3e0; border-radius: 8px;'>⚠️ {stats['error']}</div>"
    
    total = stats.get('total_memories', 0)
    sessions = stats.get('active_sessions', 0)
    by_type = stats.get('by_type', {})
    
    html = f"""
    <div style="display: flex; flex-direction: column; gap: 24px; padding: 10px;">
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px;">
            <div style="background: white; border: 1px solid #e0e0e0; padding: 20px; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);">
                <div style="color: #6b7280; font-size: 14px; font-weight: 500; margin-bottom: 8px;">TOTAL MEMORIES</div>
                <div style="font-size: 32px; font-weight: 700; color: #111827;">{total}</div>
                <div style="height: 4px; background: #6366f1; width: 40%; margin-top: 12px; border-radius: 2px;"></div>
            </div>
            <div style="background: white; border: 1px solid #e0e0e0; padding: 20px; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);">
                <div style="color: #6b7280; font-size: 14px; font-weight: 500; margin-bottom: 8px;">ACTIVE SESSIONS</div>
                <div style="font-size: 32px; font-weight: 700; color: #111827;">{sessions}</div>
                <div style="height: 4px; background: #ec4899; width: 40%; margin-top: 12px; border-radius: 2px;"></div>
            </div>
            <div style="background: white; border: 1px solid #e0e0e0; padding: 20px; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);">
                <div style="color: #6b7280; font-size: 14px; font-weight: 500; margin-bottom: 8px;">SESSION ID</div>
                <div style="font-size: 20px; font-weight: 700; color: #111827; overflow: hidden; text-overflow: ellipsis;">{current_session}</div>
                <div style="height: 4px; background: #10b981; width: 40%; margin-top: 12px; border-radius: 2px;"></div>
            </div>
        </div>
        
        <div style="background: white; border: 1px solid #e0e0e0; padding: 24px; border-radius: 12px;">
            <h3 style="margin-top: 0; margin-bottom: 20px; font-size: 18px; color: #111827;">Memory Composition</h3>
            <div style="display: flex; flex-direction: column; gap: 16px;">
    """
    
    type_meta = {
        'semantic': ('Semantic (Facts)', '#10b981', '🧠'),
        'episodic': ('Episodic (Events)', '#f59e0b', '📝'),
        'short_term': ('Short-Term (Context)', '#6366f1', '💬'),
    }
    
    max_count = max(by_type.values()) if by_type and any(by_type.values()) else 1
    
    for m_type, count in by_type.items():
        name, color, icon = type_meta.get(m_type, (m_type.capitalize(), '#6b7280', '📦'))
        percentage = (count / max_count) * 100
        html += f"""
                <div style="display: flex; align-items: center; gap: 12px;">
                    <div style="font-size: 20px;">{icon}</div>
                    <div style="flex: 1;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                            <span style="font-size: 14px; font-weight: 500; color: #374151;">{name}</span>
                            <span style="font-size: 14px; font-weight: 600; color: #111827;">{count}</span>
                        </div>
                        <div style="height: 8px; background: #f3f4f6; border-radius: 4px; overflow: hidden;">
                            <div style="height: 100%; background: {color}; width: {percentage}%; transition: width 0.5s ease-out;"></div>
                        </div>
                    </div>
                </div>
        """
    
    html += """
            </div>
        </div>
    </div>
    """
    return html


async def get_memory_cards_html() -> str:
    """Generate visual cards for memories."""
    global agent
    if not agent or not agent.memory:
        return "<div style='text-align: center; padding: 40px; color: #6b7280;'>Initialize agent to view memories</div>"
    
    memories = []
    for mem_id, mem in list(agent.memory._memories.items()):
        memories.append(mem)
    
    if not memories:
        return "<div style='text-align: center; padding: 40px; color: #6b7280;'>No memories stored yet. Start chatting!</div>"
    
    # Sort by creation
    memories.sort(key=lambda x: x.created_at, reverse=True)
    
    html = '<div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 16px; padding: 10px;">'
    
    for mem in memories:
        type_colors = {
            MemoryType.SEMANTIC: ('#10b981', '🧠'),
            MemoryType.EPISODIC: ('#f59e0b', '📝'),
            MemoryType.SHORT_TERM: ('#6366f1', '💬'),
        }
        color, icon = type_colors.get(mem.memory_type, ('#6b7280', '📦'))
        
        strength_pct = mem.current_strength * 100
        importance_pct = mem.importance_score * 100
        
        # Color based on strength
        strength_color = "#ef4444" if mem.current_strength < 0.3 else "#f59e0b" if mem.current_strength < 0.7 else "#10b981"
        
        html += f"""
        <div style="background: white; border: 1px solid #e5e7eb; border-radius: 12px; padding: 16px; transition: transform 0.2s, box-shadow 0.2s; position: relative; overflow: hidden;">
            <div style="position: absolute; top: 0; left: 0; width: 4px; height: 100%; background: {color};"></div>
            
            <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 12px;">
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="font-size: 18px;">{icon}</span>
                    <span style="font-size: 12px; font-weight: 600; color: {color}; text-transform: uppercase; letter-spacing: 0.05em;">{mem.memory_type.value}</span>
                </div>
                <span style="font-size: 11px; color: #9ca3af;">{mem.created_at.strftime('%b %d, %H:%M')}</span>
            </div>
            
            <div style="font-size: 14px; color: #1f2937; line-height: 1.5; margin-bottom: 16px; min-height: 42px;">
                {mem.content[:150]}{'...' if len(mem.content) > 150 else ''}
            </div>
            
            <div style="display: flex; flex-direction: column; gap: 8px;">
                <div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                        <span style="font-size: 11px; font-weight: 500; color: #6b7280;">Strength</span>
                        <span style="font-size: 11px; font-weight: 600; color: {strength_color};">{strength_pct:.0f}%</span>
                    </div>
                    <div style="height: 4px; background: #f3f4f6; border-radius: 2px;">
                        <div style="height: 100%; background: {strength_color}; width: {strength_pct}%; border-radius: 2px;"></div>
                    </div>
                </div>
                <div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                        <span style="font-size: 11px; font-weight: 500; color: #6b7280;">Importance</span>
                        <span style="font-size: 11px; font-weight: 600; color: #374151;">{importance_pct:.0f}%</span>
                    </div>
                    <div style="height: 4px; background: #f3f4f6; border-radius: 2px;">
                        <div style="height: 100%; background: #374151; width: {importance_pct}%; border-radius: 2px;"></div>
                    </div>
                </div>
            </div>
            
            {f'<div style="margin-top: 12px; display: flex; flex-wrap: wrap; gap: 4px;">' + 
             ''.join([f'<span style="background: #f3f4f6; color: #4b5563; font-size: 10px; padding: 2px 6px; border-radius: 4px;">#{t}</span>' for t in mem.metadata.tags]) + 
             '</div>' if mem.metadata.tags else ''}
        </div>
        """
    
    html += '</div>'
    return html


def get_memories_dataframe():
    """Fallback dataframe view."""
    memories = run_async(get_all_memories())
    if not memories: return None
    df = pd.DataFrame(memories)
    df = df[["type", "content", "strength", "importance", "created", "active"]]
    return df


async def recall_memories(query: str) -> str:
    """Search memories and return formatted results."""
    global agent
    if not agent or not query.strip(): return ""
    
    memories = await agent.recall(query, limit=5)
    if not memories: return f"<div style='padding: 20px; text-align: center; color: #6b7280;'>No results for '{query}'</div>"
    
    html = f"<div style='margin-bottom: 15px; font-weight: 600; color: #374151;'>Search results for '{query}':</div>"
    html += "<div style='display: flex; flex-direction: column; gap: 12px;'>"
    
    for mem in memories:
        m_type = mem['type']
        color = '#10b981' if m_type == 'semantic' else '#f59e0b' if m_type == 'episodic' else '#6366f1'
        html += f"""
        <div style="background: #f9fafb; border-left: 4px solid {color}; padding: 12px; border-radius: 4px;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                <span style="font-size: 10px; font-weight: 700; color: {color}; text-transform: uppercase;">{m_type}</span>
                <span style="font-size: 10px; color: #9ca3af;">Relevance: {mem['score']:.2f}</span>
            </div>
            <div style="font-size: 13px; color: #111827;">{mem['content']}</div>
        </div>
        """
    html += "</div>"
    return html


def recall_sync(query: str) -> str:
    return run_async(recall_memories(query))


async def remember_something(content: str, mem_type: str, tags: str) -> str:
    """Manually store a memory."""
    global agent
    if not agent or not content.strip(): return "Error: Agent not ready or empty content"
    
    type_map = {"Semantic (Facts)": MemoryType.SEMANTIC, "Episodic (Experiences)": MemoryType.EPISODIC}
    memory_type = type_map.get(mem_type, MemoryType.SEMANTIC)
    tag_list = [t.strip() for t in tags.split(",") if t.strip()]
    
    mem = await agent.memory.remember(content, memory_type=memory_type, tags=tag_list)
    return f"✓ Successfully stored memory (ID: {mem.id[:12]})"


def new_session(session_name: str) -> tuple[list, str]:
    """Start a new session."""
    global agent, current_session
    if not agent: return [], "⚠️ Initialize agent first"
    
    new_name = session_name.strip() or f"session_{datetime.now().strftime('%H%M%S')}"
    current_session = new_name
    agent.config.session_id = new_name
    return [], f"✓ New session started: {new_name}"


def create_ui():
    """Create the Gradio UI with Claude-like aesthetics."""
    
    custom_css = """
    .main-container { padding: 0 !important; }
    .header { background: #f9fafb; border-bottom: 1px solid #e5e7eb; padding: 1rem 2rem; margin-bottom: 1rem; }
    .tab-nav { border-bottom: none !important; margin-bottom: 0 !important; }
    .gradio-container { font-family: 'Inter', system-ui, -apple-system, sans-serif !important; }
    .chat-container { border: 1px solid #e5e7eb; border-radius: 12px; background: white; overflow: hidden; }
    .sidebar { background: #f9fafb; border-radius: 12px; padding: 1.5rem; height: 100%; border: 1px solid #e5e7eb; }
    button.primary { background: #6366f1 !important; border: none !important; }
    button.primary:hover { background: #4f46e5 !important; }
    .memory-card:hover { transform: translateY(-2px); box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1); }
    """
    
    with gr.Blocks(title="LLM Memory Agent", css=custom_css) as demo:
        
        with gr.Div(elem_classes="header"):
            with gr.Row():
                with gr.Column(scale=8):
                    gr.Markdown("# 🧠 LLM Memory Agent")
                    gr.Markdown("Cognitive Architecture for Long-Term AI Memory")
                with gr.Column(scale=2):
                    refresh_btn = gr.Button("🔄 Sync System", size="sm")
        
        with gr.Tabs(elem_id="main-tabs") as tabs:
            
            # ============ CHAT TAB ============
            with gr.Tab("💬 Conversations", id="chat"):
                with gr.Row(equal_height=True):
                    with gr.Column(scale=3):
                        with gr.Div(elem_classes="chat-container"):
                            chatbot = gr.Chatbot(
                                show_label=False,
                                height=600,
                                type="messages",
                                show_share_button=False,
                                layout="bubble",
                            )
                            with gr.Row(variant="compact"):
                                msg_input = gr.Textbox(
                                    placeholder="Message your agent...",
                                    show_label=False,
                                    scale=9,
                                    container=False,
                                )
                                send_btn = gr.Button("↑", variant="primary", scale=1)
                        
                        with gr.Row(style={"margin-top": "10px"}):
                            clear_btn = gr.Button("🗑️ Reset chat", size="sm", variant="secondary")
                            new_sess_input = gr.Textbox(placeholder="Name (optional)", show_label=False, container=False, scale=2)
                            new_sess_btn = gr.Button("🆕 New Session", size="sm")
                    
                    with gr.Column(scale=1):
                        with gr.Div(elem_classes="sidebar"):
                            gr.Markdown("### 📊 System Pulse")
                            stats_display = gr.HTML(value=format_stats_html)
                            gr.Markdown("---")
                            gr.Markdown("### 🔦 Quick Recall")
                            mini_search = gr.Textbox(placeholder="Ask memory...", show_label=False)
                            mini_results = gr.HTML()
                            mini_search.change(recall_sync, mini_search, mini_results)
            
            # ============ MEMORY BROWSER TAB ============
            with gr.Tab("🗃️ Knowledge Base", id="memories"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 🔍 Intelligent Search")
                        search_box = gr.Textbox(placeholder="Search across all memory tiers...", show_label=False)
                        search_btn = gr.Button("Search Memory", variant="primary")
                        
                        gr.Markdown("### ➕ Manual Entry")
                        m_content = gr.Textbox(placeholder="Fact or experience...", lines=3, label="Content")
                        with gr.Row():
                            m_type = gr.Dropdown(choices=["Semantic (Facts)", "Episodic (Experiences)"], value="Semantic (Facts)", label="Type")
                            m_tags = gr.Textbox(placeholder="work, hobby", label="Tags")
                        m_btn = gr.Button("💾 Commit to Memory")
                        m_status = gr.Markdown()
                        
                    with gr.Column(scale=3):
                        gr.Markdown("### 🌐 Global Memory Web")
                        with gr.Row():
                            filter_type = gr.Radio(["All", "Semantic", "Episodic"], value="All", label="Filter", scale=2)
                            sort_by = gr.Dropdown(["Recent", "Importance", "Strength"], value="Recent", label="Sort By", scale=1)
                        
                        memory_grid = gr.HTML(value=run_async(get_memory_cards_html))
                        refresh_grid_btn = gr.Button("🔄 Refresh View", size="sm")
            
            # ============ SETTINGS TAB ============
            with gr.Tab("⚙️ Configuration", id="settings"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 🤖 Model Settings")
                        model_pick = gr.Dropdown(
                            choices=["llama3.2", "mistral", "gemma3:27b", "qwen2.5:32b"],
                            value="gemma3:27b",
                            label="Ollama Model"
                        )
                        temp_slider = gr.Slider(0, 1, 0.7, label="Temperature")
                        
                        gr.Markdown("### 💾 Persistence")
                        db_path = gr.Textbox(value="./data/memory.db", label="SQLite Path")
                        
                        init_go = gr.Button("🚀 Initialize/Restart Agent", variant="primary")
                        init_out = gr.Textbox(label="System Logs", interactive=False)
                    
                    with gr.Column():
                        gr.Markdown("### 🛠️ Advanced Controls")
                        gr.Button("🧹 Run Garbage Collection", size="sm")
                        gr.Button("🧪 Force Consolidation", size="sm")
                        gr.Button("⚠️ Wipe All Memories", size="sm", variant="stop")
        
        # Event Handlers
        msg_input.submit(chat_sync, [msg_input, chatbot], [msg_input, chatbot])
        send_btn.click(chat_sync, [msg_input, chatbot], [msg_input, chatbot])
        clear_btn.click(lambda: [], None, chatbot)
        new_sess_btn.click(new_session, new_sess_input, [chatbot, new_sess_input])
        
        init_go.click(init_agent_sync, [model_pick, gr.State("default")], init_out)
        
        m_btn.click(remember_sync, [m_content, m_type, m_tags], m_status)
        
        refresh_btn.click(format_stats_html, None, stats_display)
        refresh_btn.click(get_memory_cards_html, None, memory_grid)
        refresh_grid_btn.click(get_memory_cards_html, None, memory_grid)

    return demo


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    
    demo = create_ui()
    # In Gradio 6.0, theme and css go into launch()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
