#!/usr/bin/env python3
"""
Streamlit Web UI for Multi-Agent System with Memory V3.

A more stable alternative to Gradio.

Usage:
    streamlit run llm_memory/agents/streamlit_ui.py
    streamlit run llm_memory/agents/streamlit_ui.py -- --port 8501
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import json

# Add package to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import streamlit as st
except ImportError:
    print("❌ Streamlit not installed. Install with: pip install streamlit")
    sys.exit(1)

from llm_memory.agents.config import AgentConfig, ModelConfig
from llm_memory.agents.graph import (
    create_graph,
    run_graph_stream,
    get_graph_memory,
    get_memory_stats,
)
from llm_memory.agents.state import Message


# Page config
st.set_page_config(
    page_title="Multi-Agent System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


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
    return ["llama3.2:latest", "qwen2.5:3b", "qwen2.5:32b", "gemma3:27b"]


def initialize_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "graph" not in st.session_state:
        st.session_state.graph = None
    if "agent_log" not in st.session_state:
        st.session_state.agent_log = []
    if "user_id" not in st.session_state:
        st.session_state.user_id = f"user_{datetime.now().strftime('%Y%m%d')}"


def create_graph_instance(model_name: str):
    """Create or update the agent graph."""
    config = AgentConfig(
        default_model=ModelConfig(
            provider="ollama",
            model_name=model_name,
            temperature=0.7,
        ),
        enable_memory=True,
        memory_persist_path="./ui_agent_memory",
    )
    st.session_state.graph = create_graph(config=config, user_id=st.session_state.user_id)


def process_message(user_message: str, model: str, max_iterations: int):
    """Process user message through multi-agent system."""
    if not check_ollama():
        return "❌ Ollama is not running! Start it with: `ollama serve`", []
    
    if st.session_state.graph is None:
        create_graph_instance(model)
    
    agent_log = []
    final_response = None
    
    try:
        session_id = f"ui-{datetime.now().strftime('%H%M%S')}"
        
        for node_name, update in run_graph_stream(
            st.session_state.graph,
            user_message,
            thread_id=session_id,
            max_iterations=max_iterations
        ):
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            if node_name == "supervisor":
                msgs = update.get("messages")
                if isinstance(msgs, Message):
                    routing = msgs.content.replace("[Supervisor]", "").strip()[:100]
                    agent_log.append(f"[{timestamp}] 📋 SUPERVISOR: {routing}")
            
            elif node_name in ["research", "code", "analysis", "writer"]:
                icons = {"research": "🔍", "code": "💻", "analysis": "📊", "writer": "✍️"}
                agent_log.append(f"[{timestamp}] {icons[node_name]} {node_name.upper()} working...")
            
            elif node_name == "compile_response":
                final_response = update.get("final_response", "")
                agent_log.append(f"[{timestamp}] ✅ Response compiled")
        
        if final_response:
            return final_response, agent_log
        else:
            return "⚠️ No response generated. Try again or increase iterations.", agent_log
    
    except Exception as e:
        return f"❌ Error: {str(e)}", agent_log


def main():
    """Main Streamlit app."""
    initialize_session_state()
    
    # Sidebar
    with st.sidebar:
        st.title("⚙️ Settings")
        
        # Model selection
        available_models = get_available_models()
        selected_model = st.selectbox(
            "Model",
            available_models,
            index=0 if available_models else 0,
        )
        
        max_iterations = st.slider("Max Iterations", 3, 15, 8)
        
        st.divider()
        
        # Status
        if check_ollama():
            st.success("✅ Ollama running")
        else:
            st.error("❌ Ollama not running")
            st.code("ollama serve")
        
        st.divider()
        
        # Memory stats
        st.subheader("📊 Memory Stats")
        stats = get_memory_stats()
        if stats:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Memories", stats.get("total_memories", 0))
                st.metric("Entities", stats.get("indexed_entities", 0))
            with col2:
                st.metric("KG Facts", stats.get("kg_triples", 0))
                st.metric("Speakers", stats.get("indexed_speakers", 0))
        else:
            st.info("No memory data yet")
        
        st.divider()
        
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.session_state.agent_log = []
            st.rerun()
    
    # Main content
    st.title("🤖 Multi-Agent System with Memory V3")
    st.markdown("""
    A team of AI agents collaborate to answer your questions:
    **🔍 Research** · **💻 Code** · **📊 Analysis** · **✍️ Writer**
    """)
    
    # Tabs
    tab_chat, tab_memory, tab_kg = st.tabs(["💬 Chat", "🧠 Memory", "🔗 Knowledge Graph"])
    
    with tab_chat:
        # Chat history
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask me anything..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Process with agents
            with st.chat_message("assistant"):
                with st.spinner("Agents working..."):
                    response, agent_log = process_message(prompt, selected_model, max_iterations)
                    st.session_state.agent_log = agent_log
                st.markdown(response)
            
            st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Agent activity expander
        if st.session_state.agent_log:
            with st.expander("🔍 Agent Activity"):
                st.code("\n".join(st.session_state.agent_log))
    
    with tab_memory:
        st.subheader("📚 Stored Memories")
        
        memory = get_graph_memory()
        if memory and memory.memories:
            # Memory list
            items = list(memory.memories.values())[:30]
            
            for item in items:
                type_icons = {
                    "fact": "🧠",
                    "event": "📅",
                    "conversation": "💬",
                    "preference": "❤️",
                }
                icon = type_icons.get(item.memory_type, "📝")
                
                with st.container():
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.markdown(f"**{icon} {item.memory_type.upper()}**")
                        st.markdown(item.content[:300])
                    with col2:
                        if item.event_date:
                            st.caption(f"📅 {item.event_date}")
                        if item.speaker:
                            st.caption(f"👤 {item.speaker}")
                        st.caption(f"⭐ {item.importance:.2f}")
                    st.divider()
        else:
            st.info("No memories stored yet. Start chatting to build memory!")
    
    with tab_kg:
        st.subheader("🔗 Knowledge Graph Facts")
        
        memory = get_graph_memory()
        if memory and memory.knowledge_graph.triples:
            triples = memory.knowledge_graph.triples[:50]
            
            # Display as table
            data = []
            for t in triples:
                data.append({
                    "Subject": t.subject,
                    "Predicate": t.predicate,
                    "Object": t.obj,
                    "Confidence": f"{t.confidence:.2f}",
                })
            
            st.dataframe(data, use_container_width=True)
            
            # Stats
            st.metric("Total Triples", len(memory.knowledge_graph.triples))
        else:
            st.info("No knowledge graph facts yet. Memories will build the graph!")


if __name__ == "__main__":
    main()
