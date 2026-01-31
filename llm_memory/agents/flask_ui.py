#!/usr/bin/env python3
"""
Flask Web UI for Multi-Agent System with Memory V3.

A minimal, stable alternative to Gradio/Streamlit.

Usage:
    python llm_memory/agents/flask_ui.py
    python llm_memory/agents/flask_ui.py --port 5000
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import json
import argparse

# Add package to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from flask import Flask, render_template_string, request, jsonify
except ImportError:
    print("❌ Flask not installed. Install with: pip install flask")
    sys.exit(1)

from llm_memory.agents.config import AgentConfig, ModelConfig
from llm_memory.agents.graph import (
    create_graph,
    run_graph_stream,
    get_graph_memory,
    get_memory_stats,
)
from llm_memory.agents.state import Message

app = Flask(__name__)

# Global state
_graph = None
_config = None
_user_id = "flask_user"
_chat_history = []

# HTML Template
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Multi-Agent System</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { 
            text-align: center; 
            color: white; 
            padding: 20px;
            margin-bottom: 20px;
        }
        .header h1 { font-size: 2.5em; margin-bottom: 10px; }
        .header p { opacity: 0.9; }
        
        .main-grid {
            display: grid;
            grid-template-columns: 1fr 300px;
            gap: 20px;
        }
        
        .chat-container {
            background: white;
            border-radius: 16px;
            padding: 20px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        }
        
        .chat-messages {
            height: 500px;
            overflow-y: auto;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 12px;
            margin-bottom: 15px;
        }
        
        .message {
            margin-bottom: 15px;
            padding: 12px 15px;
            border-radius: 12px;
            max-width: 85%;
        }
        
        .user-message {
            background: #667eea;
            color: white;
            margin-left: auto;
        }
        
        .assistant-message {
            background: white;
            border: 1px solid #e1e4e8;
            color: #333;
        }
        
        .input-row {
            display: flex;
            gap: 10px;
        }
        
        .input-row input {
            flex: 1;
            padding: 12px 15px;
            border: 2px solid #e1e4e8;
            border-radius: 10px;
            font-size: 16px;
            outline: none;
            transition: border-color 0.2s;
        }
        
        .input-row input:focus {
            border-color: #667eea;
        }
        
        .input-row button {
            padding: 12px 25px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 16px;
            cursor: pointer;
            transition: transform 0.2s, background 0.2s;
        }
        
        .input-row button:hover {
            background: #5a6fd6;
            transform: translateY(-2px);
        }
        
        .input-row button:disabled {
            background: #ccc;
            transform: none;
            cursor: not-allowed;
        }
        
        .sidebar {
            display: flex;
            flex-direction: column;
            gap: 15px;
        }
        
        .card {
            background: white;
            border-radius: 12px;
            padding: 15px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        }
        
        .card h3 {
            color: #333;
            margin-bottom: 10px;
            font-size: 1em;
        }
        
        .stat-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
        }
        
        .stat-item {
            text-align: center;
            padding: 10px;
            background: #f8f9fa;
            border-radius: 8px;
        }
        
        .stat-value {
            font-size: 1.5em;
            font-weight: bold;
            color: #667eea;
        }
        
        .stat-label {
            font-size: 0.8em;
            color: #666;
        }
        
        .model-select {
            width: 100%;
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 8px;
            font-size: 14px;
        }
        
        .activity-log {
            font-family: monospace;
            font-size: 11px;
            background: #1a1a2e;
            color: #00ff88;
            padding: 10px;
            border-radius: 8px;
            height: 150px;
            overflow-y: auto;
        }
        
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
        }
        
        .spinner {
            border: 3px solid #f3f3f3;
            border-top: 3px solid #667eea;
            border-radius: 50%;
            width: 30px;
            height: 30px;
            animation: spin 1s linear infinite;
            margin: 0 auto 10px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .status-badge {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: bold;
        }
        
        .status-online { background: #d4edda; color: #155724; }
        .status-offline { background: #f8d7da; color: #721c24; }
        
        @media (max-width: 768px) {
            .main-grid { grid-template-columns: 1fr; }
            .chat-messages { height: 400px; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 Multi-Agent System</h1>
            <p>Memory V3 with Knowledge Graph • Powered by LangGraph + Ollama</p>
        </div>
        
        <div class="main-grid">
            <div class="chat-container">
                <div class="chat-messages" id="chatMessages">
                    <div class="message assistant-message">
                        <strong>🤖 Assistant:</strong> Hello! I'm a team of AI agents ready to help. 
                        I have: 🔍 Research, 💻 Code, 📊 Analysis, and ✍️ Writing capabilities. 
                        What can I help you with today?
                    </div>
                </div>
                
                <div class="loading" id="loading">
                    <div class="spinner"></div>
                    <div id="loadingText">Agents working...</div>
                </div>
                
                <div class="input-row">
                    <input type="text" id="userInput" placeholder="Ask me anything..." 
                           onkeypress="if(event.key==='Enter') sendMessage()">
                    <button onclick="sendMessage()" id="sendBtn">Send 🚀</button>
                </div>
            </div>
            
            <div class="sidebar">
                <div class="card">
                    <h3>⚙️ Settings</h3>
                    <select class="model-select" id="modelSelect">
                        <option value="llama3.2:latest">llama3.2:latest</option>
                        <option value="qwen2.5:3b">qwen2.5:3b</option>
                        <option value="qwen2.5:32b">qwen2.5:32b</option>
                        <option value="gemma3:27b">gemma3:27b</option>
                    </select>
                    <br><br>
                    <div id="ollamaStatus"></div>
                </div>
                
                <div class="card">
                    <h3>📊 Memory Stats</h3>
                    <div class="stat-grid" id="statsGrid">
                        <div class="stat-item">
                            <div class="stat-value" id="statMemories">0</div>
                            <div class="stat-label">Memories</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-value" id="statKG">0</div>
                            <div class="stat-label">KG Facts</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-value" id="statEntities">0</div>
                            <div class="stat-label">Entities</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-value" id="statSpeakers">0</div>
                            <div class="stat-label">Speakers</div>
                        </div>
                    </div>
                </div>
                
                <div class="card">
                    <h3>📋 Agent Activity</h3>
                    <div class="activity-log" id="activityLog">
                        Waiting for activity...
                    </div>
                </div>
                
                <div class="card">
                    <button onclick="clearChat()" style="width:100%;padding:10px;background:#dc3545;color:white;border:none;border-radius:8px;cursor:pointer;">
                        🗑️ Clear Chat
                    </button>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // Check Ollama status
        async function checkOllama() {
            try {
                const resp = await fetch('/api/status');
                const data = await resp.json();
                
                const statusEl = document.getElementById('ollamaStatus');
                if (data.ollama_running) {
                    statusEl.innerHTML = '<span class="status-badge status-online">✅ Ollama Running</span>';
                } else {
                    statusEl.innerHTML = '<span class="status-badge status-offline">❌ Ollama Offline</span>';
                }
                
                // Update stats
                document.getElementById('statMemories').textContent = data.stats.total_memories || 0;
                document.getElementById('statKG').textContent = data.stats.kg_triples || 0;
                document.getElementById('statEntities').textContent = data.stats.indexed_entities || 0;
                document.getElementById('statSpeakers').textContent = data.stats.indexed_speakers || 0;
            } catch (e) {
                console.error('Status check failed:', e);
            }
        }
        
        // Send message
        async function sendMessage() {
            const input = document.getElementById('userInput');
            const message = input.value.trim();
            if (!message) return;
            
            // Add user message
            addMessage(message, 'user');
            input.value = '';
            
            // Show loading
            document.getElementById('loading').style.display = 'block';
            document.getElementById('sendBtn').disabled = true;
            
            try {
                const model = document.getElementById('modelSelect').value;
                const resp = await fetch('/api/chat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({message, model})
                });
                
                const data = await resp.json();
                
                // Add assistant message
                addMessage(data.response, 'assistant');
                
                // Update activity log
                if (data.activity && data.activity.length > 0) {
                    document.getElementById('activityLog').innerHTML = data.activity.join('<br>');
                }
                
                // Update stats
                checkOllama();
                
            } catch (e) {
                addMessage('Error: ' + e.message, 'assistant');
            }
            
            document.getElementById('loading').style.display = 'none';
            document.getElementById('sendBtn').disabled = false;
        }
        
        function addMessage(text, type) {
            const container = document.getElementById('chatMessages');
            const div = document.createElement('div');
            div.className = `message ${type}-message`;
            
            if (type === 'user') {
                div.innerHTML = `<strong>👤 You:</strong> ${escapeHtml(text)}`;
            } else {
                div.innerHTML = `<strong>🤖 Assistant:</strong> ${formatMarkdown(text)}`;
            }
            
            container.appendChild(div);
            container.scrollTop = container.scrollHeight;
        }
        
        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
        
        function formatMarkdown(text) {
            // Basic markdown formatting
            text = escapeHtml(text);
            text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
            text = text.replace(/`([^`]+)`/g, '<code>$1</code>');
            text = text.replace(/\\n/g, '<br>');
            return text;
        }
        
        function clearChat() {
            document.getElementById('chatMessages').innerHTML = `
                <div class="message assistant-message">
                    <strong>🤖 Assistant:</strong> Chat cleared. How can I help you?
                </div>
            `;
            fetch('/api/clear', {method: 'POST'});
        }
        
        // Init
        checkOllama();
        setInterval(checkOllama, 30000);
    </script>
</body>
</html>
'''


def check_ollama():
    """Check if Ollama is running."""
    try:
        import httpx
        response = httpx.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False


def get_graph_instance(model_name: str):
    """Get or create graph instance."""
    global _graph, _config
    
    if _graph is None:
        _config = AgentConfig(
            default_model=ModelConfig(
                provider="ollama",
                model_name=model_name,
                temperature=0.7,
            ),
            enable_memory=True,
            memory_persist_path="./flask_agent_memory",
        )
        _graph = create_graph(config=_config, user_id=_user_id)
    
    return _graph


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/status')
def status():
    stats = get_memory_stats() or {}
    return jsonify({
        'ollama_running': check_ollama(),
        'stats': stats,
    })


@app.route('/api/chat', methods=['POST'])
def chat():
    global _chat_history
    
    data = request.json
    message = data.get('message', '')
    model = data.get('model', 'llama3.2:latest')
    
    if not message:
        return jsonify({'response': 'Please enter a message', 'activity': []})
    
    if not check_ollama():
        return jsonify({
            'response': '❌ Ollama is not running! Start it with: ollama serve',
            'activity': []
        })
    
    try:
        graph = get_graph_instance(model)
        
        activity = []
        final_response = None
        session_id = f"flask-{datetime.now().strftime('%H%M%S')}"
        
        for node_name, update in run_graph_stream(
            graph, message, thread_id=session_id, max_iterations=8
        ):
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            if node_name == "supervisor":
                msgs = update.get("messages")
                if isinstance(msgs, Message):
                    activity.append(f"[{timestamp}] 📋 SUPERVISOR routing...")
            elif node_name in ["research", "code", "analysis", "writer"]:
                icons = {"research": "🔍", "code": "💻", "analysis": "📊", "writer": "✍️"}
                activity.append(f"[{timestamp}] {icons[node_name]} {node_name.upper()} working...")
            elif node_name == "compile_response":
                final_response = update.get("final_response", "")
                activity.append(f"[{timestamp}] ✅ Done")
        
        response = final_response or "No response generated."
        _chat_history.append({'role': 'user', 'content': message})
        _chat_history.append({'role': 'assistant', 'content': response})
        
        return jsonify({'response': response, 'activity': activity})
    
    except Exception as e:
        return jsonify({'response': f'Error: {str(e)}', 'activity': []})


@app.route('/api/clear', methods=['POST'])
def clear():
    global _chat_history
    _chat_history = []
    return jsonify({'status': 'ok'})


def main():
    parser = argparse.ArgumentParser(description="Multi-Agent Flask UI")
    parser.add_argument("--port", "-p", type=int, default=5000, help="Port (default: 5000)")
    parser.add_argument("--debug", "-d", action="store_true", help="Debug mode")
    
    args = parser.parse_args()
    
    print("🚀 Starting Multi-Agent Flask UI...")
    print(f"   URL: http://localhost:{args.port}")
    
    if not check_ollama():
        print("⚠️  Ollama not running. Start with: ollama serve")
    
    app.run(host='0.0.0.0', port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
