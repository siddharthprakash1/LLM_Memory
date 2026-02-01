"""
Web UI for Memory Agent V4.

A modern, "Claude-like" interface for interacting with the memory agent.
Features:
- Chat interface
- Real-time memory visualization (Facts, Graph, Timeline)
- System statistics
"""

import json
import os
from datetime import datetime
from flask import Flask, render_template_string, request, jsonify, send_from_directory
from threading import Thread
import webbrowser
import time

from llm_memory.agents_v4.graph import MemoryAgent
from llm_memory.memory_v4.memory_store import MemoryStoreV4

# Initialize Flask app
app = Flask(__name__)

# Global agent instance
AGENT = None
MEMORY_PATH = "./agent_memory_v4"

def get_agent():
    global AGENT
    if AGENT is None:
        AGENT = MemoryAgent(
            model_name="qwen2.5:32b",
            memory_path=MEMORY_PATH
        )
    return AGENT

# HTML Template (Single file for simplicity, but structured)
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLM Memory V4 - Cognitive Agent</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <script src="https://unpkg.com/lucide@latest"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
        body { font-family: 'Inter', sans-serif; }
        .scrollbar-hide::-webkit-scrollbar { display: none; }
        .scrollbar-hide { -ms-overflow-style: none; scrollbar-width: none; }
        .typing-dot { animation: typing 1.4s infinite ease-in-out both; }
        @keyframes typing {
            0%, 100% { transform: scale(0.2); opacity: 0.2; }
            50% { transform: scale(1); opacity: 1; }
        }
        .memory-card { transition: all 0.2s; }
        .memory-card:hover { transform: translateY(-2px); box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); }
    </style>
</head>
<body class="bg-gray-50 h-screen flex overflow-hidden">

    <!-- Sidebar -->
    <div class="w-64 bg-white border-r border-gray-200 flex flex-col">
        <div class="p-4 border-b border-gray-200">
            <h1 class="text-xl font-semibold text-gray-800 flex items-center gap-2">
                <i data-lucide="brain" class="w-6 h-6 text-indigo-600"></i>
                Memory V4
            </h1>
            <p class="text-xs text-gray-500 mt-1">Cognitive Architecture</p>
        </div>
        
        <div class="flex-1 overflow-y-auto p-4 space-y-6">
            <!-- Stats -->
            <div>
                <h3 class="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">System Stats</h3>
                <div class="space-y-3" id="stats-container">
                    <div class="bg-indigo-50 p-3 rounded-lg">
                        <div class="text-indigo-600 text-xs font-medium">Total Facts</div>
                        <div class="text-2xl font-bold text-indigo-900" id="stat-facts">-</div>
                    </div>
                    <div class="bg-purple-50 p-3 rounded-lg">
                        <div class="text-purple-600 text-xs font-medium">Episodes</div>
                        <div class="text-2xl font-bold text-purple-900" id="stat-episodes">-</div>
                    </div>
                    <div class="bg-blue-50 p-3 rounded-lg">
                        <div class="text-blue-600 text-xs font-medium">Temporal States</div>
                        <div class="text-2xl font-bold text-blue-900" id="stat-temporal">-</div>
                    </div>
                </div>
            </div>
            
            <!-- Controls -->
            <div>
                <h3 class="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">Controls</h3>
                <button onclick="clearMemory()" class="w-full flex items-center gap-2 px-3 py-2 text-sm text-red-600 bg-red-50 hover:bg-red-100 rounded-md transition-colors">
                    <i data-lucide="trash-2" class="w-4 h-4"></i>
                    Reset Memory
                </button>
            </div>
        </div>
        
        <div class="p-4 border-t border-gray-200 text-xs text-gray-400">
            Powered by Ollama & Qwen 2.5
        </div>
    </div>

    <!-- Main Chat Area -->
    <div class="flex-1 flex flex-col min-w-0">
        <!-- Chat Header -->
        <div class="h-16 bg-white border-b border-gray-200 flex items-center justify-between px-6">
            <h2 class="font-medium text-gray-700">Agent Session</h2>
            <div class="flex items-center gap-2">
                <span class="w-2 h-2 bg-green-500 rounded-full"></span>
                <span class="text-sm text-gray-500">Online</span>
            </div>
        </div>

        <!-- Messages -->
        <div id="chat-messages" class="flex-1 overflow-y-auto p-6 space-y-6">
            <!-- Welcome Message -->
            <div class="flex gap-4">
                <div class="w-8 h-8 bg-indigo-100 rounded-full flex items-center justify-center flex-shrink-0">
                    <i data-lucide="bot" class="w-5 h-5 text-indigo-600"></i>
                </div>
                <div class="bg-white border border-gray-200 rounded-2xl rounded-tl-none p-4 shadow-sm max-w-2xl">
                    <p class="text-gray-800">Hello! I'm your memory-augmented assistant. I can remember facts, events, and understand time. How can I help you today?</p>
                </div>
            </div>
        </div>

        <!-- Input Area -->
        <div class="p-6 bg-white border-t border-gray-200">
            <form id="chat-form" class="flex gap-4 max-w-4xl mx-auto" onsubmit="sendMessage(event)">
                <input type="text" id="user-input" 
                    class="flex-1 border border-gray-300 rounded-xl px-4 py-3 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent shadow-sm"
                    placeholder="Type a message... (e.g., 'I moved to NYC 3 years ago')"
                    autocomplete="off">
                <button type="submit" 
                    class="bg-indigo-600 text-white px-6 py-3 rounded-xl hover:bg-indigo-700 transition-colors font-medium flex items-center gap-2 shadow-sm">
                    <span>Send</span>
                    <i data-lucide="send" class="w-4 h-4"></i>
                </button>
            </form>
        </div>
    </div>

    <!-- Right Panel: The "Brain" -->
    <div class="w-96 bg-white border-l border-gray-200 flex flex-col shadow-xl z-10">
        <!-- Tabs -->
        <div class="flex border-b border-gray-200">
            <button onclick="switchTab('facts')" id="tab-facts" class="flex-1 py-3 text-sm font-medium text-indigo-600 border-b-2 border-indigo-600">Facts</button>
            <button onclick="switchTab('graph')" id="tab-graph" class="flex-1 py-3 text-sm font-medium text-gray-500 hover:text-gray-700">Graph</button>
            <button onclick="switchTab('timeline')" id="tab-timeline" class="flex-1 py-3 text-sm font-medium text-gray-500 hover:text-gray-700">Timeline</button>
        </div>

        <!-- Content -->
        <div class="flex-1 overflow-hidden relative bg-gray-50">
            
            <!-- Facts View -->
            <div id="view-facts" class="absolute inset-0 overflow-y-auto p-4 space-y-3">
                <!-- Facts will be injected here -->
                <div class="text-center text-gray-400 mt-10">
                    <i data-lucide="database" class="w-8 h-8 mx-auto mb-2 opacity-50"></i>
                    <p class="text-sm">No facts stored yet.</p>
                </div>
            </div>

            <!-- Graph View -->
            <div id="view-graph" class="absolute inset-0 hidden">
                <div id="network-container" class="w-full h-full"></div>
            </div>

            <!-- Timeline View -->
            <div id="view-timeline" class="absolute inset-0 overflow-y-auto p-4 hidden">
                <div class="border-l-2 border-indigo-200 ml-3 pl-6 space-y-6" id="timeline-container">
                    <!-- Timeline items -->
                </div>
            </div>
        </div>
    </div>

    <script>
        // Icons
        lucide.createIcons();

        // State
        let currentTab = 'facts';

        function switchTab(tab) {
            currentTab = tab;
            
            // Update tabs
            ['facts', 'graph', 'timeline'].forEach(t => {
                const btn = document.getElementById(`tab-${t}`);
                const view = document.getElementById(`view-${t}`);
                
                if (t === tab) {
                    btn.className = "flex-1 py-3 text-sm font-medium text-indigo-600 border-b-2 border-indigo-600";
                    view.classList.remove('hidden');
                } else {
                    btn.className = "flex-1 py-3 text-sm font-medium text-gray-500 hover:text-gray-700";
                    view.classList.add('hidden');
                }
            });

            if (tab === 'graph') {
                updateGraph();
            }
        }

        async function sendMessage(e) {
            e.preventDefault();
            const input = document.getElementById('user-input');
            const message = input.value.trim();
            if (!message) return;

            // Add user message
            addMessage(message, 'user');
            input.value = '';

            // Add loading indicator
            const loadingId = addLoading();

            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message })
                });
                const data = await response.json();
                
                removeLoading(loadingId);
                addMessage(data.response, 'agent');
                
                // Refresh memory view
                fetchMemory();
                
            } catch (error) {
                removeLoading(loadingId);
                addMessage("Sorry, something went wrong.", 'agent');
                console.error(error);
            }
        }

        function addMessage(text, sender) {
            const container = document.getElementById('chat-messages');
            const div = document.createElement('div');
            div.className = "flex gap-4 " + (sender === 'user' ? "flex-row-reverse" : "");
            
            const icon = sender === 'user' 
                ? '<div class="w-8 h-8 bg-gray-200 rounded-full flex items-center justify-center flex-shrink-0"><i data-lucide="user" class="w-5 h-5 text-gray-600"></i></div>'
                : '<div class="w-8 h-8 bg-indigo-100 rounded-full flex items-center justify-center flex-shrink-0"><i data-lucide="bot" class="w-5 h-5 text-indigo-600"></i></div>';
            
            const bubble = sender === 'user'
                ? '<div class="bg-indigo-600 text-white rounded-2xl rounded-tr-none p-4 shadow-sm max-w-2xl"><p>' + text + '</p></div>'
                : '<div class="bg-white border border-gray-200 rounded-2xl rounded-tl-none p-4 shadow-sm max-w-2xl"><p class="text-gray-800">' + text + '</p></div>';

            div.innerHTML = sender === 'user' ? bubble + icon : icon + bubble;
            container.appendChild(div);
            lucide.createIcons();
            container.scrollTop = container.scrollHeight;
        }

        function addLoading() {
            const container = document.getElementById('chat-messages');
            const id = 'loading-' + Date.now();
            const div = document.createElement('div');
            div.id = id;
            div.className = "flex gap-4";
            div.innerHTML = `
                <div class="w-8 h-8 bg-indigo-100 rounded-full flex items-center justify-center flex-shrink-0">
                    <i data-lucide="bot" class="w-5 h-5 text-indigo-600"></i>
                </div>
                <div class="bg-white border border-gray-200 rounded-2xl rounded-tl-none p-4 shadow-sm">
                    <div class="flex gap-1">
                        <div class="w-2 h-2 bg-gray-400 rounded-full typing-dot"></div>
                        <div class="w-2 h-2 bg-gray-400 rounded-full typing-dot" style="animation-delay: 0.2s"></div>
                        <div class="w-2 h-2 bg-gray-400 rounded-full typing-dot" style="animation-delay: 0.4s"></div>
                    </div>
                </div>
            `;
            container.appendChild(div);
            lucide.createIcons();
            container.scrollTop = container.scrollHeight;
            return id;
        }

        function removeLoading(id) {
            const el = document.getElementById(id);
            if (el) el.remove();
        }

        async function fetchMemory() {
            try {
                const response = await fetch('/api/memory');
                const data = await response.json();
                
                updateStats(data.stats);
                updateFacts(data.facts);
                updateTimeline(data.temporal);
                if (currentTab === 'graph') updateGraph(data.facts);
                
            } catch (error) {
                console.error("Failed to fetch memory:", error);
            }
        }

        function updateStats(stats) {
            document.getElementById('stat-facts').textContent = stats.total_facts || 0;
            document.getElementById('stat-episodes').textContent = stats.total_episodes || 0;
            document.getElementById('stat-temporal').textContent = stats.temporal_states || 0;
        }

        function updateFacts(facts) {
            const container = document.getElementById('view-facts');
            if (!facts || facts.length === 0) {
                container.innerHTML = `
                    <div class="text-center text-gray-400 mt-10">
                        <i data-lucide="database" class="w-8 h-8 mx-auto mb-2 opacity-50"></i>
                        <p class="text-sm">No facts stored yet.</p>
                    </div>`;
                lucide.createIcons();
                return;
            }

            container.innerHTML = facts.map(f => `
                <div class="bg-white p-3 rounded-lg border border-gray-200 memory-card">
                    <div class="flex items-start justify-between mb-1">
                        <span class="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-indigo-100 text-indigo-800">
                            ${f.fact_type}
                        </span>
                        <span class="text-xs text-gray-400">${f.source_date || 'Unknown date'}</span>
                    </div>
                    <p class="text-sm text-gray-800 font-medium">${f.subject} ${f.predicate} ${f.object}</p>
                    ${f.duration ? `<p class="text-xs text-indigo-500 mt-1">Duration: ${f.duration}</p>` : ''}
                </div>
            `).join('');
            lucide.createIcons();
        }

        function updateTimeline(states) {
            const container = document.getElementById('timeline-container');
            if (!states || states.length === 0) {
                container.innerHTML = '<p class="text-sm text-gray-400">No timeline data.</p>';
                return;
            }

            // Sort by start date (if available)
            // For simplicity, just listing them
            container.innerHTML = states.map(s => `
                <div class="relative">
                    <div class="absolute -left-8 top-1 w-4 h-4 rounded-full bg-indigo-200 border-2 border-white"></div>
                    <div class="bg-white p-3 rounded-lg border border-gray-200 shadow-sm">
                        <p class="text-sm font-medium text-gray-800">${s.description}</p>
                        <p class="text-xs text-gray-500 mt-1">
                            <span class="font-semibold text-indigo-600">${s.duration_text || 'Ongoing'}</span>
                            • ${s.subject}
                        </p>
                    </div>
                </div>
            `).join('');
        }

        let network = null;
        async function updateGraph(factsData) {
            if (!factsData) {
                const response = await fetch('/api/memory');
                const data = await response.json();
                factsData = data.facts;
            }

            const container = document.getElementById('network-container');
            const nodes = new vis.DataSet();
            const edges = new vis.DataSet();
            const nodeSet = new Set();

            factsData.forEach(f => {
                // Subject Node
                if (!nodeSet.has(f.subject)) {
                    nodes.add({ id: f.subject, label: f.subject, group: 'subject', color: '#E0E7FF' });
                    nodeSet.add(f.subject);
                }
                
                // Object Node
                if (!nodeSet.has(f.object)) {
                    nodes.add({ id: f.object, label: f.object, group: 'object', color: '#F3F4F6' });
                    nodeSet.add(f.object);
                }

                // Edge
                edges.add({ from: f.subject, to: f.object, label: f.predicate, arrows: 'to', font: { size: 10, align: 'middle' } });
            });

            const data = { nodes, edges };
            const options = {
                nodes: { shape: 'box', font: { face: 'Inter' } },
                physics: { stabilization: true }
            };

            if (network) {
                network.setData(data);
            } else {
                network = new vis.Network(container, data, options);
            }
        }

        async function clearMemory() {
            if (!confirm("Are you sure? This will delete all memories.")) return;
            await fetch('/api/clear', { method: 'POST' });
            fetchMemory();
            document.getElementById('chat-messages').innerHTML = '';
            addMessage("Memory cleared.", 'agent');
        }

        // Initial load
        fetchMemory();
        lucide.createIcons();
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    message = data.get('message')
    if not message:
        return jsonify({"error": "No message"}), 400
    
    agent = get_agent()
    
    # Run chat with the agent's chat method (which handles state)
    # But we need to capture the output.
    # The agent.chat method prints to stdout. We should modify agent.chat or use graph directly.
    # Let's use graph directly but ensure we pass the thread_id for persistence.
    
    from langchain_core.messages import HumanMessage
    
    # Use a fixed thread_id for the UI session
    # In a real app, this would come from the client/cookie
    thread_id = "web_session_1"
    
    # Get current state to check if we have history
    # This ensures we don't overwrite the state, but append to it
    config = {"configurable": {"thread_id": thread_id}}
    
    inputs = {
        "messages": [HumanMessage(content=message)],
        "user_id": "default",
        # Don't overwrite memory_context here
    }
    
    # Run graph with config
    response_text = ""
    for event in agent.graph.stream(inputs, config=config):
        for key, value in event.items():
            if key == "agent":
                response_text = value["messages"][0].content
    
    return jsonify({"response": response_text})

@app.route('/api/memory', methods=['GET'])
def get_memory():
    agent = get_agent()
    stats = agent.memory.stats()
    
    # Get all facts
    facts = list(agent.memory.facts.values())
    facts_json = [f.to_dict() for f in facts if f.is_current]
    
    # Get temporal states
    temporal = list(agent.memory.temporal_states.values())
    temporal_json = [
        {
            "description": t.description,
            "subject": t.subject,
            "duration_text": t.calculate_duration_from_reference()
        }
        for t in temporal
    ]
    
    return jsonify({
        "stats": stats,
        "facts": facts_json,
        "temporal": temporal_json
    })

@app.route('/api/clear', methods=['POST'])
def clear_memory():
    agent = get_agent()
    agent.memory.clear()
    return jsonify({"status": "cleared"})

def open_browser():
    time.sleep(1.5)
    webbrowser.open('http://127.0.0.1:5000')

def main():
    print("Starting Memory V4 Web UI...")
    Thread(target=open_browser).start()
    app.run(host='0.0.0.0', port=5000, debug=False)

if __name__ == "__main__":
    main()
