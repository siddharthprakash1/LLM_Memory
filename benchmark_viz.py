"""
Benchmark Visualization UI.

A specialized dashboard to visualize the memory system's internal state
during benchmark runs. Shows real-time:
- Conversation flow
- Fact extraction stream
- Knowledge Graph growth
- Temporal state updates
- Reasoning chains
"""

import json
import os
import time
from flask import Flask, render_template_string, request, jsonify
from threading import Thread
import webbrowser
import requests

from llm_memory.memory_v4.memory_store import MemoryStoreV4
from llm_memory.memory_v4.retrieval import create_retriever

# Initialize Flask app
app = Flask(__name__)

# Global state
MEMORY = None
MEMORY_PATH = "./benchmark_viz_memory"
CURRENT_CONVERSATION = []
REASONING_LOGS = []

def get_memory():
    global MEMORY
    if MEMORY is None:
        MEMORY = MemoryStoreV4(
            user_id="benchmark_viz",
            persist_path=MEMORY_PATH,
            model_name="qwen2.5:32b"
        )
    return MEMORY

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Memory V4 Benchmark Visualizer</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <script src="https://unpkg.com/lucide@latest"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Inter:wght@300;400;500;600&display=swap');
        body { font-family: 'Inter', sans-serif; }
        .mono { font-family: 'JetBrains Mono', monospace; }
        .scrollbar-hide::-webkit-scrollbar { display: none; }
        .animate-pulse-slow { animation: pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
        .fade-in { animation: fadeIn 0.3s ease-out forwards; }
    </style>
</head>
<body class="bg-gray-900 text-gray-100 h-screen flex overflow-hidden">

    <!-- Left Panel: Conversation Stream -->
    <div class="w-1/3 border-r border-gray-800 flex flex-col bg-gray-900">
        <div class="p-4 border-b border-gray-800 bg-gray-800/50">
            <h2 class="text-sm font-bold text-gray-400 uppercase tracking-wider flex items-center gap-2">
                <i data-lucide="message-square" class="w-4 h-4 text-blue-400"></i>
                Conversation Stream
            </h2>
        </div>
        <div id="conversation-log" class="flex-1 overflow-y-auto p-4 space-y-4">
            <!-- Messages injected here -->
        </div>
        <div class="p-4 border-t border-gray-800 bg-gray-800/30">
            <div class="flex gap-2">
                <button onclick="startBenchmark()" class="flex-1 bg-blue-600 hover:bg-blue-500 text-white px-4 py-2 rounded-md text-sm font-medium transition-colors flex items-center justify-center gap-2">
                    <i data-lucide="play" class="w-4 h-4"></i> Start Benchmark
                </button>
                <button onclick="resetSystem()" class="bg-red-900/30 hover:bg-red-900/50 text-red-400 px-4 py-2 rounded-md text-sm font-medium transition-colors">
                    <i data-lucide="rotate-ccw" class="w-4 h-4"></i>
                </button>
            </div>
        </div>
    </div>

    <!-- Middle Panel: The Brain (Graph & Facts) -->
    <div class="w-1/3 border-r border-gray-800 flex flex-col bg-gray-900">
        <div class="h-1/2 flex flex-col border-b border-gray-800">
            <div class="p-4 border-b border-gray-800 bg-gray-800/50 flex justify-between items-center">
                <h2 class="text-sm font-bold text-gray-400 uppercase tracking-wider flex items-center gap-2">
                    <i data-lucide="network" class="w-4 h-4 text-purple-400"></i>
                    Knowledge Graph
                </h2>
                <span class="text-xs text-gray-500" id="node-count">0 Nodes</span>
            </div>
            <div id="network-container" class="flex-1 bg-gray-900 relative">
                <!-- Graph rendered here -->
                <div class="absolute inset-0 flex items-center justify-center text-gray-700 pointer-events-none">
                    Waiting for data...
                </div>
            </div>
        </div>
        
        <div class="h-1/2 flex flex-col">
            <div class="p-4 border-b border-gray-800 bg-gray-800/50">
                <h2 class="text-sm font-bold text-gray-400 uppercase tracking-wider flex items-center gap-2">
                    <i data-lucide="database" class="w-4 h-4 text-green-400"></i>
                    Extracted Facts (Live)
                </h2>
            </div>
            <div id="facts-log" class="flex-1 overflow-y-auto p-4 space-y-2">
                <!-- Facts injected here -->
            </div>
        </div>
    </div>

    <!-- Right Panel: Temporal & Reasoning -->
    <div class="w-1/3 flex flex-col bg-gray-900">
        <div class="h-1/2 flex flex-col border-b border-gray-800">
            <div class="p-4 border-b border-gray-800 bg-gray-800/50">
                <h2 class="text-sm font-bold text-gray-400 uppercase tracking-wider flex items-center gap-2">
                    <i data-lucide="clock" class="w-4 h-4 text-orange-400"></i>
                    Temporal State
                </h2>
            </div>
            <div id="temporal-log" class="flex-1 overflow-y-auto p-4 space-y-3">
                <!-- Temporal states injected here -->
            </div>
        </div>

        <div class="h-1/2 flex flex-col">
            <div class="p-4 border-b border-gray-800 bg-gray-800/50">
                <h2 class="text-sm font-bold text-gray-400 uppercase tracking-wider flex items-center gap-2">
                    <i data-lucide="cpu" class="w-4 h-4 text-pink-400"></i>
                    Reasoning & Stats
                </h2>
            </div>
            <div class="flex-1 p-4 space-y-6 overflow-y-auto">
                <div class="grid grid-cols-2 gap-4">
                    <div class="bg-gray-800 p-3 rounded-lg border border-gray-700">
                        <div class="text-gray-500 text-xs">Facts</div>
                        <div class="text-2xl font-bold text-white" id="stat-facts">0</div>
                    </div>
                    <div class="bg-gray-800 p-3 rounded-lg border border-gray-700">
                        <div class="text-gray-500 text-xs">Episodes</div>
                        <div class="text-2xl font-bold text-white" id="stat-episodes">0</div>
                    </div>
                </div>
                
                <div id="reasoning-log" class="space-y-3 text-xs mono text-gray-400">
                    <!-- Logs -->
                </div>
            </div>
        </div>
    </div>

    <script>
        lucide.createIcons();
        let network = null;
        let nodes = new vis.DataSet();
        let edges = new vis.DataSet();

        function initGraph() {
            const container = document.getElementById('network-container');
            const data = { nodes: nodes, edges: edges };
            const options = {
                nodes: {
                    shape: 'dot',
                    size: 10,
                    font: { size: 12, color: '#9ca3af', face: 'Inter' },
                    borderWidth: 0,
                    shadow: true
                },
                edges: {
                    width: 1,
                    color: { color: '#4b5563', highlight: '#60a5fa' },
                    arrows: { to: { enabled: true, scaleFactor: 0.5 } },
                    smooth: { type: 'continuous' }
                },
                physics: {
                    stabilization: false,
                    barnesHut: {
                        gravitationalConstant: -2000,
                        springConstant: 0.04,
                        springLength: 95
                    }
                },
                interaction: { hover: true, tooltipDelay: 200 }
            };
            network = new vis.Network(container, data, options);
        }

        async function startBenchmark() {
            await fetch('/api/start_benchmark', { method: 'POST' });
        }

        async function resetSystem() {
            await fetch('/api/reset', { method: 'POST' });
            nodes.clear();
            edges.clear();
            document.getElementById('conversation-log').innerHTML = '';
            document.getElementById('facts-log').innerHTML = '';
            document.getElementById('temporal-log').innerHTML = '';
            document.getElementById('reasoning-log').innerHTML = '';
            updateStats({total_facts: 0, total_episodes: 0});
        }

        function updateGraph(facts) {
            const newNodes = [];
            const newEdges = [];
            const existingNodeIds = new Set(nodes.getIds());
            const existingEdgeIds = new Set(edges.getIds());

            facts.forEach(f => {
                // Nodes
                [f.subject, f.object].forEach(label => {
                    if (!existingNodeIds.has(label)) {
                        let color = '#60a5fa'; // Blue for entities
                        if (f.fact_type === 'preference') color = '#f472b6'; // Pink
                        if (f.fact_type === 'temporal') color = '#fb923c'; // Orange
                        
                        try {
                            nodes.add({ id: label, label: label.length > 15 ? label.substring(0,12)+'...' : label, title: label, color: color });
                            existingNodeIds.add(label);
                        } catch(e) {}
                    }
                });

                // Edge
                const edgeId = `${f.subject}-${f.predicate}-${f.object}`;
                if (!existingEdgeIds.has(edgeId)) {
                    try {
                        edges.add({ id: edgeId, from: f.subject, to: f.object, label: f.predicate, font: { size: 8, align: 'middle', color: '#6b7280' } });
                        existingEdgeIds.add(edgeId);
                    } catch(e) {}
                }
            });
            
            document.getElementById('node-count').innerText = `${nodes.length} Nodes`;
        }

        function addMessage(msg) {
            const div = document.createElement('div');
            div.className = "bg-gray-800 rounded-lg p-3 border border-gray-700 fade-in";
            div.innerHTML = `
                <div class="flex justify-between items-start mb-1">
                    <span class="text-xs font-bold text-${msg.speaker === 'Caroline' ? 'blue' : 'purple'}-400">${msg.speaker}</span>
                    <span class="text-[10px] text-gray-500">${msg.date}</span>
                </div>
                <p class="text-sm text-gray-300">${msg.text}</p>
            `;
            const container = document.getElementById('conversation-log');
            container.appendChild(div);
            container.scrollTop = container.scrollHeight;
        }

        function addFacts(facts) {
            const container = document.getElementById('facts-log');
            facts.forEach(f => {
                const div = document.createElement('div');
                div.className = "bg-gray-800/50 p-2 rounded border border-gray-700/50 text-xs flex gap-2 items-center fade-in";
                
                let icon = 'circle';
                let color = 'text-gray-400';
                if (f.fact_type === 'preference') { icon = 'heart'; color = 'text-pink-400'; }
                if (f.fact_type === 'state_change') { icon = 'arrow-right'; color = 'text-green-400'; }
                
                div.innerHTML = `
                    <i data-lucide="${icon}" class="w-3 h-3 ${color} flex-shrink-0"></i>
                    <div class="flex-1">
                        <span class="text-gray-300 font-medium">${f.subject}</span>
                        <span class="text-gray-500">${f.predicate}</span>
                        <span class="text-gray-300 font-medium">${f.object}</span>
                    </div>
                `;
                container.prepend(div); // Newest on top
            });
            lucide.createIcons();
        }

        function updateTemporal(states) {
            const container = document.getElementById('temporal-log');
            container.innerHTML = ''; // Re-render all to sort
            
            states.forEach(s => {
                const div = document.createElement('div');
                div.className = "bg-gray-800 p-3 rounded-lg border-l-2 border-orange-500";
                div.innerHTML = `
                    <div class="flex justify-between">
                        <span class="text-xs font-bold text-gray-300">${s.subject}</span>
                        <span class="text-xs text-orange-400 font-mono">${s.duration_text || 'Ongoing'}</span>
                    </div>
                    <p class="text-xs text-gray-400 mt-1">${s.description}</p>
                `;
                container.appendChild(div);
            });
        }

        function updateStats(stats) {
            document.getElementById('stat-facts').innerText = stats.total_facts;
            document.getElementById('stat-episodes').innerText = stats.total_episodes;
        }

        function updateReasoning(logs) {
            const container = document.getElementById('reasoning-log');
            container.innerHTML = '';
            
            logs.forEach(log => {
                const div = document.createElement('div');
                div.className = "bg-gray-800 p-2 rounded border border-gray-700/50 fade-in";
                
                let content = `<div class="text-pink-400 font-bold mb-1">${log.type}</div>`;
                
                if (log.type === 'DECOMPOSE') {
                    content += `<div class="pl-2 border-l border-pink-500/30">`;
                    log.steps.forEach((step, i) => {
                        content += `<div class="text-gray-300">Step ${i+1}: ${step}</div>`;
                    });
                    content += `</div>`;
                } else if (log.type === 'RETRIEVE') {
                    content += `<div class="text-gray-400">Query: "${log.query}"</div>`;
                    content += `<div class="text-green-400 mt-1">Found ${log.count} facts</div>`;
                } else {
                    content += `<div class="text-gray-300">${log.message}</div>`;
                }
                
                div.innerHTML = content;
                container.prepend(div);
            });
        }

        // Poll for updates
        setInterval(async () => {
            try {
                const res = await fetch('/api/state');
                const data = await res.json();
                
                // Update conversation if new
                if (data.last_message && (!window.lastMsgId || window.lastMsgId !== data.last_message.id)) {
                    addMessage(data.last_message);
                    window.lastMsgId = data.last_message.id;
                    
                    // If new message, update everything else
                    updateGraph(data.facts);
                    addFacts(data.new_facts);
                    updateTemporal(data.temporal);
                    updateStats(data.stats);
                }
                
                // Update reasoning logs if any
                if (data.reasoning_logs && data.reasoning_logs.length > 0) {
                    updateReasoning(data.reasoning_logs);
                }
            } catch(e) {}
        }, 1000);

        initGraph();
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/state')
def get_state():
    """Get current system state for polling."""
    memory = get_memory()
    
    # Get latest message
    last_msg = CURRENT_CONVERSATION[-1] if CURRENT_CONVERSATION else None
    
    # Get all facts
    all_facts = list(memory.facts.values())
    current_facts = [f.to_dict() for f in all_facts if f.is_current]
    
    # Get recent facts (last 5)
    sorted_facts = sorted(all_facts, key=lambda x: x.extraction_time, reverse=True)
    new_facts = [f.to_dict() for f in sorted_facts[:5]]
    
    # Get temporal states
    temporal = [
        {
            "subject": s.subject,
            "description": s.description,
            "duration_text": s.calculate_duration_from_reference()
        }
        for s in memory.temporal_states.values()
    ]
    
    # Get reasoning logs (simulated for now, would need real logs from memory)
    # In a real implementation, we'd pull this from a log buffer in memory_store
    reasoning_logs = getattr(memory, 'reasoning_logs', [])
    
    return jsonify({
        "last_message": last_msg,
        "facts": current_facts,
        "new_facts": new_facts,
        "temporal": temporal,
        "stats": memory.stats(),
        "reasoning_logs": reasoning_logs
    })

@app.route('/api/reset', methods=['POST'])
def reset():
    global CURRENT_CONVERSATION
    memory = get_memory()
    memory.clear()
    CURRENT_CONVERSATION = []
    return jsonify({"status": "cleared"})

@app.route('/api/start_benchmark', methods=['POST'])
def start_benchmark_endpoint():
    """Start the benchmark feeder in a background thread."""
    Thread(target=run_benchmark_feeder).start()
    return jsonify({"status": "started"})

def run_benchmark_feeder():
    """Feeds benchmark data into the system."""
    import json
    
    # Load data
    data_path = "benchmarks/locomo_data/data/locomo10.json"
    if not os.path.exists(data_path):
        print("Data not found")
        return
        
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    # Use first conversation
    conv = data[0]['conversation']
    
    # Extract turns (simplified logic from benchmark script)
    turns = []
    if isinstance(conv, dict):
        # Sort session keys
        keys = sorted([k for k in conv.keys() if k.startswith('session_') and not k.endswith('_time')], 
                     key=lambda x: int(x.split('_')[1]))
        for k in keys:
            date = conv.get(f"{k}_date_time", "Unknown Date")
            for t in conv[k]:
                turns.append({
                    "speaker": t['speaker'],
                    "text": t['text'],
                    "date": date
                })
    
    # Feed turns
    memory = get_memory()
    
    for i, turn in enumerate(turns):
        # 1. Update UI Conversation Log
        msg_obj = {
            "id": i,
            "speaker": turn['speaker'],
            "text": turn['text'],
            "date": turn['date']
        }
        CURRENT_CONVERSATION.append(msg_obj)
        
        # 2. Process in Memory (The Heavy Lifting)
        memory.add_conversation_turn(
            speaker=turn['speaker'],
            text=turn['text'],
            date=turn['date']
        )
        
        # Simulate typing/processing delay for visual effect
        time.sleep(2.5)

def open_browser():
    time.sleep(1.5)
    webbrowser.open('http://127.0.0.1:5001')

def main():
    print("Starting Benchmark Visualizer on port 5001...")
    Thread(target=open_browser).start()
    app.run(host='0.0.0.0', port=5001, debug=False)

if __name__ == "__main__":
    main()
