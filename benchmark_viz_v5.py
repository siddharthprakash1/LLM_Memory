"""
Benchmark Visualization UI (V5 Enhanced).

A specialized dashboard to visualize Memory V5's internal state
during benchmark runs. Features a graph-centric layout.
"""

import json
import os
import time
from flask import Flask, render_template_string, request, jsonify
from threading import Thread
import webbrowser

from llm_memory.memory_v5.memory_store_v5 import MemoryStoreV5
from llm_memory.memory_v5 import create_memory_v5

# Initialize Flask app
app = Flask(__name__)

# Global state
MEMORY = None
MEMORY_PATH = "./benchmark_viz_v5_memory"
CURRENT_CONVERSATION = []

def get_memory():
    global MEMORY
    if MEMORY is None:
        MEMORY = create_memory_v5(
            user_id="benchmark_viz_v5",
            persist_path=MEMORY_PATH,
            use_llm=False  # Revert to Rule-based to demonstrate IMPROVEMENTS
        )
    return MEMORY

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Memory V5 Graph Visualizer</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <script src="https://unpkg.com/lucide@latest"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Outfit:wght@300;400;500;600;700&display=swap');
        body { font-family: 'Outfit', sans-serif; }
        .mono { font-family: 'JetBrains Mono', monospace; }
        .scrollbar-hide::-webkit-scrollbar { display: none; }
        
        /* Glassmorphism */
        .glass {
            background: rgba(17, 24, 39, 0.7);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.05);
        }
        
        .fade-in { animation: fadeIn 0.4s ease-out forwards; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(5px); } to { opacity: 1; transform: translateY(0); } }
    </style>
</head>
<body class="bg-[#0B0F17] text-gray-100 h-screen overflow-hidden flex flex-col">

    <!-- Header -->
    <header class="h-14 border-b border-gray-800 glass flex items-center justify-between px-6 z-20 relative">
        <div class="flex items-center gap-3">
            <div class="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center font-bold text-white">V5</div>
            <h1 class="font-semibold text-lg tracking-tight text-white">Memory Graph <span class="text-gray-500 font-normal">Visualizer</span></h1>
        </div>
        <div class="flex gap-3">
            <div class="hidden md:flex items-center gap-6 mr-6 text-sm">
                <div class="flex items-center gap-2">
                    <span class="w-2 h-2 rounded-full bg-blue-400 shadow-[0_0_10px_rgba(96,165,250,0.5)]"></span>
                    <span class="text-gray-400">Person</span>
                </div>
                <div class="flex items-center gap-2">
                    <span class="w-2 h-2 rounded-full bg-orange-400 shadow-[0_0_10px_rgba(251,146,60,0.5)]"></span>
                    <span class="text-gray-400">Event</span>
                </div>
                <div class="flex items-center gap-2">
                    <span class="w-2 h-2 rounded-full bg-emerald-400 shadow-[0_0_10px_rgba(52,211,153,0.5)]"></span>
                    <span class="text-gray-400">Concept</span>
                </div>
            </div>
            
            <button onclick="resetSystem()" class="px-4 py-1.5 rounded-md bg-gray-800 hover:bg-gray-700 text-gray-300 text-sm border border-gray-700 transition-colors">
                Reset
            </button>
            <button onclick="startBenchmark()" class="px-4 py-1.5 rounded-md bg-blue-600 hover:bg-blue-500 text-white text-sm font-medium shadow-[0_0_15px_rgba(37,99,235,0.3)] transition-all">
                <i data-lucide="play" class="w-3 h-3 inline mr-1"></i> Run Benchmark
            </button>
        </div>
    </header>

    <!-- Main Workspace -->
    <div class="flex-1 flex overflow-hidden relative">
    
        <!-- Left: Conversation (Chat Stream) -->
        <div class="w-80 flex-shrink-0 flex flex-col border-r border-gray-800 glass z-10">
            <div class="p-3 border-b border-gray-800/50">
                <h2 class="text-xs font-bold text-gray-500 uppercase tracking-widest">Live Conversation</h2>
            </div>
            <div id="conversation-log" class="flex-1 overflow-y-auto p-4 space-y-4">
                <!-- Chat Bubbles -->
                <div class="text-center text-gray-600 text-sm mt-10 italic">Waiting...</div>
            </div>
        </div>

        <!-- Center: The Graph -->
        <div class="flex-1 relative bg-[#05080E] overflow-hidden">
            <!-- Canvas -->
            <div id="network-container" class="absolute inset-0"></div>
            
            <!-- Graph Overlay Stats -->
            <div class="absolute bottom-6 left-6 flex gap-4 pointer-events-none">
                <div class="glass px-4 py-2 rounded-lg flex flex-col">
                    <span class="text-[10px] text-gray-500 uppercase">Nodes</span>
                    <span class="text-xl font-bold font-mono text-white" id="stat-nodes">0</span>
                </div>
                <div class="glass px-4 py-2 rounded-lg flex flex-col">
                    <span class="text-[10px] text-gray-500 uppercase">Edges</span>
                    <span class="text-xl font-bold font-mono text-white" id="stat-edges">0</span>
                </div>
            </div>
        </div>

        <!-- Right: Metrics & Info -->
        <div class="w-80 flex-shrink-0 flex flex-col border-l border-gray-800 glass z-10">
            
            <!-- Top: Temporal State -->
            <div class="h-1/2 flex flex-col border-b border-gray-800">
                <div class="p-3 border-b border-gray-800/50 flex justify-between items-center">
                    <h2 class="text-xs font-bold text-gray-500 uppercase tracking-widest">Temporal State</h2>
                    <i data-lucide="clock" class="w-3 h-3 text-orange-400"></i>
                </div>
                <div id="temporal-log" class="flex-1 overflow-y-auto p-3 space-y-2">
                    <!-- Temporal Items -->
                </div>
            </div>

            <!-- Bottom: Facts Stream -->
            <div class="h-1/2 flex flex-col">
                <div class="p-3 border-b border-gray-800/50 flex justify-between items-center">
                    <h2 class="text-xs font-bold text-gray-500 uppercase tracking-widest">Extracted Facts</h2>
                    <span class="flex h-2 w-2 rounded-full bg-green-500 animate-pulse"></span>
                </div>
                <div id="facts-log" class="flex-1 overflow-y-auto p-3 space-y-2">
                    <!-- Facts -->
                </div>
            </div>
        </div>
    
    </div>

    <!-- Scripts -->
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
                    font: { size: 14, color: '#e5e7eb', face: 'Outfit', strokeWidth: 0 },
                    borderWidth: 0,
                    shadow: { enabled: true, color: 'rgba(0,0,0,0.5)', size: 10, x: 0, y: 0 },
                    scaling: { min: 10, max: 30 }
                },
                edges: {
                    width: 2,
                    color: { color: 'rgba(255, 255, 255, 0.5)', highlight: '#60a5fa' },
                    arrows: { to: { enabled: true, scaleFactor: 0.5 } },
                    smooth: { type: 'continuous', roundness: 0.5 }
                },
                physics: {
                    stabilization: false,
                    barnesHut: {
                        gravitationalConstant: -2000,
                        centralGravity: 0.3,
                        springLength: 150,
                        springConstant: 0.04,
                        damping: 0.09
                    },
                    solver: 'barnesHut'
                },
                interaction: { hover: true, zoomView: true }
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
            document.getElementById('conversation-log').innerHTML = '<div class="text-center text-gray-600 text-sm mt-10 italic">Ready</div>';
            document.getElementById('facts-log').innerHTML = '';
            document.getElementById('temporal-log').innerHTML = '';
            document.getElementById('stat-nodes').innerText = '0';
            document.getElementById('stat-edges').innerText = '0';
        }

        function getNodeColor(type) {
            // Neon Palette
            switch(type.toLowerCase()) {
                case 'person': return { background: '#3b82f6', border: '#60a5fa' }; // Blue
                case 'event': return { background: '#f97316', border: '#fb923c' }; // Orange
                case 'location': return { background: '#8b5cf6', border: '#a78bfa' }; // Purple
                case 'organization': return { background: '#ec4899', border: '#f472b6' }; // Pink
                case 'time': return { background: '#10b981', border: '#34d399' }; // Emerald
                default: return { background: '#4b5563', border: '#6b7280' }; // Gray
            }
        }

        function updateGraph(facts) {
            const existingNodeIds = new Set(nodes.getIds());
            const existingEdgeIds = new Set(edges.getIds());
            
            facts.forEach(f => {
                // Nodes
                [f.subject, f.object].forEach(label => {
                    if (!existingNodeIds.has(label)) {
                        const colors = getNodeColor(f.subject_type || 'concept');
                        try {
                            nodes.add({ 
                                id: label, 
                                label: label.length > 15 ? label.substring(0,12)+'...' : label, 
                                title: label, 
                                color: colors 
                            });
                            existingNodeIds.add(label);
                        } catch(e) {}
                    }
                });

                // Edge
                const edgeId = `${f.subject}-${f.predicate}-${f.object}`;
                if (!existingEdgeIds.has(edgeId)) {
                    try {
                        edges.add({ 
                            id: edgeId, 
                            from: f.subject, 
                            to: f.object, 
                            label: f.predicate, 
                            font: { size: 10, align: 'middle', color: '#9ca3af', strokeWidth: 0 } 
                        });
                        existingEdgeIds.add(edgeId);
                    } catch(e) {}
                }
            });
            
            document.getElementById('stat-nodes').innerText = nodes.length;
            document.getElementById('stat-edges').innerText = edges.length;
        }

        function addMessage(msg) {
            const container = document.getElementById('conversation-log');
            if (container.children[0]?.classList.contains('text-center')) container.innerHTML = '';
            
            const div = document.createElement('div');
            div.className = `p-3 rounded-xl border border-white/5 bg-white/5 text-sm fade-in mb-3`;
            
            div.innerHTML = `
                <div class="flex justify-between items-center mb-1 text-xs opacity-60">
                    <span class="font-bold text-${msg.speaker === 'Caroline' ? 'blue' : 'purple'}-400">${msg.speaker}</span>
                    <span>${msg.date}</span>
                </div>
                <p class="text-gray-200 leading-relaxed">${msg.text}</p>
            `;
            container.appendChild(div);
            container.scrollTop = container.scrollHeight;
        }

        function addFacts(facts) {
            const container = document.getElementById('facts-log');
            facts.forEach(f => {
                const div = document.createElement('div');
                div.className = "group flex items-center gap-2 p-2 rounded-lg hover:bg-white/5 transition-colors border-b border-white/5 last:border-0 fade-in text-xs";
                div.innerHTML = `
                    <div class="w-1.5 h-1.5 rounded-full bg-blue-500 shadow-[0_0_8px_rgba(59,130,246,0.5)]"></div>
                    <div class="flex-1 min-w-0">
                        <span class="text-gray-300 font-medium truncate">${f.subject}</span>
                        <span class="text-gray-500 px-1 text-[10px] uppercase">${f.predicate}</span>
                        <span class="text-gray-300 font-medium truncate">${f.object}</span>
                    </div>
                `;
                container.prepend(div);
            });
            // Keep limit
            while (container.children.length > 30) container.lastChild.remove();
        }

        function updateTemporal(states) {
            const container = document.getElementById('temporal-log');
            container.innerHTML = '';
            states.forEach(s => {
                const div = document.createElement('div');
                div.className = "p-3 bg-gradient-to-r from-orange-900/20 to-transparent border-l-2 border-orange-500 rounded-r-lg mb-2";
                div.innerHTML = `
                    <div class="flex justify-between text-xs mb-1">
                        <span class="font-bold text-gray-200">${s.subject}</span>
                        <span class="text-orange-400 mono">${s.duration}</span>
                    </div>
                    <div class="text-[10px] text-gray-500 truncate">${s.desc}</div>
                `;
                container.appendChild(div);
            });
        }

        // Poll Loop
        setInterval(async () => {
            try {
                const res = await fetch('/api/state');
                const data = await res.json();
                
                // Messages
                if (data.last_message && (!window.lastMsgId || window.lastMsgId !== data.last_message.id)) {
                    window.lastMsgId = data.last_message.id;
                    addMessage(data.last_message);
                    
                    // Update everything else on new message
                    updateGraph(data.facts);
                    addFacts(data.new_facts);
                    updateTemporal(data.temporal);
                }
                
            } catch(e) {}
        }, 1000);

        // Init
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
    
    # Get all facts from V5 triplets
    all_facts = []
    for triplet in memory.graph.triplets.values():
        if triplet.is_current:
            all_facts.append({
                "subject": triplet.subject.name,
                "subject_type": triplet.subject.entity_type.value,
                "predicate": triplet.predicate.relation_type.value,
                "object": triplet.object.name,
                "object_type": triplet.object.entity_type.value,
            })
    
    # Get recent facts (last 5)
    sorted_triplets = sorted(
        memory.graph.triplets.values(), 
        key=lambda x: x.extraction_time, 
        reverse=True
    )
    new_facts = []
    for t in sorted_triplets[:5]:
        new_facts.append({
            "subject": t.subject.name,
            "predicate": t.predicate.relation_type.value,
            "object": t.object.name
        })
    
    # Get temporal states
    temporal = []
    for st in memory.temporal_tracker.states.values():
        temporal.append({
            "subject": st.subject,
            "desc": st.description,
            "duration": st.calculate_duration_from_reference()
        })
    
    return jsonify({
        "last_message": last_msg,
        "facts": all_facts,
        "new_facts": new_facts,
        "temporal": temporal
    })

@app.route('/api/reset', methods=['POST'])
def reset():
    """Reset memory and clear UI state."""
    global CURRENT_CONVERSATION, MEMORY
    
    # Clear the memory (V5 has a clear method)
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
    import sys
    
    try:
        print("[Feeder] Starting...", flush=True)
        
        # Load data
        data_path = "benchmarks/locomo_data/data/locomo10.json"
        if not os.path.exists(data_path):
            print(f"[Feeder] Data not found: {data_path}", flush=True)
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
        
        print(f"[Feeder] Loaded {len(turns)} turns", flush=True)
        
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
            try:
                memory.add_conversation_turn(
                    speaker=turn['speaker'],
                    text=turn['text'],
                    date=turn['date']
                )
            except Exception as e:
                print(f"[Feeder] Error on turn {i}: {e}", flush=True)
            
            # Simulate typing/processing delay for visual effect
            time.sleep(0.3)
        
        print("[Feeder] Complete!", flush=True)
        
    except Exception as e:
        import traceback
        print(f"[Feeder] CRITICAL ERROR: {e}", flush=True)
        traceback.print_exc()
        sys.stdout.flush()

def open_browser():
    time.sleep(1.5)
    webbrowser.open('http://127.0.0.1:5002')

def main():
    print("Starting V5 Benchmark Visualizer on port 5002...")
    Thread(target=open_browser).start()
    app.run(host='0.0.0.0', port=5002, debug=False)

if __name__ == "__main__":
    main()
