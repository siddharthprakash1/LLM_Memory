"""
LongMemEval Graph Visualization.

Adapts the rich graph UI from benchmark_viz.py to run the LongMemEval benchmark.
"""

import json
import os
import time
import sys
import threading
import webbrowser
from pathlib import Path
from flask import Flask, render_template_string, request, jsonify

# Add package to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_memory.memory_v4.memory_store import MemoryStoreV4
from llm_memory.memory_v4.retrieval import create_retriever
from benchmarks.longmemeval_benchmark import LongMemEvalRunner, exact_match, normalize_answer

# Initialize Flask app
app = Flask(__name__)

# Global state
MEMORY = None
MEMORY_PATH = "./longmemeval_graph_memory"
CURRENT_CONVERSATION = []
EXTRACTED_FACTS = []
TEMPORAL_STATES = []
REASONING_LOGS = []
BENCHMARK_STATE = {
    "status": "idle",
    "total_questions": 0,
    "current_index": 0,
    "current_question": None,
    "results": [],
    "accuracy": 0.0
}

def get_memory(user_id="viz_user"):
    global MEMORY
    if MEMORY is None:
        MEMORY = MemoryStoreV4(
            user_id=user_id,
            persist_path=MEMORY_PATH,
            model_name="qwen2.5:32b",
            use_llm_extraction=True
        )
    return MEMORY

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LongMemEval Graph Viz</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <script src="https://unpkg.com/lucide@latest"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Inter:wght@300;400;500;600&display=swap');
        body { font-family: 'Inter', sans-serif; }
        .mono { font-family: 'JetBrains Mono', monospace; }
        .scrollbar-hide::-webkit-scrollbar { display: none; }
        .fade-in { animation: fadeIn 0.3s ease-out forwards; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(5px); } to { opacity: 1; transform: translateY(0); } }
        
        /* CustomScrollbar */
        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-track { background: #1f2937; }
        ::-webkit-scrollbar-thumb { background: #374151; border-radius: 3px; }
        ::-webkit-scrollbar-thumb:hover { background: #4b5563; }
    </style>
</head>
<body class="bg-gray-950 text-gray-100 h-screen flex flex-col overflow-hidden">

    <!-- Header / Benchmark Status -->
    <div class="h-14 border-b border-gray-800 bg-gray-900 flex items-center justify-between px-4">
        <div class="flex items-center gap-3">
            <div class="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center">
                <i data-lucide="brain-circuit" class="text-white w-5 h-5"></i>
            </div>
            <h1 class="font-bold text-lg text-gray-200">LongMemEval <span class="text-blue-500">for Memory V4</span></h1>
            <div class="px-2 py-0.5 rounded text-[10px] font-bold bg-blue-900/30 text-blue-400 border border-blue-500/30">CORE ARCHITECTURE</div>
        </div>
        
        <div class="flex items-center gap-6">
            <div class="text-sm">
                <span class="text-gray-500">Status:</span>
                <span id="status-badge" class="ml-1 px-2 py-0.5 rounded text-xs font-bold bg-gray-800 text-gray-400">IDLE</span>
            </div>
            <div class="text-sm">
                <span class="text-gray-500">Progress:</span>
                <span id="progress-text" class="ml-1 text-gray-300">0/0</span>
            </div>
            <div class="text-sm">
                <span class="text-gray-500">Accuracy:</span>
                <span id="accuracy-text" class="ml-1 font-mono font-bold text-blue-400">0.0%</span>
            </div>
        </div>

        <div class="flex gap-2">
            <button onclick="startBenchmark()" id="btn-start" class="bg-blue-600 hover:bg-blue-500 text-white px-4 py-1.5 rounded text-sm font-medium transition-colors flex items-center gap-2">
                <i data-lucide="play" class="w-4 h-4"></i> Start
            </button>
        </div>
    </div>

    <!-- Main Content -->
    <div class="flex-1 flex overflow-hidden">
        
        <!-- Left: Conversation & Facts -->
        <div class="w-1/4 border-r border-gray-800 flex flex-col bg-gray-900/50">
            <!-- Top: Conversation -->
            <div class="h-1/2 flex flex-col border-b border-gray-800">
                <div class="p-3 border-b border-gray-800 bg-gray-800/30 flex justify-between items-center">
                    <h2 class="text-xs font-bold text-gray-400 uppercase tracking-wider flex items-center gap-2">
                        <i data-lucide="message-square" class="w-3 h-3 text-blue-400"></i> History Stream
                    </h2>
                    <span id="msg-count" class="text-[10px] text-gray-600">0 msgs</span>
                </div>
                <div id="conversation-log" class="flex-1 overflow-y-auto p-3 space-y-3"></div>
            </div>
            
            <!-- Bottom: Facts -->
            <div class="h-1/2 flex flex-col">
                <div class="p-3 border-b border-gray-800 bg-gray-800/30 flex justify-between items-center">
                    <h2 class="text-xs font-bold text-gray-400 uppercase tracking-wider flex items-center gap-2">
                        <i data-lucide="database" class="w-3 h-3 text-green-400"></i> Live Extraction
                    </h2>
                </div>
                <div id="facts-log" class="flex-1 overflow-y-auto p-3 space-y-2"></div>
            </div>
        </div>

        <!-- Middle: Graph -->
        <div class="w-2/4 border-r border-gray-800 flex flex-col bg-gray-950 relative">
            <div class="absolute top-3 left-3 z-10 bg-gray-900/80 backdrop-blur rounded p-2 border border-gray-800">
                <div class="text-xs text-gray-400" id="node-count">0 Nodes</div>
                <div class="text-xs text-gray-400" id="edge-count">0 Edges</div>
            </div>
            <div id="network-container" class="flex-1 w-full h-full"></div>
            <div class="h-1/3 border-t border-gray-800 flex flex-col bg-gray-900/30">
                 <div class="p-3 border-b border-gray-800 bg-gray-800/30">
                    <h2 class="text-xs font-bold text-gray-400 uppercase tracking-wider flex items-center gap-2">
                        <i data-lucide="clock" class="w-3 h-3 text-orange-400"></i> Temporal State
                    </h2>
                </div>
                <div id="temporal-log" class="flex-1 overflow-y-auto p-3 grid grid-cols-2 gap-2 content-start"></div>
            </div>
        </div>

        <!-- Right: Question & Results -->
        <div class="w-1/4 flex flex-col bg-gray-900/50">
            <!-- Current Question -->
            <div class="flex-1 flex flex-col border-b border-gray-800">
                <div class="p-3 border-b border-gray-800 bg-gray-800/30">
                    <h2 class="text-xs font-bold text-gray-400 uppercase tracking-wider flex items-center gap-2">
                        <i data-lucide="help-circle" class="w-3 h-3 text-purple-400"></i> Current Question
                    </h2>
                </div>
                <div id="current-question-panel" class="p-4 flex-1 overflow-y-auto">
                    <div class="text-center text-gray-600 mt-10 text-sm">Waiting to start...</div>
                </div>
            </div>

            <!-- Results Log -->
            <div class="h-1/2 flex flex-col">
                <div class="p-3 border-b border-gray-800 bg-gray-800/30">
                    <h2 class="text-xs font-bold text-gray-400 uppercase tracking-wider flex items-center gap-2">
                        <i data-lucide="clipboard-check" class="w-3 h-3 text-teal-400"></i> Results Log
                    </h2>
                </div>
                <div id="results-log" class="flex-1 overflow-y-auto p-3 space-y-2"></div>
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
                    size: 8,
                    font: { size: 10, color: '#9ca3af', face: 'Inter' },
                    borderWidth: 0,
                    shadow: true
                },
                edges: {
                    width: 1,
                    color: { color: '#4b5563', highlight: '#60a5fa' },
                    arrows: { to: { enabled: true, scaleFactor: 0.4 } },
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
                layout: { randomSeed: 2 }
            };
            network = new vis.Network(container, data, options);
        }

        async function startBenchmark() {
            document.getElementById('btn-start').disabled = true;
            document.getElementById('btn-start').classList.add('opacity-50');
            await fetch('/api/start', { method: 'POST' });
        }

        // --- Render Helpers ---

        function renderMessage(msg) {
            const div = document.createElement('div');
            div.className = "bg-gray-800 rounded p-2 border border-gray-700/50 fade-in";
            div.innerHTML = `
                <div class="flex justify-between items-start mb-1">
                    <span class="text-[11px] font-bold text-${msg.role === 'user' ? 'blue' : 'purple'}-400 uppercase">${msg.role}</span>
                    <span class="text-[10px] text-gray-600">${msg.date}</span>
                </div>
                <p class="text-xs text-gray-300 leading-relaxed">${msg.content}</p>
            `;
            return div;
        }

        function renderFact(f) {
            const div = document.createElement('div');
            // Fact type colors
            let borderColor = 'border-gray-700/50';
            let iconColor = 'text-gray-400';
            if(f.type === 'preference') { borderColor = 'border-pink-900/30'; iconColor = 'text-pink-400'; }
            if(f.type === 'temporal') { borderColor = 'border-orange-900/30'; iconColor = 'text-orange-400'; }

            div.className = `bg-gray-800/30 p-2 rounded border ${borderColor} text-xs flex gap-2 items-center fade-in`;
            div.innerHTML = `
                <div class="w-1.5 h-1.5 rounded-full ${f.type === 'preference' ? 'bg-pink-500' : 'bg-blue-500'} flex-shrink-0"></div>
                <div class="flex-1 min-w-0">
                    <div class="flex items-center gap-1 overflow-hidden">
                         <span class="text-gray-300 font-medium truncate">${f.subject}</span>
                         <span class="text-gray-500 text-[10px]">&rarr;</span>
                         <span class="text-gray-300 font-medium truncate">${f.object}</span>
                    </div>
                    <div class="text-[10px] text-gray-500 truncate">${f.predicate}</div>
                </div>
            `;
            return div;
        }

        function renderTemporal(t) {
            const div = document.createElement('div');
            div.className = "bg-gray-800 p-2 rounded border-l-2 border-orange-500 fade-in";
            div.innerHTML = `
                <div class="flex justify-between">
                    <span class="text-[10px] font-bold text-gray-300 truncate">${t.subject}</span>
                    <span class="text-[10px] text-orange-400 font-mono">${t.duration}</span>
                </div>
                <p class="text-[10px] text-gray-500 truncate mt-0.5">${t.desc}</p>
            `;
            return div;
        }

        function renderQuestion(q) {
            if(!q) return `<div class="text-center text-gray-600 mt-10 text-sm">Idle</div>`;
            return `
                <div class="space-y-4 fade-in">
                    <div>
                        <div class="text-[10px] text-gray-500 uppercase tracking-widest mb-1">Question Type</div>
                        <div class="text-xs font-mono text-purple-400 bg-purple-900/20 px-2 py-1 rounded inline-block">
                            ${q.type}
                        </div>
                    </div>
                    <div>
                        <div class="text-[10px] text-gray-500 uppercase tracking-widest mb-1">Question</div>
                        <p class="text-sm text-gray-200 font-medium">${q.text}</p>
                    </div>
                    ${q.prediction ? `
                    <div class="pt-4 border-t border-gray-800">
                        <div class="text-[10px] text-gray-500 uppercase tracking-widest mb-1">Answer</div>
                        <p class="text-sm text-white">${q.prediction}</p>
                    </div>
                    <div class="grid grid-cols-2 gap-4">
                         <div>
                            <div class="text-[10px] text-gray-500 uppercase tracking-widest mb-1">Ground Truth</div>
                            <p class="text-xs text-gray-400">${q.ground_truth}</p>
                         </div>
                         <div>
                             <div class="text-[10px] text-gray-500 uppercase tracking-widest mb-1">Result</div>
                             <span class="${q.correct ? 'text-green-400' : 'text-red-400'} font-bold text-sm">
                                ${q.correct ? 'CORRECT' : 'INCORRECT'}
                             </span>
                         </div>
                    </div>
                    ` : `
                    <div class="pt-4 border-t border-gray-800 animate-pulse">
                        <div class="text-xs text-blue-400">Processing memory...</div>
                    </div>
                    `}
                </div>
            `;
        }

        function renderResultItem(r) {
            const div = document.createElement('div');
            div.className = `p-2 rounded border ${r.correct ? 'bg-green-900/10 border-green-900/30' : 'bg-red-900/10 border-red-900/30'} fade-in text-xs`;
            div.innerHTML = `
                <div class="flex justify-between mb-1">
                    <span class="font-mono text-gray-500">${r.id.substring(0,6)}</span>
                    <span class="${r.correct ? 'text-green-400' : 'text-red-400'} font-bold">${r.correct ? 'PASS' : 'FAIL'}</span>
                </div>
                <div class="text-gray-300 truncate">${r.text}</div>
            `;
            return div;
        }

        // --- Main Update Loop ---
        let lastMsgId = -1;
        let lastResultCount = 0;
        let lastQuestionId = null;

        setInterval(async () => {
            try {
                const res = await fetch('/api/state');
                const data = await res.json();
                
                // Status Bar
                const statusBadge = document.getElementById('status-badge');
                statusBadge.innerText = data.benchmark.status.toUpperCase();
                statusBadge.className = `ml-1 px-2 py-0.5 rounded text-xs font-bold ${data.benchmark.status === 'running' ? 'bg-green-900 text-green-400' : 'bg-gray-800 text-gray-400'}`;
                
                document.getElementById('progress-text').innerText = `${data.benchmark.current_index}/${data.benchmark.total_questions}`;
                document.getElementById('accuracy-text').innerText = `${data.benchmark.accuracy.toFixed(1)}%`;

                // Conversation Log
                const log = document.getElementById('conversation-log');
                if (data.conversation.length > 0) {
                     // Check if we have new messages
                    const newMsgs = data.conversation.slice(lastMsgId + 1);
                    newMsgs.forEach((msg, idx) => {
                        log.appendChild(renderMessage(msg));
                    });
                    if (newMsgs.length > 0) {
                        lastMsgId = data.conversation.length - 1;
                        log.scrollTop = log.scrollHeight;
                        document.getElementById('msg-count').innerText = `${data.conversation.length} msgs`;
                    }
                } else if (lastMsgId !== -1) {
                    // Reset
                    log.innerHTML = '';
                    lastMsgId = -1;
                }

                // Facts Log - Just overwrite for simplicity or diff? 
                // Let's verify lengths to avoid constant DOM thrashing
                const factsLog = document.getElementById('facts-log');
                if (factsLog.childElementCount !== data.facts.length) {
                    factsLog.innerHTML = '';
                    // Show newest first
                    [...data.facts].reverse().forEach(f => factsLog.appendChild(renderFact(f)));
                }

                // Temporal Log
                const tempLog = document.getElementById('temporal-log');
                if (tempLog.childElementCount !== data.temporal.length) {
                    tempLog.innerHTML = '';
                    data.temporal.forEach(t => tempLog.appendChild(renderTemporal(t)));
                }

                // Graph Update
                updateGraph(data.facts);

                // Current Question
                const qPanel = document.getElementById('current-question-panel');
                const q = data.benchmark.current_question;
                // re-render if it changed or if prediction status changed
                const qSig = q ? `${q.id}-${!!q.prediction}` : 'null';
                if (lastQuestionId !== qSig) {
                     qPanel.innerHTML = renderQuestion(q);
                     lastQuestionId = qSig;
                }

                // Results Log
                const resLog = document.getElementById('results-log');
                if (data.benchmark.results.length > lastResultCount) {
                    const newResults = data.benchmark.results.slice(lastResultCount);
                    newResults.forEach(r => resLog.prepend(renderResultItem(r)));
                    lastResultCount = data.benchmark.results.length;
                }

            } catch(e) { console.error(e); }
        }, 500);

        function updateGraph(facts) {
            const existingNodes = new Set(nodes.getIds());
            const existingEdges = new Set(edges.getIds());
            const newNodes = [];
            const newEdges = [];

            facts.forEach(f => {
                [f.subject, f.object].forEach(label => {
                    if (!existingNodes.has(label)) {
                        let color = '#60a5fa'; // Blue
                        if(f.type === 'preference') color = '#f472b6'; // Pink
                        if(f.type === 'temporal') color = '#fb923c'; // Orange
                        newNodes.push({ id: label, label: label, color: color });
                        existingNodes.add(label);
                    }
                });
                const edgeId = `${f.subject}-${f.predicate}-${f.object}`;
                if (!existingEdges.has(edgeId)) {
                    newEdges.push({ id: edgeId, from: f.subject, to: f.object, label: f.predicate });
                    existingEdges.add(edgeId);
                }
            });

            if (newNodes.length > 0) nodes.add(newNodes);
            if (newEdges.length > 0) edges.add(newEdges);
            
            document.getElementById('node-count').innerText = `${nodes.length} Nodes`;
            document.getElementById('edge-count').innerText = `${edges.length} Edges`;
        }

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
    """Get full state."""
    memory = get_memory()
    
    # Format Facts
    facts = []
    if memory and hasattr(memory, 'facts'):
        for f in memory.facts.values():
            if f.is_current:
                facts.append({
                    "subject": f.subject,
                    "predicate": f.predicate,
                    "object": f.object,
                    "type": f.fact_type
                })
    
    # Format Temporal
    metrics = []
    if memory and hasattr(memory, 'temporal_states'):
        for s in memory.temporal_states.values():
            metrics.append({
                "subject": s.subject,
                "desc": s.description,
                "duration": s.calculate_duration_from_reference()
            })

    return jsonify({
        "conversation": CURRENT_CONVERSATION,
        "facts": facts,
        "temporal": metrics,
        "benchmark": BENCHMARK_STATE
    })

@app.route('/api/start', methods=['POST'])
def start_benchmark():
    BENCHMARK_STATE['status'] = 'starting'
    threading.Thread(target=run_benchmark_loop).start()
    return jsonify({"status": "started"})

def run_benchmark_loop():
    """Main benchmark loop."""
    global MEMORY, CURRENT_CONVERSATION
    
    BENCHMARK_STATE['status'] = 'running'
    BENCHMARK_STATE['results'] = []
    BENCHMARK_STATE['accuracy'] = 0.0
    
    # 1. Initialize Runner & Load Data
    runner = LongMemEvalRunner(
        model_name="qwen2.5:32b",
        memory_path=MEMORY_PATH,
        max_questions=5  # Limit for demo
    )
    dataset_path = "benchmarks/datasets/longmemeval/longmemeval_s_cleaned.json"
    questions = runner.load_dataset(dataset_path)
    
    BENCHMARK_STATE['total_questions'] = len(questions)
    
    for i, q_data in enumerate(questions):
        BENCHMARK_STATE['current_index'] = i + 1
        
        # Reset UI
        CURRENT_CONVERSATION = []
        MEMORY = None # Force reset
        
        # Clean local memory dir for this user
        import shutil
        if os.path.exists(MEMORY_PATH):
            shutil.rmtree(MEMORY_PATH)
            
        memory = get_memory()
        
        # Setup Current Question in UI
        BENCHMARK_STATE['current_question'] = {
            "id": q_data['question_id'],
            "type": q_data['question_type'],
            "text": q_data['question'],
            "ground_truth": q_data['answer'],
            "prediction": None,
            "correct": False
        }
        
        # --- PHASE 1: Feed History ---
        sessions = q_data['haystack_sessions']
        dates = q_data['haystack_dates']
        sess_ids = q_data['haystack_session_ids']
        
        for sess_idx, (session, date, sess_id) in enumerate(zip(sessions, dates, sess_ids)):
            for turn in session:
                # Add to memory
                memory.add_conversation_turn(
                    speaker=turn['role'],
                    text=turn['content'],
                    date=date,
                    session_id=sess_id
                )
                
                # Update UI
                CURRENT_CONVERSATION.append({
                    "role": turn['role'],
                    "content": turn['content'],
                    "date": date
                })
                
                # Small delay for visual effect
                time.sleep(0.4) 
        
        # --- PHASE 2: Answer ---
        BENCHMARK_STATE['current_question']['prediction'] = "Processing..."
        time.sleep(1) # Pause before answering
        
        try:
            answer = memory.answer_question(q_data['question'])
        except Exception as e:
            answer = f"Error: {str(e)}"
            
        is_correct = exact_match(answer, q_data['answer'])
        
        # Update Question State
        BENCHMARK_STATE['current_question']['prediction'] = answer
        BENCHMARK_STATE['current_question']['correct'] = is_correct
        
        # Add to Results Log
        BENCHMARK_STATE['results'].append({
            "id": q_data['question_id'],
            "text": q_data['question'],
            "correct": is_correct
        })
        
        # Update Stats
        correct_count = sum(1 for r in BENCHMARK_STATE['results'] if r['correct'])
        BENCHMARK_STATE['accuracy'] = (correct_count / (i + 1)) * 100
        
        # Pause before next question
        time.sleep(3)
    
    BENCHMARK_STATE['status'] = 'completed'

if __name__ == "__main__":
    url = "http://localhost:5001"
    print(f"Starting Graph Viz at {url}")
    threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    app.run(host='0.0.0.0', port=5001)
