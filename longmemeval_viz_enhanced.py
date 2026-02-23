"""
LongMemEval Enhanced Visualization UI (V4 Memory System).

A visually rich dashboard with the Knowledge Graph as the centerpiece.
Shows real-time memory formation during LongMemEval benchmark runs.

Run:  ./venv/bin/python longmemeval_viz_enhanced.py
Open: http://127.0.0.1:5003
"""

import json
import os
import time
from flask import Flask, render_template_string, request, jsonify
from threading import Thread
import webbrowser

from llm_memory.memory_v4.memory_store import MemoryStoreV4
from llm_memory.memory_v4.retrieval import create_retriever

app = Flask(__name__)

# --------------- Global State ---------------
MEMORY = None
MEMORY_PATH = "./longmemeval_viz_enhanced_memory"
CURRENT_CONVERSATION = []
BENCHMARK_STATUS = {"running": False, "progress": 0, "total": 0, "current_session": ""}
FACT_HISTORY = []  # track fact count over time for the sparkline chart

def get_memory():
    global MEMORY
    if MEMORY is None:
        MEMORY = MemoryStoreV4(
            user_id="longmemeval_viz_enhanced",
            persist_path=MEMORY_PATH,
            model_name="qwen3:32b",          # Best 32B model - fast + semantic
            use_llm_extraction=True,         # Full semantic LLM extraction
        )
    return MEMORY

# =============================================
# HTML Template  (Graph-dominant layout)
# =============================================
HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LongMemEval V4 - Memory Brain</title>
<script src="https://cdn.tailwindcss.com"></script>
<script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
<script src="https://unpkg.com/lucide@latest"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Inter:wght@300;400;500;600;700&display=swap');
  * { box-sizing: border-box; }
  body { font-family: 'Inter', sans-serif; margin: 0; overflow: hidden; background: #0a0a0f; }
  .mono { font-family: 'JetBrains Mono', monospace; }

  /* Custom scrollbar */
  ::-webkit-scrollbar { width: 4px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: #334155; border-radius: 4px; }

  /* Animations */
  @keyframes fadeSlideIn {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
  }
  @keyframes pulseGlow {
    0%, 100% { box-shadow: 0 0 8px rgba(99,102,241,.15); }
    50%      { box-shadow: 0 0 20px rgba(99,102,241,.35); }
  }
  @keyframes countUp {
    from { opacity: 0; transform: scale(0.8); }
    to   { opacity: 1; transform: scale(1); }
  }
  .fade-in      { animation: fadeSlideIn .35s ease-out forwards; }
  .glow-pulse   { animation: pulseGlow 2.5s ease-in-out infinite; }
  .count-pop    { animation: countUp .25s ease-out; }

  /* Graph container glow border */
  #graph-wrapper {
    position: relative;
    border: 1px solid #1e1e2e;
    border-radius: 16px;
    overflow: hidden;
    background: radial-gradient(ellipse at center, #0f0f1a 0%, #0a0a0f 70%);
  }
  #graph-wrapper::before {
    content: '';
    position: absolute;
    inset: -1px;
    border-radius: 16px;
    padding: 1px;
    background: linear-gradient(135deg, rgba(99,102,241,.25), rgba(236,72,153,.2), rgba(251,146,60,.2));
    -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
    mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
    -webkit-mask-composite: xor;
    mask-composite: exclude;
    pointer-events: none;
    z-index: 10;
  }

  /* Stat card glass */
  .stat-card {
    background: rgba(15,15,26,.8);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,.06);
    border-radius: 12px;
    padding: 12px 16px;
    min-width: 0;
  }

  /* Progress bar */
  .progress-track { background: #1e1e2e; border-radius: 999px; height: 4px; overflow: hidden; }
  .progress-fill  { height: 100%; border-radius: 999px; background: linear-gradient(90deg, #6366f1, #ec4899); transition: width .4s ease; }

  /* Fact pill */
  .fact-pill {
    background: rgba(15,15,26,.6);
    border: 1px solid rgba(255,255,255,.05);
    border-radius: 8px;
    padding: 6px 10px;
    font-size: 11px;
    display: flex;
    gap: 6px;
    align-items: center;
  }
</style>
</head>

<body class="text-gray-200 h-screen flex flex-col">

  <!-- ===== TOP BAR ===== -->
  <header class="flex items-center justify-between px-6 py-3 border-b border-white/5 bg-[#0a0a0f]/90 backdrop-blur z-20 flex-shrink-0">
    <div class="flex items-center gap-3">
      <div class="w-8 h-8 rounded-lg bg-gradient-to-br from-indigo-500 to-pink-500 flex items-center justify-center">
        <i data-lucide="brain" class="w-4 h-4 text-white"></i>
      </div>
      <div>
        <h1 class="text-sm font-bold text-white tracking-tight">LongMemEval V4</h1>
        <p class="text-[10px] text-gray-500 -mt-0.5">CORE-Style Memory Visualizer</p>
      </div>
    </div>

    <!-- Live stats row -->
    <div class="flex items-center gap-4">
      <div class="flex items-center gap-2 text-xs">
        <span class="text-gray-500">Nodes</span>
        <span class="font-bold text-indigo-400 mono" id="hdr-nodes">0</span>
      </div>
      <div class="flex items-center gap-2 text-xs">
        <span class="text-gray-500">Edges</span>
        <span class="font-bold text-pink-400 mono" id="hdr-edges">0</span>
      </div>
      <div class="flex items-center gap-2 text-xs">
        <span class="text-gray-500">Facts</span>
        <span class="font-bold text-emerald-400 mono" id="hdr-facts">0</span>
      </div>
      <div class="flex items-center gap-2 text-xs">
        <span class="text-gray-500">Episodes</span>
        <span class="font-bold text-orange-400 mono" id="hdr-episodes">0</span>
      </div>
      <div class="flex items-center gap-2 text-xs">
        <span class="text-gray-500">Temporal</span>
        <span class="font-bold text-amber-400 mono" id="hdr-temporal">0</span>
      </div>
    </div>

    <div class="flex items-center gap-2">
      <button onclick="startBenchmark()" id="btn-start" class="bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold px-4 py-2 rounded-lg transition-all flex items-center gap-1.5">
        <i data-lucide="play" class="w-3.5 h-3.5"></i> Start
      </button>
      <button onclick="resetSystem()" class="bg-white/5 hover:bg-white/10 text-gray-400 text-xs px-3 py-2 rounded-lg transition-all">
        <i data-lucide="rotate-ccw" class="w-3.5 h-3.5"></i>
      </button>
    </div>
  </header>

  <!-- ===== PROGRESS BAR ===== -->
  <div class="px-6 py-1 flex-shrink-0" id="progress-wrapper" style="display:none;">
    <div class="flex items-center gap-3">
      <div class="progress-track flex-1"><div class="progress-fill" id="progress-bar" style="width:0%"></div></div>
      <span class="text-[10px] text-gray-500 mono" id="progress-text">0 / 0</span>
    </div>
  </div>

  <!-- ===== MAIN CONTENT ===== -->
  <div class="flex flex-1 min-h-0">

    <!-- LEFT SIDEBAR: Conversation + Facts -->
    <aside class="w-[320px] flex flex-col border-r border-white/5 flex-shrink-0">

      <!-- Conversation -->
      <div class="flex-1 flex flex-col min-h-0">
        <div class="px-4 py-2.5 border-b border-white/5 flex items-center gap-2">
          <i data-lucide="message-square" class="w-3.5 h-3.5 text-blue-400"></i>
          <span class="text-[11px] font-semibold text-gray-400 uppercase tracking-wider">Conversation</span>
          <span class="ml-auto text-[10px] text-gray-600 mono" id="msg-count">0</span>
        </div>
        <div id="conversation-log" class="flex-1 overflow-y-auto p-3 space-y-2">
        </div>
      </div>

      <!-- Facts stream -->
      <div class="h-[280px] flex flex-col border-t border-white/5">
        <div class="px-4 py-2.5 border-b border-white/5 flex items-center gap-2">
          <i data-lucide="sparkles" class="w-3.5 h-3.5 text-emerald-400"></i>
          <span class="text-[11px] font-semibold text-gray-400 uppercase tracking-wider">Live Facts</span>
        </div>
        <div id="facts-log" class="flex-1 overflow-y-auto p-3 space-y-1.5">
        </div>
      </div>
    </aside>

    <!-- CENTER: THE GRAPH (dominant) -->
    <main class="flex-1 flex flex-col p-4 min-h-0">
      <div id="graph-wrapper" class="flex-1 min-h-0">
        <div id="network-container" class="w-full h-full"></div>
      </div>

      <!-- Bottom stat cards under graph -->
      <div class="grid grid-cols-5 gap-3 mt-3">
        <div class="stat-card">
          <div class="text-[10px] text-gray-500 uppercase tracking-wider">Total Facts</div>
          <div class="text-xl font-bold text-emerald-400 mono mt-1" id="stat-facts">0</div>
        </div>
        <div class="stat-card">
          <div class="text-[10px] text-gray-500 uppercase tracking-wider">Episodes</div>
          <div class="text-xl font-bold text-blue-400 mono mt-1" id="stat-episodes">0</div>
        </div>
        <div class="stat-card">
          <div class="text-[10px] text-gray-500 uppercase tracking-wider">Graph Nodes</div>
          <div class="text-xl font-bold text-indigo-400 mono mt-1" id="stat-nodes">0</div>
        </div>
        <div class="stat-card">
          <div class="text-[10px] text-gray-500 uppercase tracking-wider">Graph Edges</div>
          <div class="text-xl font-bold text-pink-400 mono mt-1" id="stat-edges">0</div>
        </div>
        <div class="stat-card">
          <div class="text-[10px] text-gray-500 uppercase tracking-wider">Fact Growth</div>
          <canvas id="sparkline" class="mt-1" height="30"></canvas>
        </div>
      </div>
    </main>

    <!-- RIGHT SIDEBAR: Temporal + Reasoning -->
    <aside class="w-[300px] flex flex-col border-l border-white/5 flex-shrink-0">

      <!-- Temporal -->
      <div class="flex-1 flex flex-col min-h-0">
        <div class="px-4 py-2.5 border-b border-white/5 flex items-center gap-2">
          <i data-lucide="clock" class="w-3.5 h-3.5 text-orange-400"></i>
          <span class="text-[11px] font-semibold text-gray-400 uppercase tracking-wider">Temporal States</span>
        </div>
        <div id="temporal-log" class="flex-1 overflow-y-auto p-3 space-y-2">
        </div>
      </div>

      <!-- Reasoning -->
      <div class="h-[300px] flex flex-col border-t border-white/5">
        <div class="px-4 py-2.5 border-b border-white/5 flex items-center gap-2">
          <i data-lucide="cpu" class="w-3.5 h-3.5 text-violet-400"></i>
          <span class="text-[11px] font-semibold text-gray-400 uppercase tracking-wider">Reasoning Log</span>
        </div>
        <div id="reasoning-log" class="flex-1 overflow-y-auto p-3 space-y-2 mono text-[11px] text-gray-500">
        </div>
      </div>
    </aside>
  </div>

<script>
// ============================================================
// INIT
// ============================================================
lucide.createIcons();
let network = null;
let visNodes = new vis.DataSet();
let visEdges = new vis.DataSet();
let sparkData = [];
let sparkChart = null;

function initGraph() {
  const container = document.getElementById('network-container');
  const options = {
    nodes: {
      shape: 'dot',
      size: 14,
      font: { size: 11, color: '#94a3b8', face: 'Inter', strokeWidth: 3, strokeColor: '#0a0a0f' },
      borderWidth: 2,
      borderWidthSelected: 3,
      shadow: { enabled: true, color: 'rgba(99,102,241,.2)', size: 12, x: 0, y: 0 },
      chosen: { node: function(values) { values.size = 20; values.shadowSize = 20; } }
    },
    edges: {
      width: 1.5,
      color: { color: '#1e293b', highlight: '#6366f1', hover: '#6366f1' },
      arrows: { to: { enabled: true, scaleFactor: 0.4 } },
      smooth: { type: 'continuous', roundness: 0.3 },
      font: { size: 8, align: 'middle', color: '#475569', strokeWidth: 2, strokeColor: '#0a0a0f' },
      chosen: { edge: function(values) { values.width = 3; } }
    },
    physics: {
      stabilization: false,
      barnesHut: {
        gravitationalConstant: -3000,
        centralGravity: 0.15,
        springConstant: 0.03,
        springLength: 120,
        damping: 0.12
      }
    },
    interaction: {
      hover: true,
      tooltipDelay: 150,
      zoomView: true,
      dragView: true,
      navigationButtons: false,
      keyboard: false
    }
  };
  network = new vis.Network(container, { nodes: visNodes, edges: visEdges }, options);
}

// Sparkline chart
function initSparkline() {
  const ctx = document.getElementById('sparkline').getContext('2d');
  sparkChart = new Chart(ctx, {
    type: 'line',
    data: { labels: [], datasets: [{ data: [], borderColor: '#6366f1', borderWidth: 1.5, fill: true, backgroundColor: 'rgba(99,102,241,.1)', pointRadius: 0, tension: 0.4 }] },
    options: {
      responsive: true, maintainAspectRatio: false,
      scales: { x: { display: false }, y: { display: false, beginAtZero: true } },
      plugins: { legend: { display: false } },
      animation: { duration: 300 }
    }
  });
}

// ============================================================
// ACTIONS
// ============================================================
async function startBenchmark() {
  document.getElementById('btn-start').disabled = true;
  document.getElementById('btn-start').classList.add('opacity-50');
  await fetch('/api/start_benchmark', { method: 'POST' });
}

async function resetSystem() {
  await fetch('/api/reset', { method: 'POST' });
  visNodes.clear(); visEdges.clear();
  sparkData = [];
  document.getElementById('conversation-log').innerHTML = '';
  document.getElementById('facts-log').innerHTML = '';
  document.getElementById('temporal-log').innerHTML = '';
  document.getElementById('reasoning-log').innerHTML = '';
  document.getElementById('progress-wrapper').style.display = 'none';
  document.getElementById('btn-start').disabled = false;
  document.getElementById('btn-start').classList.remove('opacity-50');
  ['stat-facts','stat-episodes','stat-nodes','stat-edges','hdr-nodes','hdr-edges','hdr-facts','hdr-episodes','hdr-temporal','msg-count'].forEach(id => {
    const el = document.getElementById(id); if(el) el.innerText = '0';
  });
}

// ============================================================
// GRAPH UPDATE (with colour palette)
// ============================================================
const TYPE_COLORS = {
  'preference':    { bg: '#ec4899', border: '#f472b6' },
  'state_change':  { bg: '#10b981', border: '#34d399' },
  'temporal':      { bg: '#f59e0b', border: '#fbbf24' },
  'attribute':     { bg: '#6366f1', border: '#818cf8' },
  'relationship':  { bg: '#8b5cf6', border: '#a78bfa' },
  'default':       { bg: '#3b82f6', border: '#60a5fa' },
};

function getColor(factType) {
  return TYPE_COLORS[factType] || TYPE_COLORS['default'];
}

function updateGraph(facts) {
  const existingNodeIds = new Set(visNodes.getIds());
  const existingEdgeIds = new Set(visEdges.getIds());

  facts.forEach(f => {
    const c = getColor(f.fact_type);
    [f.subject, f.object].forEach(label => {
      if (!existingNodeIds.has(label)) {
        try {
          visNodes.add({
            id: label,
            label: label.length > 18 ? label.substring(0,15) + '...' : label,
            title: `<b>${label}</b><br>${f.fact_type}`,
            color: { background: c.bg, border: c.border, highlight: { background: c.border, border: '#fff' } },
          });
          existingNodeIds.add(label);
        } catch(e) {}
      }
    });

    const edgeId = `${f.subject}--${f.predicate}--${f.object}`;
    if (!existingEdgeIds.has(edgeId)) {
      try {
        visEdges.add({ id: edgeId, from: f.subject, to: f.object, label: f.predicate });
        existingEdgeIds.add(edgeId);
      } catch(e) {}
    }
  });

  // Update header + stat counts
  const nc = visNodes.length, ec = visEdges.length;
  document.getElementById('hdr-nodes').innerText = nc;
  document.getElementById('hdr-edges').innerText = ec;
  document.getElementById('stat-nodes').innerText = nc;
  document.getElementById('stat-edges').innerText = ec;
}

// ============================================================
// CONVERSATION
// ============================================================
function addMessage(msg) {
  const d = document.createElement('div');
  const isUser = msg.speaker === 'User';
  d.className = `rounded-lg p-2.5 text-xs fade-in ${isUser ? 'bg-blue-500/10 border border-blue-500/20' : 'bg-violet-500/10 border border-violet-500/20'}`;
  d.innerHTML = `
    <div class="flex justify-between mb-1">
      <span class="font-semibold ${isUser ? 'text-blue-400' : 'text-violet-400'}">${msg.speaker}</span>
      <span class="text-[9px] text-gray-600">${msg.date || ''}</span>
    </div>
    <p class="text-gray-300 leading-relaxed">${msg.text.length > 200 ? msg.text.substring(0,200)+'...' : msg.text}</p>
  `;
  const c = document.getElementById('conversation-log');
  c.appendChild(d);
  c.scrollTop = c.scrollHeight;
  document.getElementById('msg-count').innerText = c.children.length;
}

// ============================================================
// FACTS
// ============================================================
function addFacts(facts) {
  const c = document.getElementById('facts-log');
  facts.forEach(f => {
    const col = getColor(f.fact_type);
    const d = document.createElement('div');
    d.className = 'fact-pill fade-in';
    d.innerHTML = `
      <span class="w-2 h-2 rounded-full flex-shrink-0" style="background:${col.bg}"></span>
      <span class="text-gray-400">${f.subject}</span>
      <span class="text-gray-600">${f.predicate}</span>
      <span class="text-gray-300 font-medium">${f.object}</span>
    `;
    c.prepend(d);
  });
}

// ============================================================
// TEMPORAL
// ============================================================
function updateTemporal(states) {
  const c = document.getElementById('temporal-log');
  c.innerHTML = '';
  states.forEach(s => {
    const d = document.createElement('div');
    d.className = 'bg-orange-500/5 border border-orange-500/15 rounded-lg p-2.5 text-xs';
    d.innerHTML = `
      <div class="flex justify-between items-center">
        <span class="font-semibold text-orange-300">${s.subject}</span>
        <span class="text-[10px] text-orange-500 mono">${s.duration_text || 'Ongoing'}</span>
      </div>
      <p class="text-gray-500 mt-1 text-[10px]">${s.description}</p>
    `;
    c.appendChild(d);
  });
  document.getElementById('hdr-temporal').innerText = states.length;
}

// ============================================================
// REASONING
// ============================================================
function updateReasoning(logs) {
  const c = document.getElementById('reasoning-log');
  c.innerHTML = '';
  logs.slice(-15).forEach(log => {
    const d = document.createElement('div');
    d.className = 'bg-violet-500/5 border border-violet-500/15 rounded-lg p-2 fade-in';
    let html = `<div class="text-violet-400 font-bold text-[10px] mb-0.5">${log.type || 'INFO'}</div>`;
    if (log.type === 'DECOMPOSE' && log.steps) {
      log.steps.forEach((s,i) => { html += `<div class="text-gray-400 pl-2 border-l border-violet-500/20">${i+1}. ${s}</div>`; });
    } else if (log.type === 'RETRIEVE') {
      html += `<div class="text-gray-500">"${log.query}"</div><div class="text-emerald-500 mt-0.5">${log.count} facts</div>`;
    } else {
      html += `<div class="text-gray-400">${log.message || JSON.stringify(log)}</div>`;
    }
    d.innerHTML = html;
    c.prepend(d);
  });
}

// ============================================================
// SPARKLINE UPDATE
// ============================================================
function updateSparkline(factCount) {
  sparkData.push(factCount);
  if (sparkData.length > 60) sparkData.shift();
  sparkChart.data.labels = sparkData.map((_,i) => i);
  sparkChart.data.datasets[0].data = sparkData;
  sparkChart.update('none');
}

// ============================================================
// STATS
// ============================================================
function updateStats(stats) {
  document.getElementById('stat-facts').innerText = stats.total_facts || 0;
  document.getElementById('stat-episodes').innerText = stats.total_episodes || 0;
  document.getElementById('hdr-facts').innerText = stats.total_facts || 0;
  document.getElementById('hdr-episodes').innerText = stats.total_episodes || 0;
  updateSparkline(stats.total_facts || 0);
}

// ============================================================
// POLLING
// ============================================================
setInterval(async () => {
  try {
    const res = await fetch('/api/state');
    const data = await res.json();

    if (data.last_message && (!window._lastId || window._lastId !== data.last_message.id)) {
      addMessage(data.last_message);
      window._lastId = data.last_message.id;
      updateGraph(data.facts);
      addFacts(data.new_facts);
      updateTemporal(data.temporal);
      updateStats(data.stats);
    }

    if (data.reasoning_logs && data.reasoning_logs.length > 0) {
      updateReasoning(data.reasoning_logs);
    }

    // Progress bar
    if (data.benchmark_status) {
      const bs = data.benchmark_status;
      if (bs.running) {
        document.getElementById('progress-wrapper').style.display = 'block';
        const pct = bs.total > 0 ? (bs.progress / bs.total * 100) : 0;
        document.getElementById('progress-bar').style.width = pct + '%';
        document.getElementById('progress-text').innerText = `${bs.progress} / ${bs.total}  (${bs.current_session})`;
      } else {
        document.getElementById('progress-wrapper').style.display = 'none';
      }
    }
  } catch(e) {}
}, 800);

// ============================================================
// BOOT
// ============================================================
initGraph();
initSparkline();
</script>
</body>
</html>
"""

# =============================================
# Flask Routes
# =============================================

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/state')
def get_state():
    memory = get_memory()

    last_msg = CURRENT_CONVERSATION[-1] if CURRENT_CONVERSATION else None

    all_facts = list(memory.facts.values())
    current_facts = [f.to_dict() for f in all_facts if f.is_current]

    sorted_facts = sorted(all_facts, key=lambda x: x.extraction_time, reverse=True)
    new_facts = [f.to_dict() for f in sorted_facts[:5]]

    temporal = [
        {
            "subject": s.subject,
            "description": s.description,
            "duration_text": s.calculate_duration_from_reference()
        }
        for s in memory.temporal_states.values()
    ]

    reasoning_logs = getattr(memory, 'reasoning_logs', [])

    return jsonify({
        "last_message": last_msg,
        "facts": current_facts,
        "new_facts": new_facts,
        "temporal": temporal,
        "stats": memory.stats(),
        "reasoning_logs": reasoning_logs,
        "benchmark_status": BENCHMARK_STATUS,
    })


@app.route('/api/reset', methods=['POST'])
def reset():
    global CURRENT_CONVERSATION, BENCHMARK_STATUS, FACT_HISTORY
    memory = get_memory()
    memory.clear()
    CURRENT_CONVERSATION = []
    BENCHMARK_STATUS = {"running": False, "progress": 0, "total": 0, "current_session": ""}
    FACT_HISTORY = []
    return jsonify({"status": "cleared"})


@app.route('/api/start_benchmark', methods=['POST'])
def start_benchmark_endpoint():
    Thread(target=run_benchmark_feeder, daemon=True).start()
    return jsonify({"status": "started"})


# =============================================
# Benchmark Feeder
# =============================================

def run_benchmark_feeder():
    global BENCHMARK_STATUS

    data_path = "benchmarks/longmemeval/data/longmemeval_s_cleaned.json"
    if not os.path.exists(data_path):
        print(f"[ERROR] Data not found at {data_path}")
        return

    with open(data_path, 'r') as f:
        data = json.load(f)

    # Process first 3 entries (each has many sessions)
    entries = data[:3]

    memory = get_memory()

    for entry_idx, entry in enumerate(entries):
        haystack_sessions = entry.get('haystack_sessions', [])
        haystack_dates = entry.get('haystack_dates', [])

        turns = []
        for session_idx, session in enumerate(haystack_sessions):
            date = haystack_dates[session_idx] if session_idx < len(haystack_dates) else "Unknown Date"
            for turn in session:
                role = turn.get('role')
                content = turn.get('content')
                speaker = "User" if role == "user" else "Assistant"
                turns.append({'speaker': speaker, 'text': content, 'date': date, 'session': f"E{entry_idx}_S{session_idx}"})

        BENCHMARK_STATUS = {
            "running": True,
            "progress": 0,
            "total": len(turns),
            "current_session": f"Entry {entry_idx + 1}/{len(entries)}"
        }

        for i, turn in enumerate(turns):
            BENCHMARK_STATUS["progress"] = i + 1

            msg_obj = {
                "id": len(CURRENT_CONVERSATION),
                "speaker": turn['speaker'],
                "text": turn['text'],
                "date": turn['date']
            }
            CURRENT_CONVERSATION.append(msg_obj)

            memory.add_conversation_turn(
                speaker=turn['speaker'],
                text=turn['text'],
                date=turn['date'],
                session_id=turn['session'],
            )

            time.sleep(1.0)

    BENCHMARK_STATUS = {"running": False, "progress": 0, "total": 0, "current_session": "Done"}


# =============================================
# Main
# =============================================

def open_browser():
    time.sleep(1.5)
    webbrowser.open('http://127.0.0.1:5003')

def main():
    print("=" * 50)
    print("  LongMemEval Enhanced Visualizer (V4)")
    print("  http://127.0.0.1:5003")
    print("=" * 50)
    Thread(target=open_browser, daemon=True).start()
    app.run(host='0.0.0.0', port=5003, debug=False)

if __name__ == "__main__":
    main()
