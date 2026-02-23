"""
Enhanced Benchmark Visualization UI.

A specialized dashboard to visualize the memory system's internal state
during benchmark runs with improved aesthetics and larger graph display.
Shows real-time:
- Conversation flow
- Fact extraction stream
- Knowledge Graph growth (ENLARGED)
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

# Import from parent directory
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_memory.memory_v4.memory_store import MemoryStoreV4
from llm_memory.memory_v4.retrieval import create_retriever

# Initialize Flask app
app = Flask(__name__)

# Global state
MEMORY = None
MEMORY_PATH = "./benchmark_viz_enhanced_memory"
CURRENT_CONVERSATION = []
REASONING_LOGS = []

def get_memory():
    global MEMORY
    if MEMORY is None:
        MEMORY = MemoryStoreV4(
            user_id="benchmark_viz_enhanced",
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
    <title>Memory V4 Benchmark Visualizer - Enhanced</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <script src="https://unpkg.com/lucide@latest"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Inter:wght@300;400;500;600;700&display=swap');
        
        :root {
            --primary: #6366f1;
            --primary-light: #818cf8;
            --accent-blue: #38bdf8;
            --accent-purple: #c084fc;
            --accent-pink: #f472b6;
            --accent-orange: #fb923c;
            --accent-green: #4ade80;
            --bg-dark: #0f0f1a;
            --bg-card: #1a1a2e;
            --bg-card-hover: #232340;
            --border: #2a2a45;
        }
        
        * {
            box-sizing: border-box;
        }
        
        body { 
            font-family: 'Inter', sans-serif; 
            background: var(--bg-dark);
            background-image: 
                radial-gradient(ellipse at 10% 20%, rgba(99, 102, 241, 0.08) 0%, transparent 50%),
                radial-gradient(ellipse at 90% 80%, rgba(192, 132, 252, 0.06) 0%, transparent 50%);
        }
        
        .mono { font-family: 'JetBrains Mono', monospace; }
        
        .scrollbar-thin::-webkit-scrollbar {
            width: 6px;
        }
        .scrollbar-thin::-webkit-scrollbar-track {
            background: transparent;
        }
        .scrollbar-thin::-webkit-scrollbar-thumb {
            background: rgba(99, 102, 241, 0.3);
            border-radius: 3px;
        }
        .scrollbar-thin::-webkit-scrollbar-thumb:hover {
            background: rgba(99, 102, 241, 0.5);
        }
        
        @keyframes fadeIn { 
            from { opacity: 0; transform: translateY(10px); } 
            to { opacity: 1; transform: translateY(0); } 
        }
        @keyframes slideIn { 
            from { opacity: 0; transform: translateX(-20px); } 
            to { opacity: 1; transform: translateX(0); } 
        }
        @keyframes pulse-glow {
            0%, 100% { box-shadow: 0 0 20px rgba(99, 102, 241, 0.3); }
            50% { box-shadow: 0 0 40px rgba(99, 102, 241, 0.5); }
        }
        @keyframes float {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-5px); }
        }
        
        .fade-in { animation: fadeIn 0.4s ease-out forwards; }
        .slide-in { animation: slideIn 0.3s ease-out forwards; }
        .pulse-glow { animation: pulse-glow 2s ease-in-out infinite; }
        .float { animation: float 3s ease-in-out infinite; }
        
        .glass-card {
            background: linear-gradient(135deg, rgba(26, 26, 46, 0.9) 0%, rgba(26, 26, 46, 0.7) 100%);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(99, 102, 241, 0.15);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }
        
        .glass-card-hover:hover {
            border-color: rgba(99, 102, 241, 0.3);
            box-shadow: 0 12px 40px rgba(99, 102, 241, 0.15);
            transform: translateY(-2px);
            transition: all 0.3s ease;
        }
        
        .gradient-border {
            position: relative;
        }
        .gradient-border::before {
            content: '';
            position: absolute;
            inset: 0;
            border-radius: inherit;
            padding: 1px;
            background: linear-gradient(135deg, var(--primary), var(--accent-purple));
            -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
            -webkit-mask-composite: xor;
            mask-composite: exclude;
            pointer-events: none;
        }
        
        .stat-card {
            background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(192, 132, 252, 0.05) 100%);
            border: 1px solid rgba(99, 102, 241, 0.2);
        }
        
        .btn-primary {
            background: linear-gradient(135deg, var(--primary) 0%, var(--accent-purple) 100%);
            box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3);
            transition: all 0.3s ease;
        }
        .btn-primary:hover {
            box-shadow: 0 6px 25px rgba(99, 102, 241, 0.5);
            transform: translateY(-2px);
        }
        
        .btn-danger {
            background: linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(239, 68, 68, 0.1) 100%);
            border: 1px solid rgba(239, 68, 68, 0.3);
        }
        .btn-danger:hover {
            background: linear-gradient(135deg, rgba(239, 68, 68, 0.3) 0%, rgba(239, 68, 68, 0.2) 100%);
            border-color: rgba(239, 68, 68, 0.5);
        }
        
        .section-header {
            background: linear-gradient(90deg, rgba(99, 102, 241, 0.1) 0%, transparent 100%);
            border-bottom: 1px solid rgba(99, 102, 241, 0.2);
        }
        
        .message-bubble {
            background: linear-gradient(135deg, rgba(30, 30, 55, 0.9) 0%, rgba(26, 26, 46, 0.8) 100%);
            border: 1px solid rgba(99, 102, 241, 0.15);
            transition: all 0.2s ease;
        }
        .message-bubble:hover {
            border-color: rgba(99, 102, 241, 0.3);
            transform: translateX(4px);
        }
        
        .fact-pill {
            background: linear-gradient(135deg, rgba(30, 30, 55, 0.8) 0%, rgba(26, 26, 46, 0.6) 100%);
            border: 1px solid rgba(99, 102, 241, 0.1);
        }
        
        .temporal-card {
            background: linear-gradient(135deg, rgba(251, 146, 60, 0.1) 0%, transparent 100%);
            border-left: 3px solid var(--accent-orange);
        }
        
        #network-container {
            background: radial-gradient(ellipse at center, rgba(99, 102, 241, 0.05) 0%, transparent 70%);
        }
        
        .graph-placeholder {
            background: linear-gradient(135deg, rgba(99, 102, 241, 0.05) 0%, rgba(192, 132, 252, 0.03) 100%);
        }
    </style>
</head>
<body class="text-gray-100 h-screen flex flex-col overflow-hidden">

    <!-- Header -->
    <header class="px-6 py-4 border-b border-gray-800/50 flex items-center justify-between glass-card">
        <div class="flex items-center gap-4">
            <div class="w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-500 to-purple-500 flex items-center justify-center float">
                <i data-lucide="brain" class="w-5 h-5 text-white"></i>
            </div>
            <div>
                <h1 class="text-lg font-bold bg-gradient-to-r from-indigo-400 to-purple-400 bg-clip-text text-transparent">
                    Memory V4 Benchmark Visualizer
                </h1>
                <p class="text-xs text-gray-500">Real-time knowledge extraction & graph visualization</p>
            </div>
        </div>
        <div class="flex gap-3">
            <button onclick="startBenchmark()" class="btn-primary text-white px-5 py-2.5 rounded-xl text-sm font-medium flex items-center gap-2">
                <i data-lucide="play" class="w-4 h-4"></i> Start Benchmark
            </button>
            <button onclick="resetSystem()" class="btn-danger text-red-400 px-4 py-2.5 rounded-xl text-sm font-medium flex items-center gap-2">
                <i data-lucide="rotate-ccw" class="w-4 h-4"></i> Reset
            </button>
        </div>
    </header>

    <div class="flex flex-1 overflow-hidden">
        <!-- Left Sidebar: Conversation Stream -->
        <div class="w-80 border-r border-gray-800/50 flex flex-col glass-card">
            <div class="section-header p-4">
                <h2 class="text-sm font-bold text-gray-300 uppercase tracking-wider flex items-center gap-2">
                    <i data-lucide="message-square" class="w-4 h-4 text-blue-400"></i>
                    Conversation Stream
                </h2>
            </div>
            <div id="conversation-log" class="flex-1 overflow-y-auto p-4 space-y-3 scrollbar-thin">
                <div class="text-center text-gray-600 text-sm py-8">
                    <i data-lucide="message-circle" class="w-8 h-8 mx-auto mb-2 opacity-30"></i>
                    <p>No messages yet</p>
                    <p class="text-xs">Start benchmark to see conversation</p>
                </div>
            </div>
        </div>

        <!-- Main Area: Large Graph -->
        <div class="flex-1 flex flex-col">
            <div class="section-header p-4 flex justify-between items-center">
                <h2 class="text-sm font-bold text-gray-300 uppercase tracking-wider flex items-center gap-2">
                    <i data-lucide="network" class="w-4 h-4 text-purple-400"></i>
                    Knowledge Graph
                </h2>
                <div class="flex items-center gap-4">
                    <span class="text-xs text-gray-500 mono" id="node-count">0 Nodes</span>
                    <span class="text-xs text-gray-500 mono" id="edge-count">0 Edges</span>
                </div>
            </div>
            <div id="network-container" class="flex-1 relative">
                <div id="graph-placeholder" class="absolute inset-0 flex flex-col items-center justify-center text-gray-600 pointer-events-none graph-placeholder">
                    <div class="w-20 h-20 rounded-2xl bg-gradient-to-br from-indigo-500/20 to-purple-500/20 flex items-center justify-center mb-4">
                        <i data-lucide="git-branch" class="w-10 h-10 text-indigo-400/50"></i>
                    </div>
                    <p class="text-lg font-medium text-gray-500">Knowledge Graph</p>
                    <p class="text-sm text-gray-600">Start benchmark to visualize entity relationships</p>
                </div>
            </div>
        </div>

        <!-- Right Sidebar: Facts, Temporal, Stats -->
        <div class="w-96 border-l border-gray-800/50 flex flex-col glass-card">
            <!-- Extracted Facts -->
            <div class="h-1/3 flex flex-col border-b border-gray-800/50">
                <div class="section-header p-4">
                    <h2 class="text-sm font-bold text-gray-300 uppercase tracking-wider flex items-center gap-2">
                        <i data-lucide="database" class="w-4 h-4 text-green-400"></i>
                        Extracted Facts
                        <span class="ml-auto text-xs font-normal text-gray-500" id="fact-count">0 facts</span>
                    </h2>
                </div>
                <div id="facts-log" class="flex-1 overflow-y-auto p-3 space-y-2 scrollbar-thin">
                    <div class="text-center text-gray-600 text-xs py-4">
                        <p>Facts will appear here</p>
                    </div>
                </div>
            </div>
            
            <!-- Temporal State -->
            <div class="h-1/3 flex flex-col border-b border-gray-800/50">
                <div class="section-header p-4">
                    <h2 class="text-sm font-bold text-gray-300 uppercase tracking-wider flex items-center gap-2">
                        <i data-lucide="clock" class="w-4 h-4 text-orange-400"></i>
                        Temporal State
                    </h2>
                </div>
                <div id="temporal-log" class="flex-1 overflow-y-auto p-3 space-y-2 scrollbar-thin">
                    <div class="text-center text-gray-600 text-xs py-4">
                        <p>Temporal events will appear here</p>
                    </div>
                </div>
            </div>
            
            <!-- Stats & Reasoning -->
            <div class="h-1/3 flex flex-col">
                <div class="section-header p-4">
                    <h2 class="text-sm font-bold text-gray-300 uppercase tracking-wider flex items-center gap-2">
                        <i data-lucide="cpu" class="w-4 h-4 text-pink-400"></i>
                        Stats & Reasoning
                    </h2>
                </div>
                <div class="flex-1 p-4 space-y-4 overflow-y-auto scrollbar-thin">
                    <div class="grid grid-cols-2 gap-3">
                        <div class="stat-card p-4 rounded-xl">
                            <div class="flex items-center gap-2 mb-1">
                                <i data-lucide="file-text" class="w-4 h-4 text-indigo-400"></i>
                                <span class="text-gray-500 text-xs">Facts</span>
                            </div>
                            <div class="text-3xl font-bold bg-gradient-to-r from-indigo-400 to-purple-400 bg-clip-text text-transparent" id="stat-facts">0</div>
                        </div>
                        <div class="stat-card p-4 rounded-xl">
                            <div class="flex items-center gap-2 mb-1">
                                <i data-lucide="layers" class="w-4 h-4 text-purple-400"></i>
                                <span class="text-gray-500 text-xs">Episodes</span>
                            </div>
                            <div class="text-3xl font-bold bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent" id="stat-episodes">0</div>
                        </div>
                    </div>
                    
                    <div id="reasoning-log" class="space-y-2 text-xs mono text-gray-400">
                        <!-- Logs -->
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        lucide.createIcons();
        let network = null;
        let nodes = new vis.DataSet();
        let edges = new vis.DataSet();
        let totalFactCount = 0;

        function initGraph() {
            const container = document.getElementById('network-container');
            const data = { nodes: nodes, edges: edges };
            const options = {
                nodes: {
                    shape: 'dot',
                    size: 16,
                    font: { 
                        size: 14, 
                        color: '#e2e8f0', 
                        face: 'Inter',
                        strokeWidth: 3,
                        strokeColor: '#0f0f1a'
                    },
                    borderWidth: 2,
                    borderWidthSelected: 4,
                    shadow: {
                        enabled: true,
                        color: 'rgba(99, 102, 241, 0.3)',
                        size: 10
                    },
                    scaling: {
                        min: 12,
                        max: 30
                    }
                },
                edges: {
                    width: 2,
                    color: { 
                        color: 'rgba(99, 102, 241, 0.4)', 
                        highlight: '#818cf8',
                        hover: '#818cf8'
                    },
                    arrows: { 
                        to: { 
                            enabled: true, 
                            scaleFactor: 0.6,
                            type: 'arrow'
                        } 
                    },
                    smooth: { 
                        type: 'continuous',
                        roundness: 0.5
                    },
                    font: {
                        size: 10,
                        color: '#9ca3af',
                        strokeWidth: 2,
                        strokeColor: '#0f0f1a',
                        align: 'middle'
                    },
                    hoverWidth: 3
                },
                physics: {
                    enabled: true,
                    stabilization: {
                        enabled: true,
                        iterations: 100,
                        updateInterval: 25
                    },
                    barnesHut: {
                        gravitationalConstant: -3000,
                        centralGravity: 0.2,
                        springLength: 150,
                        springConstant: 0.03,
                        damping: 0.15,
                        avoidOverlap: 0.5
                    }
                },
                interaction: { 
                    hover: true, 
                    tooltipDelay: 100,
                    zoomView: true,
                    dragView: true,
                    hideEdgesOnDrag: false,
                    hideEdgesOnZoom: false
                },
                layout: {
                    improvedLayout: true
                }
            };
            network = new vis.Network(container, data, options);
            
            // Event handlers
            network.on("hoverNode", function(params) {
                document.body.style.cursor = 'pointer';
            });
            network.on("blurNode", function(params) {
                document.body.style.cursor = 'default';
            });
        }

        async function startBenchmark() {
            document.getElementById('graph-placeholder').style.display = 'none';
            await fetch('/api/start_benchmark', { method: 'POST' });
        }

        async function resetSystem() {
            await fetch('/api/reset', { method: 'POST' });
            nodes.clear();
            edges.clear();
            totalFactCount = 0;
            document.getElementById('conversation-log').innerHTML = `
                <div class="text-center text-gray-600 text-sm py-8">
                    <i data-lucide="message-circle" class="w-8 h-8 mx-auto mb-2 opacity-30"></i>
                    <p>No messages yet</p>
                    <p class="text-xs">Start benchmark to see conversation</p>
                </div>
            `;
            document.getElementById('facts-log').innerHTML = `
                <div class="text-center text-gray-600 text-xs py-4">
                    <p>Facts will appear here</p>
                </div>
            `;
            document.getElementById('temporal-log').innerHTML = `
                <div class="text-center text-gray-600 text-xs py-4">
                    <p>Temporal events will appear here</p>
                </div>
            `;
            document.getElementById('reasoning-log').innerHTML = '';
            document.getElementById('graph-placeholder').style.display = 'flex';
            document.getElementById('fact-count').innerText = '0 facts';
            updateStats({total_facts: 0, total_episodes: 0});
            lucide.createIcons();
        }

        function getNodeColor(factType) {
            const colors = {
                'preference': { background: '#f472b6', border: '#ec4899' },
                'temporal': { background: '#fb923c', border: '#f97316' },
                'state_change': { background: '#4ade80', border: '#22c55e' },
                'relationship': { background: '#38bdf8', border: '#0ea5e9' },
                'default': { background: '#818cf8', border: '#6366f1' }
            };
            return colors[factType] || colors.default;
        }

        function updateGraph(facts) {
            const existingNodeIds = new Set(nodes.getIds());
            const existingEdgeIds = new Set(edges.getIds());

            facts.forEach(f => {
                const color = getNodeColor(f.fact_type);
                
                // Nodes
                [f.subject, f.object].forEach(label => {
                    if (!existingNodeIds.has(label) && label) {
                        try {
                            nodes.add({ 
                                id: label, 
                                label: label.length > 18 ? label.substring(0,15)+'...' : label, 
                                title: `<div class="p-2"><strong>${label}</strong></div>`, 
                                color: color,
                                font: { color: '#fff' }
                            });
                            existingNodeIds.add(label);
                        } catch(e) {}
                    }
                });

                // Edge
                if (f.subject && f.object && f.predicate) {
                    const edgeId = `${f.subject}-${f.predicate}-${f.object}`;
                    if (!existingEdgeIds.has(edgeId)) {
                        try {
                            edges.add({ 
                                id: edgeId, 
                                from: f.subject, 
                                to: f.object, 
                                label: f.predicate,
                                title: f.predicate
                            });
                            existingEdgeIds.add(edgeId);
                        } catch(e) {}
                    }
                }
            });
            
            document.getElementById('node-count').innerText = `${nodes.length} Nodes`;
            document.getElementById('edge-count').innerText = `${edges.length} Edges`;
            
            // Hide placeholder if we have data
            if (nodes.length > 0) {
                document.getElementById('graph-placeholder').style.display = 'none';
            }
        }

        function addMessage(msg) {
            const container = document.getElementById('conversation-log');
            
            // Remove placeholder if exists
            if (container.querySelector('.text-center')) {
                container.innerHTML = '';
            }
            
            const div = document.createElement('div');
            const isCaroline = msg.speaker === 'Caroline';
            div.className = "message-bubble rounded-xl p-3 fade-in";
            div.innerHTML = `
                <div class="flex justify-between items-start mb-2">
                    <span class="text-xs font-bold ${isCaroline ? 'text-blue-400' : 'text-purple-400'} flex items-center gap-1">
                        <i data-lucide="${isCaroline ? 'user' : 'user-2'}" class="w-3 h-3"></i>
                        ${msg.speaker}
                    </span>
                    <span class="text-[10px] text-gray-600 mono">${msg.date}</span>
                </div>
                <p class="text-sm text-gray-300 leading-relaxed">${msg.text}</p>
            `;
            container.appendChild(div);
            container.scrollTop = container.scrollHeight;
            lucide.createIcons();
        }

        function addFacts(facts) {
            const container = document.getElementById('facts-log');
            
            // Remove placeholder if exists
            if (container.querySelector('.text-center')) {
                container.innerHTML = '';
            }
            
            facts.forEach(f => {
                totalFactCount++;
                const div = document.createElement('div');
                div.className = "fact-pill p-2 rounded-lg text-xs flex gap-2 items-center fade-in";
                
                let icon = 'circle';
                let color = 'text-indigo-400';
                if (f.fact_type === 'preference') { icon = 'heart'; color = 'text-pink-400'; }
                if (f.fact_type === 'state_change') { icon = 'arrow-right-circle'; color = 'text-green-400'; }
                if (f.fact_type === 'temporal') { icon = 'clock'; color = 'text-orange-400'; }
                
                div.innerHTML = `
                    <i data-lucide="${icon}" class="w-3 h-3 ${color} flex-shrink-0"></i>
                    <div class="flex-1 truncate">
                        <span class="text-gray-200 font-medium">${f.subject}</span>
                        <span class="text-gray-500 mx-1">${f.predicate}</span>
                        <span class="text-gray-200 font-medium">${f.object}</span>
                    </div>
                `;
                container.prepend(div);
            });
            
            document.getElementById('fact-count').innerText = `${totalFactCount} facts`;
            lucide.createIcons();
        }

        function updateTemporal(states) {
            const container = document.getElementById('temporal-log');
            
            if (states.length === 0) {
                container.innerHTML = `
                    <div class="text-center text-gray-600 text-xs py-4">
                        <p>Temporal events will appear here</p>
                    </div>
                `;
                return;
            }
            
            container.innerHTML = '';
            
            states.forEach(s => {
                const div = document.createElement('div');
                div.className = "temporal-card p-3 rounded-lg";
                div.innerHTML = `
                    <div class="flex justify-between items-start">
                        <span class="text-xs font-bold text-gray-200">${s.subject}</span>
                        <span class="text-[10px] text-orange-400 font-mono bg-orange-500/10 px-2 py-0.5 rounded-full">
                            ${s.duration_text || 'Ongoing'}
                        </span>
                    </div>
                    <p class="text-xs text-gray-400 mt-1 leading-relaxed">${s.description}</p>
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
            
            logs.slice(0, 5).forEach(log => {
                const div = document.createElement('div');
                div.className = "bg-gray-800/50 p-2 rounded-lg border border-gray-700/50 fade-in";
                
                let content = `<div class="text-pink-400 font-bold mb-1 flex items-center gap-1">
                    <i data-lucide="zap" class="w-3 h-3"></i>
                    ${log.type}
                </div>`;
                
                if (log.type === 'DECOMPOSE') {
                    content += `<div class="pl-2 border-l-2 border-pink-500/30 space-y-1">`;
                    log.steps.forEach((step, i) => {
                        content += `<div class="text-gray-400">Step ${i+1}: ${step}</div>`;
                    });
                    content += `</div>`;
                } else if (log.type === 'RETRIEVE') {
                    content += `<div class="text-gray-500">Query: "${log.query}"</div>`;
                    content += `<div class="text-green-400 mt-1 flex items-center gap-1">
                        <i data-lucide="check-circle" class="w-3 h-3"></i>
                        Found ${log.count} facts
                    </div>`;
                } else {
                    content += `<div class="text-gray-400">${log.message}</div>`;
                }
                
                div.innerHTML = content;
                container.prepend(div);
            });
            
            lucide.createIcons();
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
    
    # Get reasoning logs
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
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                            "benchmarks/locomo_data/data/locomo10.json")
    if not os.path.exists(data_path):
        print("Data not found at:", data_path)
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
    webbrowser.open('http://127.0.0.1:5002')

def main():
    print("Starting Enhanced Benchmark Visualizer on port 5002...")
    Thread(target=open_browser).start()
    app.run(host='0.0.0.0', port=5002, debug=False)

if __name__ == "__main__":
    main()
