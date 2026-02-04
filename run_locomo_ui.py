#!/usr/bin/env python3
"""
Run LOCOMO Benchmark with Live Visualization UI
"""

import sys
import subprocess
import time
from pathlib import Path
from flask import Flask, jsonify, render_template_string, request
import threading
import json
import os

app = Flask(__name__)

# Global state
benchmark_state = {
    'status': 'idle',
    'current_conv': 0,
    'total_convs': 0,
    'current_question': 0,
    'total_questions': 0,
    'results': [],
    'accuracy': 0.0,
    'start_time': None,
    'error': None
}

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>LOCOMO Benchmark Live Dashboard</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #fff;
            padding: 20px;
            min-height: 100vh;
        }
        .container { max-width: 1400px; margin: 0 auto; }
        h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .subtitle { opacity: 0.9; margin-bottom: 30px; font-size: 1.1em; }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .card {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            border: 1px solid rgba(255,255,255,0.18);
        }
        .card h2 {
            font-size: 1.2em;
            margin-bottom: 15px;
            opacity: 0.9;
        }
        .metric {
            font-size: 3em;
            font-weight: bold;
            margin: 10px 0;
        }
        .metric-small {
            font-size: 1.5em;
            opacity: 0.8;
        }
        .progress-bar {
            background: rgba(255,255,255,0.2);
            height: 30px;
            border-radius: 15px;
            overflow: hidden;
            margin: 15px 0;
        }
        .progress-fill {
            background: linear-gradient(90deg, #4CAF50, #45a049);
            height: 100%;
            transition: width 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
        }
        .status {
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.9em;
            font-weight: bold;
        }
        .status-running { background: #4CAF50; }
        .status-idle { background: #FF9800; }
        .status-complete { background: #2196F3; }
        .status-error { background: #f44336; }
        .chart-container {
            position: relative;
            height: 300px;
            margin-top: 20px;
        }
        .results-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }
        .results-table th,
        .results-table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        .results-table th {
            background: rgba(255,255,255,0.1);
            font-weight: bold;
        }
        .correct { color: #4CAF50; font-weight: bold; }
        .incorrect { color: #f44336; font-weight: bold; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🧠 LOCOMO Benchmark Dashboard</h1>
        <div class="subtitle">Real-time Memory System Performance Monitoring</div>
        
        <div class="grid">
            <div class="card">
                <h2>Status</h2>
                <div id="status" class="status status-idle">IDLE</div>
                <div class="metric-small" id="elapsed" style="margin-top: 15px;">--</div>
            </div>
            
            <div class="card">
                <h2>Accuracy</h2>
                <div class="metric" id="accuracy">0%</div>
                <div class="metric-small" id="correct-count">0/0 correct</div>
            </div>
            
            <div class="card">
                <h2>Progress</h2>
                <div class="metric-small" id="conv-progress">Conv 0/0</div>
                <div class="progress-bar">
                    <div id="progress-fill" class="progress-fill" style="width: 0%">0%</div>
                </div>
                <div class="metric-small" id="question-progress">Question 0/0</div>
            </div>
        </div>
        
        <div class="card">
            <h2>📊 Performance Chart</h2>
            <div class="chart-container">
                <canvas id="chart"></canvas>
            </div>
        </div>
        
        <div class="card">
            <h2>📝 Recent Results</h2>
            <table class="results-table" id="results-table">
                <thead>
                    <tr>
                        <th>Conv</th>
                        <th>Q#</th>
                        <th>Category</th>
                        <th>Question</th>
                        <th>Correct</th>
                    </tr>
                </thead>
                <tbody id="results-body">
                    <tr><td colspan="5" style="text-align: center; opacity: 0.5;">No results yet...</td></tr>
                </tbody>
            </table>
        </div>
    </div>
    
    <script>
        const ctx = document.getElementById('chart').getContext('2d');
        const chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Accuracy %',
                    data: [],
                    borderColor: '#4CAF50',
                    backgroundColor: 'rgba(76, 175, 80, 0.1)',
                    tension: 0.4,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { labels: { color: '#fff' } }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        ticks: { color: '#fff' },
                        grid: { color: 'rgba(255,255,255,0.1)' }
                    },
                    x: {
                        ticks: { color: '#fff' },
                        grid: { color: 'rgba(255,255,255,0.1)' }
                    }
                }
            }
        });
        
        function updateDashboard() {
            fetch('/api/status')
                .then(r => r.json())
                .then(data => {
                    // Status
                    const statusEl = document.getElementById('status');
                    statusEl.textContent = data.status.toUpperCase();
                    statusEl.className = 'status status-' + data.status;
                    
                    // Elapsed time
                    if (data.start_time) {
                        const elapsed = Math.floor((Date.now() - new Date(data.start_time).getTime()) / 1000);
                        const mins = Math.floor(elapsed / 60);
                        const secs = elapsed % 60;
                        document.getElementById('elapsed').textContent = `${mins}m ${secs}s`;
                    }
                    
                    // Accuracy
                    document.getElementById('accuracy').textContent = data.accuracy.toFixed(1) + '%';
                    const total = data.results.length;
                    const correct = data.results.filter(r => r.correct).length;
                    document.getElementById('correct-count').textContent = `${correct}/${total} correct`;
                    
                    // Progress
                    document.getElementById('conv-progress').textContent = 
                        `Conv ${data.current_conv}/${data.total_convs}`;
                    document.getElementById('question-progress').textContent = 
                        `Question ${data.current_question}/${data.total_questions}`;
                    
                    const progress = data.total_questions > 0 
                        ? (data.current_question / data.total_questions * 100) 
                        : 0;
                    const fillEl = document.getElementById('progress-fill');
                    fillEl.style.width = progress + '%';
                    fillEl.textContent = progress.toFixed(0) + '%';
                    
                    // Chart
                    if (data.results.length > chart.data.labels.length) {
                        const newResults = data.results.slice(chart.data.labels.length);
                        newResults.forEach((r, i) => {
                            chart.data.labels.push(`Q${chart.data.labels.length + 1}`);
                            // Calculate running accuracy
                            const resultsUpTo = data.results.slice(0, chart.data.labels.length);
                            const correctUpTo = resultsUpTo.filter(x => x.correct).length;
                            const acc = (correctUpTo / resultsUpTo.length * 100);
                            chart.data.datasets[0].data.push(acc);
                        });
                        chart.update();
                    }
                    
                    // Results table
                    const tbody = document.getElementById('results-body');
                    if (data.results.length > 0) {
                        tbody.innerHTML = data.results.slice(-10).reverse().map(r => `
                            <tr>
                                <td>${r.conv_id}</td>
                                <td>${r.question_num}</td>
                                <td>${r.category}</td>
                                <td style="max-width: 300px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">
                                    ${r.question}
                                </td>
                                <td class="${r.correct ? 'correct' : 'incorrect'}">
                                    ${r.correct ? '✓' : '✗'}
                                </td>
                            </tr>
                        `).join('');
                    }
                });
        }
        
        setInterval(updateDashboard, 1000);
        updateDashboard();
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/status')
def status():
    return jsonify(benchmark_state)

@app.route('/api/start', methods=['POST'])
def start_benchmark():
    data = request.json
    benchmark_state.update({
        'status': 'running',
        'total_convs': data.get('total_convs', 0),
        'total_questions': data.get('total_questions', 0),
        'start_time': time.time()
    })
    return jsonify({'status': 'started'})

@app.route('/api/update', methods=['POST'])
def update_progress():
    data = request.json
    benchmark_state.update(data)
    return jsonify({'status': 'updated'})

@app.route('/api/result', methods=['POST'])
def add_result():
    data = request.json
    benchmark_state['results'].append(data)
    
    # Update accuracy
    correct = sum(1 for r in benchmark_state['results'] if r.get('correct'))
    total = len(benchmark_state['results'])
    benchmark_state['accuracy'] = (correct / total * 100) if total > 0 else 0
    
    return jsonify({'status': 'added'})

@app.route('/api/complete', methods=['POST'])
def complete():
    benchmark_state['status'] = 'complete'
    return jsonify({'status': 'completed'})

def run_server():
    print("\n" + "="*80)
    print("🚀 LOCOMO Benchmark Dashboard")
    print("="*80)
    print(f"\n📊 Dashboard: http://localhost:5002")
    print(f"⚡ Status: Waiting for benchmark to start...")
    print("\n" + "="*80 + "\n")
    app.run(host='0.0.0.0', port=5002, debug=False, threaded=True)

if __name__ == '__main__':
    run_server()
