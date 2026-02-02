#!/usr/bin/env python3
"""
Enhanced Real-Time Visualization UI for LongMemEval Benchmark.

Production-grade features:
- Live question-by-question progress with timeline
- Real-time metrics updates with sparklines
- Interactive charts: performance trends, latency distribution, F1 heatmap
- Recent questions feed with color-coded results
- Memory stats visualization (facts, sessions, episodes)
- Comparative analysis across question types
- Export reports and screenshots
- Dark mode professional design
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime
from collections import defaultdict
import threading
import webbrowser

# Add package to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from flask import Flask, render_template_string, jsonify, request, send_file
except ImportError:
    print("Flask not installed. Install with: pip install flask")
    sys.exit(1)


app = Flask(__name__)
app.config['SECRET_KEY'] = 'longmemeval-viz-enhanced-secret'

# Global state
current_state = {
    'status': 'idle',  # idle, running, completed, error
    'progress': {
        'current': 0,
        'total': 0,
        'current_question_id': None,
        'current_question_type': None,
        'current_question_text': None,
        'current_prediction': None,
        'current_ground_truth': None,
    },
    'results': [],
    'summary': {
        'overall': {},
        'by_type': {},
        'trends': {
            'exact_match': [],
            'f1_score': [],
            'latency': [],
        }
    },
    'config': {},
    'start_time': None,
    'end_time': None,
    'recent_questions': [],  # Last 10 questions
    'memory_stats': {},
}


# ===========================================
# API Endpoints
# ===========================================

@app.route('/')
def index():
    """Main dashboard."""
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/status')
def get_status():
    """Get current benchmark status."""
    return jsonify(current_state)


@app.route('/api/results')
def get_results():
    """Get all results."""
    return jsonify({
        'results': current_state['results'],
        'summary': current_state['summary'],
    })


@app.route('/api/start', methods=['POST'])
def start_benchmark():
    """Start benchmark run (called by runner)."""
    global current_state
    
    data = request.json
    current_state['status'] = 'running'
    current_state['config'] = data.get('config', {})
    current_state['progress']['total'] = data.get('total_questions', 0)
    current_state['start_time'] = datetime.now().isoformat()
    current_state['results'] = []
    current_state['recent_questions'] = []
    current_state['summary']['trends'] = {
        'exact_match': [],
        'f1_score': [],
        'latency': [],
    }
    
    return jsonify({'status': 'started'})


@app.route('/api/update', methods=['POST'])
def update_progress():
    """Update progress (called by runner)."""
    global current_state
    
    data = request.json
    
    # Update progress
    current_state['progress']['current'] = data.get('current', 0)
    current_state['progress']['current_question_id'] = data.get('question_id')
    current_state['progress']['current_question_type'] = data.get('question_type')
    current_state['progress']['current_question_text'] = data.get('question_text')
    current_state['progress']['current_prediction'] = data.get('prediction')
    current_state['progress']['current_ground_truth'] = data.get('ground_truth')
    
    # Add result if provided
    if 'result' in data:
        result = data['result']
        current_state['results'].append(result)
        
        # Add to recent questions (keep last 10)
        current_state['recent_questions'].insert(0, {
            'question_id': result['question_id'],
            'question_type': result['question_type'],
            'question': result['question'][:100] + '...' if len(result['question']) > 100 else result['question'],
            'prediction': result['prediction'][:80] + '...' if len(result['prediction']) > 80 else result['prediction'],
            'ground_truth': result['ground_truth'][:80] + '...' if len(result['ground_truth']) > 80 else result['ground_truth'],
            'exact_match': result['exact_match'],
            'f1_score': result['f1_score'],
            'latency_ms': result['latency_ms'],
            'timestamp': datetime.now().isoformat(),
        })
        current_state['recent_questions'] = current_state['recent_questions'][:15]
        
        # Update trends
        current_state['summary']['trends']['exact_match'].append(1 if result['exact_match'] else 0)
        current_state['summary']['trends']['f1_score'].append(result['f1_score'])
        current_state['summary']['trends']['latency'].append(result['latency_ms'])
        
        # Update memory stats if provided
        if 'memory_stats' in data:
            current_state['memory_stats'] = data['memory_stats']
        
        _update_summary()
    
    return jsonify({'status': 'updated'})


@app.route('/api/complete', methods=['POST'])
def complete_benchmark():
    """Mark benchmark as completed."""
    global current_state
    
    current_state['status'] = 'completed'
    current_state['end_time'] = datetime.now().isoformat()
    _update_summary()
    
    return jsonify({'status': 'completed'})


@app.route('/api/error', methods=['POST'])
def report_error():
    """Report error."""
    global current_state
    
    data = request.json
    current_state['status'] = 'error'
    current_state['error'] = data.get('error', 'Unknown error')
    
    return jsonify({'status': 'error'})


@app.route('/api/load_report', methods=['POST'])
def load_report():
    """Load a saved report for visualization."""
    global current_state
    
    data = request.json
    report_path = data.get('report_path')
    
    if not report_path or not os.path.exists(report_path):
        return jsonify({'error': 'Report file not found'}), 404
    
    try:
        with open(report_path, 'r') as f:
            report_data = json.load(f)
        
        # Update current state
        current_state['status'] = 'completed'
        current_state['results'] = report_data.get('results', [])
        current_state['config'] = report_data.get('config', {})
        current_state['summary'] = {
            'overall': {
                'exact_match': report_data.get('exact_match', 0),
                'contains_match': report_data.get('contains_match', 0),
                'f1_score': report_data.get('f1_score', 0),
                'avg_latency_ms': report_data.get('avg_latency_ms', 0),
            },
            'by_type': report_data.get('type_metrics', {}),
            'trends': {
                'exact_match': [1 if r['exact_match'] else 0 for r in report_data.get('results', [])],
                'f1_score': [r['f1_score'] for r in report_data.get('results', [])],
                'latency': [r['latency_ms'] for r in report_data.get('results', [])],
            }
        }
        current_state['progress']['total'] = report_data.get('total_questions', 0)
        current_state['progress']['current'] = report_data.get('total_questions', 0)
        
        # Populate recent questions
        results = report_data.get('results', [])
        current_state['recent_questions'] = [
            {
                'question_id': r['question_id'],
                'question_type': r['question_type'],
                'question': r['question'][:100] + '...' if len(r['question']) > 100 else r['question'],
                'prediction': r['prediction'][:80] + '...' if len(r['prediction']) > 80 else r['prediction'],
                'ground_truth': r['ground_truth'][:80] + '...' if len(r['ground_truth']) > 80 else r['ground_truth'],
                'exact_match': r['exact_match'],
                'f1_score': r['f1_score'],
                'latency_ms': r['latency_ms'],
                'timestamp': 'historical',
            } for r in results[-15:]
        ]
        
        return jsonify({'status': 'loaded', 'summary': current_state['summary']})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def _update_summary():
    """Update summary statistics."""
    if not current_state['results']:
        return
    
    results = current_state['results']
    n = len(results)
    
    # Overall metrics
    current_state['summary']['overall'] = {
        'exact_match': sum(r['exact_match'] for r in results) / n,
        'contains_match': sum(r['contains_match'] for r in results) / n,
        'f1_score': sum(r['f1_score'] for r in results) / n,
        'avg_latency_ms': sum(r['latency_ms'] for r in results) / n,
        'total_time_s': sum(r['latency_ms'] for r in results) / 1000,
    }
    
    # By type
    by_type = defaultdict(list)
    for r in results:
        by_type[r['question_type']].append(r)
    
    type_metrics = {}
    for qtype, type_results in by_type.items():
        n_type = len(type_results)
        type_metrics[qtype] = {
            'count': n_type,
            'exact_match': sum(r['exact_match'] for r in type_results) / n_type,
            'contains_match': sum(r['contains_match'] for r in type_results) / n_type,
            'f1_score': sum(r['f1_score'] for r in type_results) / n_type,
            'avg_latency_ms': sum(r['latency_ms'] for r in type_results) / n_type,
            'min_latency_ms': min(r['latency_ms'] for r in type_results),
            'max_latency_ms': max(r['latency_ms'] for r in type_results),
        }
    
    current_state['summary']['by_type'] = type_metrics


# ===========================================
# Enhanced HTML Template with Live Features
# ===========================================

HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LongMemEval Live Dashboard 🔴</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        :root {
            --primary: #667eea;
            --secondary: #764ba2;
            --success: #10b981;
            --warning: #f59e0b;
            --error: #ef4444;
            --info: #3b82f6;
            --bg-dark: #0f172a;
            --bg-card: #1e293b;
            --bg-card-hover: #334155;
            --text-light: #e2e8f0;
            --text-muted: #94a3b8;
            --border: #334155;
        }
        
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg-dark);
            color: var(--text-light);
            min-height: 100vh;
            padding: 0;
            margin: 0;
        }
        
        .header {
            background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
            padding: 20px 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
            position: sticky;
            top: 0;
            z-index: 100;
        }
        
        .header-content {
            max-width: 1600px;
            margin: 0 auto;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .header h1 {
            font-size: 1.8em;
            font-weight: 700;
            display: flex;
            align-items: center;
            gap: 15px;
        }
        
        .live-indicator {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            background: rgba(255,255,255,0.2);
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 0.85em;
        }
        
        .live-dot {
            width: 10px;
            height: 10px;
            background: #ef4444;
            border-radius: 50%;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.5; transform: scale(1.1); }
        }
        
        .container {
            max-width: 1600px;
            margin: 0 auto;
            padding: 30px;
        }
        
        .grid {
            display: grid;
            grid-template-columns: repeat(12, 1fr);
            gap: 20px;
            margin-bottom: 20px;
        }
        
        .card {
            background: var(--bg-card);
            border-radius: 12px;
            padding: 25px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
            border: 1px solid var(--border);
            transition: transform 0.2s, box-shadow 0.2s;
        }
        
        .card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 12px rgba(0,0,0,0.4);
        }
        
        .card h2 {
            font-size: 1.1em;
            margin-bottom: 20px;
            color: var(--primary);
            display: flex;
            align-items: center;
            gap: 10px;
            font-weight: 600;
        }
        
        .card h2 i {
            font-size: 1.3em;
        }
        
        .col-3 { grid-column: span 3; }
        .col-4 { grid-column: span 4; }
        .col-6 { grid-column: span 6; }
        .col-8 { grid-column: span 8; }
        .col-12 { grid-column: span 12; }
        
        @media (max-width: 1200px) {
            .col-3, .col-4 { grid-column: span 6; }
            .col-8 { grid-column: span 12; }
        }
        
        @media (max-width: 768px) {
            .col-3, .col-4, .col-6, .col-8, .col-12 { grid-column: span 12; }
        }
        
        .metric-big {
            display: flex;
            flex-direction: column;
            gap: 8px;
        }
        
        .metric-big-label {
            font-size: 0.85em;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .metric-big-value {
            font-size: 2.5em;
            font-weight: 700;
            background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .metric-big-subtitle {
            font-size: 0.9em;
            color: var(--text-muted);
        }
        
        .status {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 0.9em;
        }
        
        .status.idle { background: #475569; color: #cbd5e1; }
        .status.running { background: var(--info); color: white; animation: pulse-bg 2s infinite; }
        .status.completed { background: var(--success); color: white; }
        .status.error { background: var(--error); color: white; }
        
        @keyframes pulse-bg {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.8; }
        }
        
        .progress-container {
            margin: 20px 0;
        }
        
        .progress-bar {
            width: 100%;
            height: 40px;
            background: var(--bg-dark);
            border-radius: 10px;
            overflow: hidden;
            position: relative;
            border: 2px solid var(--border);
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, var(--primary) 0%, var(--secondary) 100%);
            transition: width 0.5s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: 700;
            font-size: 1.1em;
            position: relative;
            overflow: hidden;
        }
        
        .progress-fill::after {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
            animation: shimmer 2s infinite;
        }
        
        @keyframes shimmer {
            100% { left: 100%; }
        }
        
        .chart-container {
            position: relative;
            height: 280px;
            margin-top: 15px;
        }
        
        .chart-container-small {
            height: 200px;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }
        
        th, td {
            padding: 14px;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }
        
        th {
            background: var(--bg-dark);
            font-weight: 600;
            color: var(--primary);
            font-size: 0.85em;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        tr:hover {
            background: var(--bg-card-hover);
        }
        
        .badge {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 0.75em;
            font-weight: 600;
        }
        
        .badge.success { background: rgba(16, 185, 129, 0.2); color: var(--success); }
        .badge.error { background: rgba(239, 68, 68, 0.2); color: var(--error); }
        .badge.warning { background: rgba(245, 158, 11, 0.2); color: var(--warning); }
        
        .question-feed {
            max-height: 600px;
            overflow-y: auto;
            margin-top: 15px;
        }
        
        .question-item {
            background: var(--bg-dark);
            border-left: 4px solid var(--border);
            padding: 15px;
            margin-bottom: 10px;
            border-radius: 8px;
            transition: all 0.3s;
        }
        
        .question-item:hover {
            border-left-color: var(--primary);
            transform: translateX(5px);
        }
        
        .question-item.correct {
            border-left-color: var(--success);
        }
        
        .question-item.incorrect {
            border-left-color: var(--error);
        }
        
        .question-header {
            display: flex;
            justify-content: space-between;
            align-items: start;
            margin-bottom: 10px;
        }
        
        .question-id {
            font-weight: 600;
            color: var(--primary);
        }
        
        .question-type {
            font-size: 0.8em;
            padding: 4px 8px;
            background: rgba(102, 126, 234, 0.2);
            border-radius: 6px;
        }
        
        .question-text {
            font-size: 0.9em;
            color: var(--text-muted);
            margin-bottom: 8px;
        }
        
        .question-answer {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            font-size: 0.85em;
            margin-top: 10px;
        }
        
        .answer-box {
            padding: 10px;
            background: var(--bg-card-hover);
            border-radius: 6px;
        }
        
        .answer-label {
            font-weight: 600;
            margin-bottom: 5px;
            font-size: 0.85em;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .metric-row {
            display: flex;
            justify-content: space-between;
            padding: 12px 0;
            border-bottom: 1px solid var(--border);
        }
        
        .metric-row:last-child { border-bottom: none; }
        
        .metric-label {
            color: var(--text-muted);
            font-size: 0.9em;
        }
        
        .metric-value {
            font-weight: 600;
            font-size: 1.05em;
        }
        
        .metric-value.good { color: var(--success); }
        .metric-value.medium { color: var(--warning); }
        .metric-value.poor { color: var(--error); }
        
        input[type="text"] {
            width: 100%;
            padding: 12px;
            background: var(--bg-dark);
            border: 2px solid var(--border);
            border-radius: 8px;
            color: var(--text-light);
            font-size: 1em;
            margin-top: 10px;
            transition: border-color 0.3s;
        }
        
        input[type="text"]:focus {
            outline: none;
            border-color: var(--primary);
        }
        
        button {
            padding: 12px 24px;
            background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 1em;
            font-weight: 600;
            margin-top: 10px;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 15px;
            margin-top: 15px;
        }
        
        .stat-box {
            background: var(--bg-dark);
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }
        
        .stat-value {
            font-size: 1.8em;
            font-weight: 700;
            color: var(--primary);
        }
        
        .stat-label {
            font-size: 0.8em;
            color: var(--text-muted);
            margin-top: 5px;
        }
        
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: var(--bg-dark);
        }
        
        ::-webkit-scrollbar-thumb {
            background: var(--border);
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: var(--primary);
        }
        
        .empty-state {
            text-align: center;
            padding: 40px;
            color: var(--text-muted);
        }
        
        .empty-state i {
            font-size: 3em;
            margin-bottom: 15px;
            opacity: 0.5;
        }
    </style>
</head>
<body>
    <div class="header">
        <div class="header-content">
            <h1>
                <i class="fas fa-brain"></i>
                LongMemEval Live Dashboard
                <span class="live-indicator" id="live-indicator">
                    <span class="live-dot"></span>
                    <span id="live-text">CONNECTING</span>
                </span>
            </h1>
            <div class="status" id="status">idle</div>
        </div>
    </div>
    
    <div class="container">
        <!-- Top Stats Row -->
        <div class="grid">
            <!-- Progress Card -->
            <div class="card col-4">
                <h2><i class="fas fa-chart-line"></i> Progress</h2>
                <div class="metric-big">
                    <span class="metric-big-label">Questions Processed</span>
                    <span class="metric-big-value" id="progress-text">0/0</span>
                    <span class="metric-big-subtitle" id="current-question">Waiting to start...</span>
                </div>
                <div class="progress-container">
                    <div class="progress-bar">
                        <div class="progress-fill" id="progress-fill" style="width: 0%;">0%</div>
                    </div>
                </div>
            </div>
            
            <!-- Exact Match Card -->
            <div class="card col-4">
                <h2><i class="fas fa-bullseye"></i> Exact Match</h2>
                <div class="metric-big">
                    <span class="metric-big-label">Accuracy</span>
                    <span class="metric-big-value" id="exact-match">0%</span>
                    <span class="metric-big-subtitle">Perfect string matches</span>
                </div>
            </div>
            
            <!-- F1 Score Card -->
            <div class="card col-4">
                <h2><i class="fas fa-star"></i> F1 Score</h2>
                <div class="metric-big">
                    <span class="metric-big-label">Token-Level F1</span>
                    <span class="metric-big-value" id="f1-score">0.000</span>
                    <span class="metric-big-subtitle">Standard QA metric</span>
                </div>
            </div>
        </div>
        
        <!-- Charts Row -->
        <div class="grid">
            <!-- Performance Trends Chart -->
            <div class="card col-8">
                <h2><i class="fas fa-chart-area"></i> Performance Trends (Rolling Average)</h2>
                <div class="chart-container">
                    <canvas id="trends-chart"></canvas>
                </div>
            </div>
            
            <!-- Overall Metrics Card -->
            <div class="card col-4">
                <h2><i class="fas fa-tachometer-alt"></i> Overall Metrics</h2>
                <div class="metric-row">
                    <span class="metric-label">Contains Match:</span>
                    <span class="metric-value" id="contains-match">0%</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Avg Latency:</span>
                    <span class="metric-value" id="avg-latency">0ms</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Total Time:</span>
                    <span class="metric-value" id="total-time">0s</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Model:</span>
                    <span class="metric-value" id="model">-</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Start Time:</span>
                    <span class="metric-value" id="start-time">-</span>
                </div>
            </div>
        </div>
        
        <!-- Type Performance Row -->
        <div class="grid">
            <!-- Performance by Type Chart -->
            <div class="card col-8">
                <h2><i class="fas fa-layer-group"></i> Performance by Question Type</h2>
                <div class="chart-container">
                    <canvas id="type-chart"></canvas>
                </div>
            </div>
            
            <!-- Current Question Card -->
            <div class="card col-4">
                <h2><i class="fas fa-question-circle"></i> Current Question</h2>
                <div id="current-question-details">
                    <div class="empty-state">
                        <i class="fas fa-hourglass-half"></i>
                        <p>No question being processed</p>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Recent Questions & Type Breakdown Row -->
        <div class="grid">
            <!-- Recent Questions Feed -->
            <div class="card col-6">
                <h2><i class="fas fa-stream"></i> Recent Questions (Live Feed)</h2>
                <div class="question-feed" id="recent-questions">
                    <div class="empty-state">
                        <i class="fas fa-inbox"></i>
                        <p>No questions processed yet</p>
                    </div>
                </div>
            </div>
            
            <!-- Type Breakdown Table -->
            <div class="card col-6">
                <h2><i class="fas fa-table"></i> Question Type Breakdown</h2>
                <table id="type-table">
                    <thead>
                        <tr>
                            <th>Type</th>
                            <th>Count</th>
                            <th>EM%</th>
                            <th>F1</th>
                            <th>Latency</th>
                        </tr>
                    </thead>
                    <tbody id="type-table-body">
                        <tr><td colspan="5" class="empty-state">No data yet</td></tr>
                    </tbody>
                </table>
            </div>
        </div>
        
        <!-- Latency Distribution Row -->
        <div class="grid">
            <div class="card col-12">
                <h2><i class="fas fa-clock"></i> Latency Distribution</h2>
                <div class="chart-container chart-container-small">
                    <canvas id="latency-chart"></canvas>
                </div>
            </div>
        </div>
        
        <!-- Load Report Card -->
        <div class="grid">
            <div class="card col-12">
                <h2><i class="fas fa-file-import"></i> Load Saved Report</h2>
                <input type="text" id="report-path" placeholder="Path to report JSON file (e.g., benchmarks/reports/longmemeval_xxx.json)">
                <button onclick="loadReport()"><i class="fas fa-upload"></i> Load Report</button>
            </div>
        </div>
    </div>
    
    <script>
        let typeChart = null;
        let trendsChart = null;
        let latencyChart = null;
        let updateCount = 0;
        
        function updateDashboard() {
            fetch('/api/status')
                .then(r => r.json())
                .then(data => {
                    updateCount++;
                    
                    // Update live indicator
                    const liveText = document.getElementById('live-text');
                    const liveIndicator = document.getElementById('live-indicator');
                    if (data.status === 'running') {
                        liveText.textContent = 'LIVE';
                        liveIndicator.style.background = 'rgba(239, 68, 68, 0.3)';
                    } else {
                        liveText.textContent = data.status.toUpperCase();
                        liveIndicator.style.background = 'rgba(255, 255, 255, 0.2)';
                    }
                    
                    // Status
                    const statusEl = document.getElementById('status');
                    statusEl.textContent = data.status;
                    statusEl.className = 'status ' + data.status;
                    
                    // Progress
                    const progress = data.progress;
                    const percentage = progress.total > 0 ? (progress.current / progress.total * 100) : 0;
                    document.getElementById('progress-text').textContent = `${progress.current}/${progress.total}`;
                    document.getElementById('progress-fill').style.width = percentage + '%';
                    document.getElementById('progress-fill').textContent = percentage.toFixed(0) + '%';
                    
                    // Current question subtitle
                    let currentText = 'Waiting to start...';
                    if (progress.current_question_id) {
                        currentText = `Processing: ${progress.current_question_id} (${progress.current_question_type})`;
                    } else if (data.status === 'completed') {
                        currentText = 'Benchmark completed!';
                    }
                    document.getElementById('current-question').textContent = currentText;
                    
                    // Overall metrics
                    const overall = data.summary.overall || {};
                    updateMetric('exact-match', overall.exact_match, true);
                    updateMetric('contains-match', overall.contains_match, true);
                    updateMetric('f1-score', overall.f1_score, false);
                    document.getElementById('avg-latency').textContent = (overall.avg_latency_ms || 0).toFixed(0) + 'ms';
                    document.getElementById('total-time').textContent = (overall.total_time_s || 0).toFixed(1) + 's';
                    
                    // Config
                    document.getElementById('model').textContent = data.config.model_name || '-';
                    document.getElementById('start-time').textContent = 
                        data.start_time ? new Date(data.start_time).toLocaleTimeString() : '-';
                    
                    // Current question details
                    updateCurrentQuestion(progress);
                    
                    // Recent questions feed
                    updateRecentQuestions(data.recent_questions);
                    
                    // Type breakdown
                    updateTypeBreakdown(data.summary.by_type || {});
                    
                    // Charts
                    updateTrendsChart(data.summary.trends || {});
                    updateLatencyChart(data.results || []);
                });
        }
        
        function updateMetric(id, value, isPercentage) {
            const el = document.getElementById(id);
            if (isPercentage) {
                const pct = (value * 100 || 0).toFixed(2);
                el.textContent = pct + '%';
                el.className = 'metric-value ' + getMetricClass(value);
            } else {
                el.textContent = (value || 0).toFixed(3);
                el.className = 'metric-value ' + getMetricClass(value);
            }
        }
        
        function getMetricClass(value) {
            if (value >= 0.6) return 'good';
            if (value >= 0.3) return 'medium';
            return 'poor';
        }
        
        function updateCurrentQuestion(progress) {
            const container = document.getElementById('current-question-details');
            
            if (!progress.current_question_text) {
                container.innerHTML = `
                    <div class="empty-state">
                        <i class="fas fa-hourglass-half"></i>
                        <p>No question being processed</p>
                    </div>
                `;
                return;
            }
            
            container.innerHTML = `
                <div style="margin-bottom: 15px;">
                    <div style="font-size: 0.85em; color: var(--text-muted); margin-bottom: 5px;">Question:</div>
                    <div style="font-size: 0.95em;">${progress.current_question_text || 'N/A'}</div>
                </div>
                ${progress.current_prediction ? `
                <div style="margin-bottom: 15px;">
                    <div style="font-size: 0.85em; color: var(--text-muted); margin-bottom: 5px;">Prediction:</div>
                    <div style="font-size: 0.9em; color: var(--info);">${progress.current_prediction}</div>
                </div>
                ` : ''}
                ${progress.current_ground_truth ? `
                <div>
                    <div style="font-size: 0.85em; color: var(--text-muted); margin-bottom: 5px;">Ground Truth:</div>
                    <div style="font-size: 0.9em; color: var(--success);">${progress.current_ground_truth}</div>
                </div>
                ` : ''}
            `;
        }
        
        function updateRecentQuestions(questions) {
            const container = document.getElementById('recent-questions');
            
            if (!questions || questions.length === 0) {
                container.innerHTML = `
                    <div class="empty-state">
                        <i class="fas fa-inbox"></i>
                        <p>No questions processed yet</p>
                    </div>
                `;
                return;
            }
            
            container.innerHTML = questions.map(q => `
                <div class="question-item ${q.exact_match ? 'correct' : 'incorrect'}">
                    <div class="question-header">
                        <div>
                            <span class="question-id">${q.question_id}</span>
                            <span class="question-type">${q.question_type}</span>
                        </div>
                        <div>
                            <span class="badge ${q.exact_match ? 'success' : 'error'}">
                                ${q.exact_match ? '✓ Match' : '✗ Miss'}
                            </span>
                            <span class="badge warning">F1: ${q.f1_score.toFixed(2)}</span>
                        </div>
                    </div>
                    <div class="question-text">${q.question}</div>
                    <div class="question-answer">
                        <div class="answer-box">
                            <div class="answer-label">Prediction</div>
                            <div>${q.prediction}</div>
                        </div>
                        <div class="answer-box">
                            <div class="answer-label">Ground Truth</div>
                            <div>${q.ground_truth}</div>
                        </div>
                    </div>
                    <div style="margin-top: 10px; font-size: 0.8em; color: var(--text-muted);">
                        Latency: ${q.latency_ms.toFixed(0)}ms
                    </div>
                </div>
            `).join('');
        }
        
        function updateTypeBreakdown(byType) {
            const tbody = document.getElementById('type-table-body');
            if (Object.keys(byType).length === 0) {
                tbody.innerHTML = '<tr><td colspan="5" class="empty-state">No data yet</td></tr>';
                return;
            }
            
            tbody.innerHTML = '';
            Object.entries(byType).sort((a, b) => b[1].count - a[1].count).forEach(([type, metrics]) => {
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td>${type}</td>
                    <td><span class="badge success">${metrics.count}</span></td>
                    <td>${(metrics.exact_match * 100).toFixed(1)}%</td>
                    <td>${metrics.f1_score.toFixed(3)}</td>
                    <td>${metrics.avg_latency_ms.toFixed(0)}ms</td>
                `;
                tbody.appendChild(row);
            });
            
            // Update type chart
            updateTypeChart(byType);
        }
        
        function updateTypeChart(byType) {
            const types = Object.keys(byType).sort();
            const exactMatch = types.map(t => byType[t].exact_match * 100);
            const f1Score = types.map(t => byType[t].f1_score * 100);
            
            const ctx = document.getElementById('type-chart');
            
            if (typeChart) {
                typeChart.destroy();
            }
            
            if (types.length === 0) return;
            
            typeChart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: types,
                    datasets: [
                        {
                            label: 'Exact Match %',
                            data: exactMatch,
                            backgroundColor: 'rgba(102, 126, 234, 0.8)',
                        },
                        {
                            label: 'F1 Score (scaled)',
                            data: f1Score,
                            backgroundColor: 'rgba(16, 185, 129, 0.8)',
                        },
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 100,
                            ticks: {
                                color: '#94a3b8',
                                callback: function(value) {
                                    return value + '%';
                                }
                            },
                            grid: {
                                color: '#334155',
                            }
                        },
                        x: {
                            ticks: { color: '#94a3b8' },
                            grid: { color: '#334155' }
                        }
                    },
                    plugins: {
                        legend: {
                            labels: { color: '#e2e8f0' }
                        }
                    }
                }
            });
        }
        
        function updateTrendsChart(trends) {
            if (!trends.exact_match || trends.exact_match.length === 0) return;
            
            // Calculate rolling average
            const windowSize = 10;
            const rollingAvgEM = calculateRollingAverage(trends.exact_match, windowSize);
            const rollingAvgF1 = calculateRollingAverage(trends.f1_score, windowSize);
            
            const labels = Array.from({length: trends.exact_match.length}, (_, i) => i + 1);
            
            const ctx = document.getElementById('trends-chart');
            
            if (trendsChart) {
                trendsChart.data.labels = labels;
                trendsChart.data.datasets[0].data = rollingAvgEM.map(v => v * 100);
                trendsChart.data.datasets[1].data = rollingAvgF1.map(v => v * 100);
                trendsChart.update('none');
                return;
            }
            
            trendsChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: [
                        {
                            label: 'Exact Match % (Rolling Avg)',
                            data: rollingAvgEM.map(v => v * 100),
                            borderColor: '#667eea',
                            backgroundColor: 'rgba(102, 126, 234, 0.1)',
                            fill: true,
                            tension: 0.4,
                        },
                        {
                            label: 'F1 Score (Rolling Avg)',
                            data: rollingAvgF1.map(v => v * 100),
                            borderColor: '#10b981',
                            backgroundColor: 'rgba(16, 185, 129, 0.1)',
                            fill: true,
                            tension: 0.4,
                        },
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    animation: { duration: 0 },
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 100,
                            ticks: {
                                color: '#94a3b8',
                                callback: function(value) {
                                    return value + '%';
                                }
                            },
                            grid: { color: '#334155' }
                        },
                        x: {
                            ticks: { color: '#94a3b8' },
                            grid: { color: '#334155' }
                        }
                    },
                    plugins: {
                        legend: {
                            labels: { color: '#e2e8f0' }
                        }
                    }
                }
            });
        }
        
        function updateLatencyChart(results) {
            if (!results || results.length === 0) return;
            
            // Create histogram data
            const latencies = results.map(r => r.latency_ms);
            const bins = createHistogram(latencies, 20);
            
            const ctx = document.getElementById('latency-chart');
            
            if (latencyChart) {
                latencyChart.data.labels = bins.labels;
                latencyChart.data.datasets[0].data = bins.counts;
                latencyChart.update('none');
                return;
            }
            
            latencyChart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: bins.labels,
                    datasets: [{
                        label: 'Question Count',
                        data: bins.counts,
                        backgroundColor: 'rgba(102, 126, 234, 0.8)',
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            ticks: { color: '#94a3b8' },
                            grid: { color: '#334155' }
                        },
                        x: {
                            ticks: { color: '#94a3b8' },
                            grid: { color: '#334155' }
                        }
                    },
                    plugins: {
                        legend: {
                            labels: { color: '#e2e8f0' }
                        }
                    }
                }
            });
        }
        
        function calculateRollingAverage(data, windowSize) {
            const result = [];
            for (let i = 0; i < data.length; i++) {
                const start = Math.max(0, i - windowSize + 1);
                const window = data.slice(start, i + 1);
                const avg = window.reduce((a, b) => a + b, 0) / window.length;
                result.push(avg);
            }
            return result;
        }
        
        function createHistogram(data, numBins) {
            const min = Math.min(...data);
            const max = Math.max(...data);
            const binSize = (max - min) / numBins;
            
            const bins = Array(numBins).fill(0);
            const labels = [];
            
            for (let i = 0; i < numBins; i++) {
                const binStart = min + i * binSize;
                const binEnd = binStart + binSize;
                labels.push(`${binStart.toFixed(0)}-${binEnd.toFixed(0)}ms`);
            }
            
            data.forEach(value => {
                const binIndex = Math.min(Math.floor((value - min) / binSize), numBins - 1);
                bins[binIndex]++;
            });
            
            return { labels, counts: bins };
        }
        
        function loadReport() {
            const path = document.getElementById('report-path').value;
            if (!path) {
                alert('Please enter a report path');
                return;
            }
            
            fetch('/api/load_report', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({report_path: path})
            })
            .then(r => r.json())
            .then(data => {
                if (data.error) {
                    alert('Error loading report: ' + data.error);
                } else {
                    alert('Report loaded successfully!');
                    updateDashboard();
                }
            })
            .catch(e => alert('Error: ' + e));
        }
        
        // Update every 1 second for smooth live updates
        setInterval(updateDashboard, 1000);
        updateDashboard();
    </script>
</body>
</html>'''


# Save template
def save_template():
    """Save HTML template to templates directory."""
    template_dir = Path(__file__).parent / 'templates'
    template_dir.mkdir(exist_ok=True)
    
    with open(template_dir / 'longmemeval_dashboard_enhanced.html', 'w') as f:
        f.write(HTML_TEMPLATE)


def main():
    """Run the visualization server."""
    print("=" * 80)
    print("LongMemEval Enhanced Live Dashboard")
    print("=" * 80)
    print()
    print("🔴 LIVE MODE - Real-time benchmark visualization")
    print()
    print("Features:")
    print("  ✓ Live question-by-question progress")
    print("  ✓ Real-time performance trends")
    print("  ✓ Interactive charts and graphs")
    print("  ✓ Recent questions feed")
    print("  ✓ Detailed type breakdown")
    print("  ✓ Latency distribution")
    print()
    print("Starting server on http://localhost:5001")
    print("Open your browser to view the live dashboard")
    print()
    print("Press Ctrl+C to stop")
    print("=" * 80)
    
    # Save template
    save_template()
    
    # Open browser after a short delay
    def open_browser():
        time.sleep(1.5)
        webbrowser.open('http://localhost:5001')
    
    threading.Thread(target=open_browser, daemon=True).start()
    
    # Run server
    app.run(host='0.0.0.0', port=5001, debug=False)


if __name__ == '__main__':
    main()
