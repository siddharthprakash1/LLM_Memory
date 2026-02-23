#!/usr/bin/env python3
"""
Premium Real-Time Visualization UI for LongMemEval Benchmark.

Ultra-modern design with:
- Glassmorphism aesthetics
- Larger charts with better visibility
- Smooth animations and transitions
- Premium gradient color schemes
- Improved data visualization
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
app.config['SECRET_KEY'] = 'longmemeval-viz-premium-secret'

# Global state
current_state = {
    'status': 'idle',
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
    'recent_questions': [],
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
# Premium HTML Template
# ===========================================

HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LongMemEval Premium Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        :root {
            --primary: #6366f1;
            --primary-light: #818cf8;
            --secondary: #a855f7;
            --success: #10b981;
            --warning: #f59e0b;
            --error: #ef4444;
            --info: #3b82f6;
            --bg-dark: #0a0a1a;
            --bg-card: rgba(20, 20, 40, 0.8);
            --bg-card-solid: #14142a;
            --text-light: #f1f5f9;
            --text-muted: #94a3b8;
            --border: rgba(99, 102, 241, 0.2);
            --glow: rgba(99, 102, 241, 0.4);
        }
        
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--bg-dark);
            background-image: 
                radial-gradient(ellipse at 10% 20%, rgba(99, 102, 241, 0.15) 0%, transparent 50%),
                radial-gradient(ellipse at 90% 80%, rgba(168, 85, 247, 0.1) 0%, transparent 50%),
                radial-gradient(ellipse at 50% 50%, rgba(16, 185, 129, 0.05) 0%, transparent 70%);
            color: var(--text-light);
            min-height: 100vh;
            overflow-x: hidden;
        }
        
        .mono { font-family: 'JetBrains Mono', monospace; }
        
        /* Header */
        .header {
            background: linear-gradient(135deg, rgba(99, 102, 241, 0.9) 0%, rgba(168, 85, 247, 0.9) 100%);
            backdrop-filter: blur(20px);
            padding: 20px 40px;
            box-shadow: 0 4px 30px rgba(99, 102, 241, 0.3);
            position: sticky;
            top: 0;
            z-index: 100;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .header-content {
            max-width: 1800px;
            margin: 0 auto;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .header h1 {
            font-size: 1.6em;
            font-weight: 700;
            display: flex;
            align-items: center;
            gap: 15px;
            text-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
        }
        
        .logo-icon {
            width: 45px;
            height: 45px;
            background: rgba(255, 255, 255, 0.2);
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.4em;
            animation: float 3s ease-in-out infinite;
        }
        
        @keyframes float {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-5px); }
        }
        
        .live-badge {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            background: rgba(239, 68, 68, 0.9);
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 0.75em;
            font-weight: 600;
            letter-spacing: 1px;
            box-shadow: 0 0 20px rgba(239, 68, 68, 0.5);
        }
        
        .live-dot {
            width: 8px;
            height: 8px;
            background: white;
            border-radius: 50%;
            animation: pulse 1.5s ease-in-out infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.5; transform: scale(1.2); }
        }
        
        .status-badge {
            padding: 10px 20px;
            border-radius: 25px;
            font-weight: 600;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .status-badge.idle { background: rgba(71, 85, 105, 0.8); }
        .status-badge.running { 
            background: linear-gradient(135deg, var(--info), var(--primary)); 
            box-shadow: 0 0 20px rgba(59, 130, 246, 0.5);
            animation: pulse-shadow 2s infinite;
        }
        .status-badge.completed { 
            background: linear-gradient(135deg, var(--success), #059669); 
            box-shadow: 0 0 20px rgba(16, 185, 129, 0.5);
        }
        .status-badge.error { background: var(--error); }
        
        @keyframes pulse-shadow {
            0%, 100% { box-shadow: 0 0 20px rgba(59, 130, 246, 0.5); }
            50% { box-shadow: 0 0 40px rgba(59, 130, 246, 0.8); }
        }
        
        /* Container */
        .container {
            max-width: 1800px;
            margin: 0 auto;
            padding: 30px 40px;
        }
        
        /* Grid */
        .grid {
            display: grid;
            grid-template-columns: repeat(12, 1fr);
            gap: 24px;
            margin-bottom: 24px;
        }
        
        .col-3 { grid-column: span 3; }
        .col-4 { grid-column: span 4; }
        .col-5 { grid-column: span 5; }
        .col-6 { grid-column: span 6; }
        .col-7 { grid-column: span 7; }
        .col-8 { grid-column: span 8; }
        .col-12 { grid-column: span 12; }
        
        @media (max-width: 1400px) {
            .col-3, .col-4, .col-5 { grid-column: span 6; }
            .col-7, .col-8 { grid-column: span 12; }
        }
        
        /* Glass Cards */
        .card {
            background: var(--bg-card);
            backdrop-filter: blur(20px);
            border-radius: 20px;
            padding: 28px;
            border: 1px solid var(--border);
            box-shadow: 
                0 8px 32px rgba(0, 0, 0, 0.3),
                inset 0 1px 0 rgba(255, 255, 255, 0.05);
            transition: all 0.3s ease;
        }
        
        .card:hover {
            border-color: rgba(99, 102, 241, 0.4);
            box-shadow: 
                0 12px 40px rgba(99, 102, 241, 0.2),
                inset 0 1px 0 rgba(255, 255, 255, 0.08);
            transform: translateY(-4px);
        }
        
        .card h2 {
            font-size: 0.9em;
            margin-bottom: 20px;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 1.5px;
            display: flex;
            align-items: center;
            gap: 10px;
            font-weight: 600;
        }
        
        .card h2 i {
            color: var(--primary-light);
            font-size: 1.2em;
        }
        
        /* Big Metrics */
        .metric-big {
            display: flex;
            flex-direction: column;
            gap: 8px;
        }
        
        .metric-big-value {
            font-size: 3.5em;
            font-weight: 700;
            background: linear-gradient(135deg, var(--primary-light) 0%, var(--secondary) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            line-height: 1;
        }
        
        .metric-big-label {
            font-size: 0.95em;
            color: var(--text-muted);
        }
        
        /* Progress Bar */
        .progress-container {
            margin-top: 20px;
        }
        
        .progress-bar {
            width: 100%;
            height: 16px;
            background: rgba(15, 23, 42, 0.8);
            border-radius: 10px;
            overflow: hidden;
            position: relative;
            border: 1px solid var(--border);
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, var(--primary) 0%, var(--secondary) 100%);
            transition: width 0.5s ease;
            border-radius: 10px;
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
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.4), transparent);
            animation: shimmer 2s infinite;
        }
        
        @keyframes shimmer {
            100% { left: 100%; }
        }
        
        .progress-text {
            text-align: center;
            font-size: 0.85em;
            color: var(--text-muted);
            margin-top: 10px;
            font-weight: 500;
        }
        
        /* Charts - LARGER */
        .chart-container {
            position: relative;
            height: 400px;
            margin-top: 15px;
        }
        
        .chart-container-xl {
            height: 500px;
        }
        
        .chart-container-small {
            height: 280px;
        }
        
        /* Table Styles */
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }
        
        th, td {
            padding: 16px;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }
        
        th {
            background: rgba(99, 102, 241, 0.1);
            font-weight: 600;
            color: var(--primary-light);
            font-size: 0.8em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        tr:hover {
            background: rgba(99, 102, 241, 0.05);
        }
        
        /* Badges */
        .badge {
            display: inline-block;
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.75em;
            font-weight: 600;
        }
        
        .badge.success { 
            background: rgba(16, 185, 129, 0.2); 
            color: var(--success);
            border: 1px solid rgba(16, 185, 129, 0.3);
        }
        .badge.error { 
            background: rgba(239, 68, 68, 0.2); 
            color: var(--error);
            border: 1px solid rgba(239, 68, 68, 0.3);
        }
        .badge.warning { 
            background: rgba(245, 158, 11, 0.2); 
            color: var(--warning);
            border: 1px solid rgba(245, 158, 11, 0.3);
        }
        .badge.info { 
            background: rgba(99, 102, 241, 0.2); 
            color: var(--primary-light);
            border: 1px solid rgba(99, 102, 241, 0.3);
        }
        
        /* Question Feed */
        .question-feed {
            max-height: 500px;
            overflow-y: auto;
            margin-top: 15px;
            padding-right: 10px;
        }
        
        .question-item {
            background: rgba(15, 23, 42, 0.6);
            border-left: 4px solid var(--border);
            padding: 18px;
            margin-bottom: 12px;
            border-radius: 12px;
            transition: all 0.3s;
        }
        
        .question-item:hover {
            background: rgba(15, 23, 42, 0.8);
            transform: translateX(5px);
            border-left-color: var(--primary);
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
            margin-bottom: 12px;
        }
        
        .question-id {
            font-weight: 600;
            color: var(--primary-light);
            font-size: 0.9em;
        }
        
        .question-type {
            font-size: 0.75em;
            padding: 4px 10px;
            background: rgba(99, 102, 241, 0.2);
            border-radius: 8px;
            margin-left: 8px;
        }
        
        .question-text {
            font-size: 0.9em;
            color: var(--text-muted);
            margin-bottom: 12px;
            line-height: 1.5;
        }
        
        .question-answer {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
            font-size: 0.85em;
        }
        
        .answer-box {
            padding: 12px;
            background: rgba(99, 102, 241, 0.05);
            border-radius: 8px;
            border: 1px solid var(--border);
        }
        
        .answer-label {
            font-weight: 600;
            margin-bottom: 6px;
            font-size: 0.8em;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: var(--text-muted);
        }
        
        /* Metrics Row */
        .metric-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 14px 0;
            border-bottom: 1px solid var(--border);
        }
        
        .metric-row:last-child { border-bottom: none; }
        
        .metric-label {
            color: var(--text-muted);
            font-size: 0.9em;
        }
        
        .metric-value {
            font-weight: 600;
            font-size: 1.1em;
        }
        
        .metric-value.good { color: var(--success); }
        .metric-value.medium { color: var(--warning); }
        .metric-value.poor { color: var(--error); }
        
        /* Input & Button */
        input[type="text"] {
            width: 100%;
            padding: 14px 18px;
            background: rgba(15, 23, 42, 0.8);
            border: 2px solid var(--border);
            border-radius: 12px;
            color: var(--text-light);
            font-size: 1em;
            margin-top: 10px;
            transition: all 0.3s;
        }
        
        input[type="text"]:focus {
            outline: none;
            border-color: var(--primary);
            box-shadow: 0 0 20px rgba(99, 102, 241, 0.3);
        }
        
        button {
            padding: 14px 28px;
            background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
            color: white;
            border: none;
            border-radius: 12px;
            cursor: pointer;
            font-size: 1em;
            font-weight: 600;
            margin-top: 12px;
            transition: all 0.3s;
            box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4);
        }
        
        button:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(99, 102, 241, 0.5);
        }
        
        /* Empty State */
        .empty-state {
            text-align: center;
            padding: 50px;
            color: var(--text-muted);
        }
        
        .empty-icon {
            font-size: 3em;
            margin-bottom: 15px;
            opacity: 0.3;
        }
        
        /* Scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: transparent;
        }
        
        ::-webkit-scrollbar-thumb {
            background: var(--border);
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: var(--primary);
        }
        
        /* Stats Grid */
        .stats-mini-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 16px;
            margin-top: 20px;
        }
        
        .stat-mini {
            background: rgba(99, 102, 241, 0.1);
            border-radius: 12px;
            padding: 16px;
            text-align: center;
            border: 1px solid var(--border);
        }
        
        .stat-mini-value {
            font-size: 1.8em;
            font-weight: 700;
            color: var(--primary-light);
        }
        
        .stat-mini-label {
            font-size: 0.75em;
            color: var(--text-muted);
            margin-top: 5px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
    </style>
</head>
<body>
    <div class="header">
        <div class="header-content">
            <h1>
                <div class="logo-icon">
                    <i class="fas fa-brain"></i>
                </div>
                LongMemEval Premium
                <span class="live-badge" id="live-badge">
                    <span class="live-dot"></span>
                    <span id="live-text">CONNECTING</span>
                </span>
            </h1>
            <div class="status-badge" id="status">IDLE</div>
        </div>
    </div>
    
    <div class="container">
        <!-- Top Stats Row -->
        <div class="grid">
            <!-- Progress Card -->
            <div class="card col-3">
                <h2><i class="fas fa-tasks"></i> Progress</h2>
                <div class="metric-big">
                    <div class="metric-big-value" id="progress-value">0/0</div>
                    <div class="metric-big-label" id="current-question">Waiting to start...</div>
                </div>
                <div class="progress-container">
                    <div class="progress-bar">
                        <div class="progress-fill" id="progress-fill" style="width: 0%;"></div>
                    </div>
                    <div class="progress-text" id="progress-percent">0%</div>
                </div>
            </div>
            
            <!-- Exact Match Card -->
            <div class="card col-3">
                <h2><i class="fas fa-bullseye"></i> Exact Match</h2>
                <div class="metric-big">
                    <div class="metric-big-value" id="exact-match">0.00%</div>
                    <div class="metric-big-label">Perfect string matches</div>
                </div>
            </div>
            
            <!-- F1 Score Card -->
            <div class="card col-3">
                <h2><i class="fas fa-star"></i> F1 Score</h2>
                <div class="metric-big">
                    <div class="metric-big-value" id="f1-score">0.000</div>
                    <div class="metric-big-label">Token-level accuracy</div>
                </div>
            </div>
            
            <!-- Stats Card -->
            <div class="card col-3">
                <h2><i class="fas fa-clock"></i> Performance</h2>
                <div class="metric-row">
                    <span class="metric-label">Contains Match</span>
                    <span class="metric-value" id="contains-match">0%</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Avg Latency</span>
                    <span class="metric-value" id="avg-latency">0ms</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Total Time</span>
                    <span class="metric-value" id="total-time">0s</span>
                </div>
            </div>
        </div>
        
        <!-- LARGE Performance Trends Chart -->
        <div class="grid">
            <div class="card col-12">
                <h2><i class="fas fa-chart-area"></i> Real-Time Performance Trends</h2>
                <div class="chart-container chart-container-xl">
                    <canvas id="trends-chart"></canvas>
                </div>
            </div>
        </div>
        
        <!-- Type Performance & Questions -->
        <div class="grid">
            <!-- Type Performance Chart -->
            <div class="card col-7">
                <h2><i class="fas fa-layer-group"></i> Performance by Question Type</h2>
                <div class="chart-container">
                    <canvas id="type-chart"></canvas>
                </div>
            </div>
            
            <!-- Recent Questions Feed -->
            <div class="card col-5">
                <h2><i class="fas fa-stream"></i> Recent Questions</h2>
                <div class="question-feed" id="recent-questions">
                    <div class="empty-state">
                        <div class="empty-icon"><i class="fas fa-inbox"></i></div>
                        <p>No questions processed yet</p>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Latency & Type Breakdown -->
        <div class="grid">
            <!-- Latency Distribution -->
            <div class="card col-6">
                <h2><i class="fas fa-tachometer-alt"></i> Latency Distribution</h2>
                <div class="chart-container-small">
                    <canvas id="latency-chart"></canvas>
                </div>
            </div>
            
            <!-- Type Breakdown Table -->
            <div class="card col-6">
                <h2><i class="fas fa-table"></i> Question Type Breakdown</h2>
                <table>
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
        
        <!-- Load Report -->
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
        
        const chartColors = {
            primary: 'rgba(99, 102, 241, 0.8)',
            primaryLight: 'rgba(129, 140, 248, 0.8)',
            secondary: 'rgba(168, 85, 247, 0.8)',
            success: 'rgba(16, 185, 129, 0.8)',
            warning: 'rgba(245, 158, 11, 0.8)',
            grid: 'rgba(99, 102, 241, 0.1)',
            text: '#94a3b8'
        };
        
        function updateDashboard() {
            fetch('/api/status')
                .then(r => r.json())
                .then(data => {
                    // Live indicator
                    const liveText = document.getElementById('live-text');
                    const liveBadge = document.getElementById('live-badge');
                    if (data.status === 'running') {
                        liveText.textContent = 'LIVE';
                        liveBadge.style.background = 'rgba(239, 68, 68, 0.9)';
                    } else {
                        liveText.textContent = data.status.toUpperCase();
                        liveBadge.style.background = 'rgba(99, 102, 241, 0.8)';
                    }
                    
                    // Status badge
                    const statusEl = document.getElementById('status');
                    statusEl.textContent = data.status.toUpperCase();
                    statusEl.className = 'status-badge ' + data.status;
                    
                    // Progress
                    const progress = data.progress;
                    const percentage = progress.total > 0 ? (progress.current / progress.total * 100) : 0;
                    document.getElementById('progress-value').textContent = `${progress.current}/${progress.total}`;
                    document.getElementById('progress-fill').style.width = percentage + '%';
                    document.getElementById('progress-percent').textContent = percentage.toFixed(1) + '% Complete';
                    
                    let currentText = 'Waiting to start...';
                    if (progress.current_question_id) {
                        currentText = `Processing: ${progress.current_question_type}`;
                    } else if (data.status === 'completed') {
                        currentText = 'Benchmark completed!';
                    }
                    document.getElementById('current-question').textContent = currentText;
                    
                    // Overall metrics
                    const overall = data.summary.overall || {};
                    document.getElementById('exact-match').textContent = ((overall.exact_match || 0) * 100).toFixed(2) + '%';
                    document.getElementById('f1-score').textContent = (overall.f1_score || 0).toFixed(3);
                    document.getElementById('contains-match').textContent = ((overall.contains_match || 0) * 100).toFixed(1) + '%';
                    document.getElementById('avg-latency').textContent = (overall.avg_latency_ms || 0).toFixed(0) + 'ms';
                    document.getElementById('total-time').textContent = (overall.total_time_s || 0).toFixed(1) + 's';
                    
                    // Recent questions
                    updateRecentQuestions(data.recent_questions);
                    
                    // Type breakdown
                    updateTypeBreakdown(data.summary.by_type || {});
                    
                    // Charts
                    updateTrendsChart(data.summary.trends || {});
                    updateLatencyChart(data.results || []);
                });
        }
        
        function updateRecentQuestions(questions) {
            const container = document.getElementById('recent-questions');
            
            if (!questions || questions.length === 0) {
                container.innerHTML = `
                    <div class="empty-state">
                        <div class="empty-icon"><i class="fas fa-inbox"></i></div>
                        <p>No questions processed yet</p>
                    </div>
                `;
                return;
            }
            
            container.innerHTML = questions.slice(0, 8).map(q => `
                <div class="question-item ${q.exact_match ? 'correct' : 'incorrect'}">
                    <div class="question-header">
                        <div>
                            <span class="question-id">${q.question_id}</span>
                            <span class="question-type">${q.question_type}</span>
                        </div>
                        <div>
                            <span class="badge ${q.exact_match ? 'success' : 'error'}">
                                ${q.exact_match ? '✓' : '✗'}
                            </span>
                            <span class="badge info">F1: ${q.f1_score.toFixed(2)}</span>
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
                    <td><span class="badge info">${metrics.count}</span></td>
                    <td>${(metrics.exact_match * 100).toFixed(1)}%</td>
                    <td>${metrics.f1_score.toFixed(3)}</td>
                    <td>${metrics.avg_latency_ms.toFixed(0)}ms</td>
                `;
                tbody.appendChild(row);
            });
            
            updateTypeChart(byType);
        }
        
        function updateTypeChart(byType) {
            const types = Object.keys(byType).sort();
            const exactMatch = types.map(t => byType[t].exact_match * 100);
            const f1Score = types.map(t => byType[t].f1_score * 100);
            
            const ctx = document.getElementById('type-chart');
            
            if (typeChart) typeChart.destroy();
            if (types.length === 0) return;
            
            typeChart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: types,
                    datasets: [
                        {
                            label: 'Exact Match %',
                            data: exactMatch,
                            backgroundColor: chartColors.primary,
                            borderRadius: 8,
                        },
                        {
                            label: 'F1 Score (scaled)',
                            data: f1Score,
                            backgroundColor: chartColors.success,
                            borderRadius: 8,
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
                            ticks: { color: chartColors.text },
                            grid: { color: chartColors.grid }
                        },
                        x: {
                            ticks: { color: chartColors.text },
                            grid: { color: chartColors.grid }
                        }
                    },
                    plugins: {
                        legend: { labels: { color: chartColors.text, padding: 20 } }
                    }
                }
            });
        }
        
        function updateTrendsChart(trends) {
            if (!trends.exact_match || trends.exact_match.length === 0) return;
            
            const windowSize = 10;
            const rollingEM = calculateRollingAverage(trends.exact_match, windowSize);
            const rollingF1 = calculateRollingAverage(trends.f1_score, windowSize);
            const labels = Array.from({length: trends.exact_match.length}, (_, i) => i + 1);
            
            const ctx = document.getElementById('trends-chart');
            
            if (trendsChart) {
                trendsChart.data.labels = labels;
                trendsChart.data.datasets[0].data = rollingEM.map(v => v * 100);
                trendsChart.data.datasets[1].data = rollingF1.map(v => v * 100);
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
                            data: rollingEM.map(v => v * 100),
                            borderColor: '#6366f1',
                            backgroundColor: 'rgba(99, 102, 241, 0.1)',
                            fill: true,
                            tension: 0.4,
                            pointRadius: 0,
                            borderWidth: 3,
                        },
                        {
                            label: 'F1 Score (Rolling Avg)',
                            data: rollingF1.map(v => v * 100),
                            borderColor: '#10b981',
                            backgroundColor: 'rgba(16, 185, 129, 0.1)',
                            fill: true,
                            tension: 0.4,
                            pointRadius: 0,
                            borderWidth: 3,
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
                            ticks: { color: chartColors.text, callback: v => v + '%' },
                            grid: { color: chartColors.grid }
                        },
                        x: {
                            ticks: { color: chartColors.text, maxTicksLimit: 20 },
                            grid: { color: chartColors.grid }
                        }
                    },
                    plugins: {
                        legend: { labels: { color: chartColors.text, padding: 20 } }
                    }
                }
            });
        }
        
        function updateLatencyChart(results) {
            if (!results || results.length === 0) return;
            
            const latencies = results.map(r => r.latency_ms);
            const bins = createHistogram(latencies, 15);
            
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
                        label: 'Count',
                        data: bins.counts,
                        backgroundColor: chartColors.secondary,
                        borderRadius: 6,
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            ticks: { color: chartColors.text },
                            grid: { color: chartColors.grid }
                        },
                        x: {
                            ticks: { color: chartColors.text, maxRotation: 45 },
                            grid: { color: chartColors.grid }
                        }
                    },
                    plugins: {
                        legend: { display: false }
                    }
                }
            });
        }
        
        function calculateRollingAverage(data, windowSize) {
            return data.map((_, i) => {
                const start = Math.max(0, i - windowSize + 1);
                const window = data.slice(start, i + 1);
                return window.reduce((a, b) => a + b, 0) / window.length;
            });
        }
        
        function createHistogram(data, numBins) {
            const min = Math.min(...data);
            const max = Math.max(...data);
            const binSize = (max - min) / numBins || 1;
            
            const bins = Array(numBins).fill(0);
            const labels = Array.from({length: numBins}, (_, i) => {
                const start = min + i * binSize;
                return `${start.toFixed(0)}ms`;
            });
            
            data.forEach(value => {
                const idx = Math.min(Math.floor((value - min) / binSize), numBins - 1);
                bins[idx]++;
            });
            
            return { labels, counts: bins };
        }
        
        function loadReport() {
            const path = document.getElementById('report-path').value;
            if (!path) return alert('Please enter a report path');
            
            fetch('/api/load_report', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({report_path: path})
            })
            .then(r => r.json())
            .then(data => {
                if (data.error) alert('Error: ' + data.error);
                else { alert('Report loaded!'); updateDashboard(); }
            });
        }
        
        setInterval(updateDashboard, 1000);
        updateDashboard();
    </script>
</body>
</html>'''


def main():
    """Run the premium visualization server."""
    print("=" * 80)
    print("LongMemEval PREMIUM Dashboard")
    print("=" * 80)
    print()
    print("🔴 LIVE MODE - Ultra-modern real-time visualization")
    print()
    print("Features:")
    print("  ✓ Glassmorphism design with gradient backgrounds")
    print("  ✓ LARGER charts for better visibility")
    print("  ✓ Real-time performance trends")
    print("  ✓ Smooth animations and transitions")
    print("  ✓ Premium gradient color schemes")
    print()
    print("Starting server on http://localhost:5003")
    print("Open your browser to view the live dashboard")
    print()
    print("Press Ctrl+C to stop")
    print("=" * 80)
    
    # Open browser after a short delay
    def open_browser():
        time.sleep(1.5)
        webbrowser.open('http://localhost:5003')
    
    threading.Thread(target=open_browser, daemon=True).start()
    
    # Run server
    app.run(host='0.0.0.0', port=5003, debug=False)


if __name__ == '__main__':
    main()
