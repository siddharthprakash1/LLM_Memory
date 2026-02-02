#!/usr/bin/env python3
"""
Enhanced Real-Time Visualization UI for LongMemEval Benchmark.

This is a redirect to the enhanced version.
Use longmemeval_viz_enhanced.py for the latest features.
"""

import sys
from pathlib import Path

# Redirect to enhanced version
print("=" * 80)
print("NOTE: Using enhanced visualization dashboard")
print("=" * 80)
print()

# Import and run the enhanced version
sys.path.insert(0, str(Path(__file__).parent))
from longmemeval_viz_enhanced import main

if __name__ == '__main__':
    main()


# ===========================================
# API Endpoints
# ===========================================

@app.route('/')
def index():
    """Main dashboard."""
    return render_template('longmemeval_dashboard.html')


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
    
    # Add result if provided
    if 'result' in data:
        current_state['results'].append(data['result'])
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
        }
        current_state['progress']['total'] = report_data.get('total_questions', 0)
        current_state['progress']['current'] = report_data.get('total_questions', 0)
        
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
        }
    
    current_state['summary']['by_type'] = type_metrics


# ===========================================
# HTML Template
# ===========================================

HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LongMemEval Benchmark Dashboard - Live</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-annotation@3.0.1/dist/chartjs-plugin-annotation.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        :root {
            --primary: #667eea;
            --secondary: #764ba2;
            --success: #10b981;
            --warning: #f59e0b;
            --error: #ef4444;
            --info: #3b82f6;
            --bg-dark: #1a1a2e;
            --bg-card: #16213e;
            --text-light: #e1e1e1;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: var(--text-light);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        
        .header {
            text-align: center;
            color: white;
            margin-bottom: 30px;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .header p {
            font-size: 1.1em;
            opacity: 0.9;
        }
        
        .dashboard {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        
        .card {
            background: white;
            border-radius: 12px;
            padding: 25px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        
        .card h2 {
            font-size: 1.3em;
            margin-bottom: 15px;
            color: #667eea;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }
        
        .metric {
            display: flex;
            justify-content: space-between;
            padding: 10px 0;
            border-bottom: 1px solid #eee;
        }
        
        .metric:last-child { border-bottom: none; }
        
        .metric-label {
            font-weight: 600;
            color: #666;
        }
        
        .metric-value {
            font-weight: bold;
            color: #333;
            font-size: 1.1em;
        }
        
        .metric-value.good { color: #10b981; }
        .metric-value.medium { color: #f59e0b; }
        .metric-value.poor { color: #ef4444; }
        
        .progress-container {
            margin: 20px 0;
        }
        
        .progress-bar {
            width: 100%;
            height: 30px;
            background: #eee;
            border-radius: 15px;
            overflow: hidden;
            position: relative;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            transition: width 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
        }
        
        .status {
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 0.9em;
        }
        
        .status.idle { background: #ddd; color: #666; }
        .status.running { background: #3b82f6; color: white; animation: pulse 2s infinite; }
        .status.completed { background: #10b981; color: white; }
        .status.error { background: #ef4444; color: white; }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
        }
        
        .chart-container {
            position: relative;
            height: 300px;
            margin-top: 20px;
        }
        
        .full-width {
            grid-column: 1 / -1;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }
        
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }
        
        th {
            background: #f8f9fa;
            font-weight: 600;
            color: #667eea;
        }
        
        tr:hover {
            background: #f8f9fa;
        }
        
        .load-report {
            margin-top: 20px;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 8px;
        }
        
        input[type="text"] {
            width: 100%;
            padding: 10px;
            border: 2px solid #ddd;
            border-radius: 6px;
            font-size: 1em;
            margin-top: 10px;
        }
        
        button {
            padding: 10px 20px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 1em;
            margin-top: 10px;
        }
        
        button:hover {
            background: #5568d3;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧠 LongMemEval Benchmark Dashboard</h1>
            <p>Real-time visualization of Memory V4 performance on LongMemEval</p>
        </div>
        
        <div class="dashboard">
            <!-- Status Card -->
            <div class="card">
                <h2>Status</h2>
                <div class="metric">
                    <span class="metric-label">Current Status:</span>
                    <span class="status" id="status">idle</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Progress:</span>
                    <span class="metric-value" id="progress-text">0/0</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Current Question:</span>
                    <span class="metric-value" id="current-question">-</span>
                </div>
                <div class="progress-container">
                    <div class="progress-bar">
                        <div class="progress-fill" id="progress-fill" style="width: 0%;">0%</div>
                    </div>
                </div>
            </div>
            
            <!-- Overall Metrics Card -->
            <div class="card">
                <h2>Overall Metrics</h2>
                <div class="metric">
                    <span class="metric-label">Exact Match:</span>
                    <span class="metric-value" id="exact-match">0%</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Contains Match:</span>
                    <span class="metric-value" id="contains-match">0%</span>
                </div>
                <div class="metric">
                    <span class="metric-label">F1 Score:</span>
                    <span class="metric-value" id="f1-score">0.000</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Avg Latency:</span>
                    <span class="metric-value" id="avg-latency">0ms</span>
                </div>
            </div>
            
            <!-- Configuration Card -->
            <div class="card">
                <h2>Configuration</h2>
                <div class="metric">
                    <span class="metric-label">Model:</span>
                    <span class="metric-value" id="model">-</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Dataset:</span>
                    <span class="metric-value" id="dataset">LongMemEval-S</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Total Questions:</span>
                    <span class="metric-value" id="total-questions">0</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Start Time:</span>
                    <span class="metric-value" id="start-time">-</span>
                </div>
            </div>
            
            <!-- Performance by Type Chart -->
            <div class="card full-width">
                <h2>Performance by Question Type</h2>
                <div class="chart-container">
                    <canvas id="type-chart"></canvas>
                </div>
            </div>
            
            <!-- Detailed Results Table -->
            <div class="card full-width">
                <h2>Question Type Breakdown</h2>
                <table id="type-table">
                    <thead>
                        <tr>
                            <th>Question Type</th>
                            <th>Count</th>
                            <th>Exact Match %</th>
                            <th>Contains %</th>
                            <th>F1 Score</th>
                            <th>Avg Latency</th>
                        </tr>
                    </thead>
                    <tbody id="type-table-body">
                        <tr><td colspan="6" style="text-align: center; color: #999;">No data yet</td></tr>
                    </tbody>
                </table>
            </div>
            
            <!-- Load Report Card -->
            <div class="card full-width load-report">
                <h2>Load Saved Report</h2>
                <input type="text" id="report-path" placeholder="Path to report JSON file (e.g., benchmarks/reports/longmemeval_xxx.json)">
                <button onclick="loadReport()">Load Report</button>
            </div>
        </div>
    </div>
    
    <script>
        let typeChart = null;
        
        function updateDashboard() {
            fetch('/api/status')
                .then(r => r.json())
                .then(data => {
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
                    document.getElementById('current-question').textContent = 
                        progress.current_question_id ? `${progress.current_question_id} (${progress.current_question_type})` : '-';
                    
                    // Overall metrics
                    const overall = data.summary.overall || {};
                    document.getElementById('exact-match').textContent = (overall.exact_match * 100 || 0).toFixed(2) + '%';
                    document.getElementById('contains-match').textContent = (overall.contains_match * 100 || 0).toFixed(2) + '%';
                    document.getElementById('f1-score').textContent = (overall.f1_score || 0).toFixed(3);
                    document.getElementById('avg-latency').textContent = (overall.avg_latency_ms || 0).toFixed(1) + 'ms';
                    
                    // Config
                    document.getElementById('model').textContent = data.config.model_name || '-';
                    document.getElementById('total-questions').textContent = progress.total;
                    document.getElementById('start-time').textContent = 
                        data.start_time ? new Date(data.start_time).toLocaleString() : '-';
                    
                    // Type breakdown
                    updateTypeBreakdown(data.summary.by_type || {});
                });
        }
        
        function updateTypeBreakdown(byType) {
            // Update table
            const tbody = document.getElementById('type-table-body');
            if (Object.keys(byType).length === 0) {
                tbody.innerHTML = '<tr><td colspan="6" style="text-align: center; color: #999;">No data yet</td></tr>';
            } else {
                tbody.innerHTML = '';
                Object.entries(byType).sort((a, b) => a[0].localeCompare(b[0])).forEach(([type, metrics]) => {
                    const row = document.createElement('tr');
                    row.innerHTML = `
                        <td>${type}</td>
                        <td>${metrics.count}</td>
                        <td>${(metrics.exact_match * 100).toFixed(2)}%</td>
                        <td>${(metrics.contains_match * 100).toFixed(2)}%</td>
                        <td>${metrics.f1_score.toFixed(3)}</td>
                        <td>${metrics.avg_latency_ms.toFixed(1)}ms</td>
                    `;
                    tbody.appendChild(row);
                });
            }
            
            // Update chart
            updateChart(byType);
        }
        
        function updateChart(byType) {
            const types = Object.keys(byType).sort();
            const exactMatch = types.map(t => byType[t].exact_match * 100);
            const containsMatch = types.map(t => byType[t].contains_match * 100);
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
                            label: 'Contains Match %',
                            data: containsMatch,
                            backgroundColor: 'rgba(118, 75, 162, 0.8)',
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
                                callback: function(value) {
                                    return value + '%';
                                }
                            }
                        }
                    }
                }
            });
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
        
        // Update every 2 seconds
        setInterval(updateDashboard, 2000);
        updateDashboard();
    </script>
</body>
</html>'''


# Save template
def save_template():
    """Save HTML template to templates directory."""
    template_dir = Path(__file__).parent / 'templates'
    template_dir.mkdir(exist_ok=True)
    
    with open(template_dir / 'longmemeval_dashboard.html', 'w') as f:
        f.write(HTML_TEMPLATE)


def main():
    """Run the visualization server."""
    print("=" * 80)
    print("LongMemEval Visualization Dashboard")
    print("=" * 80)
    print()
    print("Starting server on http://localhost:5001")
    print("Open your browser to view the dashboard")
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
