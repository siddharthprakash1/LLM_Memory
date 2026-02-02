# Enhanced LongMemEval Visualization Dashboard 🎨

The new **production-grade live dashboard** provides real-time insights into your Memory V4 benchmark performance with professional visualizations and detailed analytics.

## 🎯 Key Features

### 1. **Live Progress Monitoring** 🔴
- Real-time question-by-question processing
- Live status indicator with pulsing animation
- Progress bar with smooth animations
- Current question display with instant updates

### 2. **Performance Trends** 📈
- **Rolling Average Charts** - See performance trends over time
- Smoothed curves for Exact Match and F1 Score
- Identifies patterns and improvements during the run
- Updates in real-time as questions are processed

### 3. **Recent Questions Feed** 📋
- Live feed of the last 15 processed questions
- Color-coded results (green=correct, red=incorrect)
- Shows predictions vs ground truth side-by-side
- Hover effects for detailed inspection
- Auto-scrolling for newest results

### 4. **Current Question View** 👁️
- See exactly what's being processed right now
- Shows question text, prediction, and ground truth
- Updates instantly when a new question starts
- Helps you understand what the model is doing

### 5. **Interactive Charts** 📊
- **Performance by Type**: Bar chart comparing question types
- **Latency Distribution**: Histogram showing processing time spread
- **Trends Chart**: Line graph with rolling averages
- All charts use Chart.js for smooth, professional rendering

### 6. **Detailed Metrics Dashboard** 📉
- Big number displays for key metrics
- Color-coded indicators (green/yellow/red)
- Per-type breakdown table
- Real-time latency tracking

### 7. **Dark Mode Professional Design** 🌙
- Modern dark theme optimized for long viewing sessions
- Gradient accents and smooth animations
- Font Awesome icons throughout
- Responsive grid layout

---

## 🚀 How to Use

### Step 1: Start the Visualization Server

```bash
# In Terminal 1
cd "/Users/siddharthprakash/Desktop/Personal/MyProjects/LLM Memory"
python benchmarks/longmemeval_viz.py
```

This will:
- Start the Flask server on `http://localhost:5001`
- Automatically open your browser
- Show "CONNECTING" status until a benchmark starts

### Step 2: Run the Benchmark

```bash
# In Terminal 2
cd "/Users/siddharthprakash/Desktop/Personal/MyProjects/LLM Memory"
python run_longmemeval.py --max-questions 5
```

### Step 3: Watch the Magic! ✨

The dashboard will **instantly** start showing:
- Progress updates (0/5 → 1/5 → 2/5...)
- Current question being processed
- Live predictions and ground truth
- Performance trends building up
- Recent questions feed populating

---

## 📊 Dashboard Layout

```
┌─────────────────────────────────────────────────────────┐
│  🧠 LongMemEval Live Dashboard          🔴 LIVE  │ RUNNING │
└─────────────────────────────────────────────────────────┘
┌──────────────┬──────────────┬──────────────┐
│  Progress    │ Exact Match  │  F1 Score    │
│   3/10       │    45.2%     │    0.567     │
│  [=====>   ] │              │              │
└──────────────┴──────────────┴──────────────┘
┌──────────────────────────────┬──────────────┐
│  Performance Trends          │   Overall    │
│  [Rolling Average Chart]     │   Metrics    │
│                              │   • Contains │
│                              │   • Latency  │
└──────────────────────────────┴──────────────┘
┌──────────────────────────────┬──────────────┐
│  Performance by Type         │   Current    │
│  [Bar Chart]                 │   Question   │
│                              │              │
└──────────────────────────────┴──────────────┘
┌──────────────────────────────┬──────────────┐
│  Recent Questions (Live)     │ Type Table   │
│  • Q1: correct ✓             │ breakdown... │
│  • Q2: incorrect ✗           │              │
│  • Q3: correct ✓             │              │
└──────────────────────────────┴──────────────┘
┌─────────────────────────────────────────────┐
│  Latency Distribution                       │
│  [Histogram Chart]                          │
└─────────────────────────────────────────────┘
```

---

## 🎨 Visual Elements

### Status Indicators
- **🔴 LIVE** - Pulsing red dot when benchmark is running
- **✓ COMPLETED** - Green when finished
- **⚠️ ERROR** - Red when an error occurs
- **⏸️ IDLE** - Gray when waiting

### Color Coding
- **Purple/Blue Gradient** - Primary theme colors
- **Green (#10b981)** - Correct answers, good metrics
- **Red (#ef4444)** - Incorrect answers, poor metrics
- **Yellow (#f59e0b)** - Warning level, medium metrics
- **Dark Background** - Easy on the eyes for long sessions

### Animations
- **Progress bar shimmer** - Animated gradient sweep
- **Pulsing indicators** - Breathing animation on live elements
- **Smooth transitions** - All updates fade in smoothly
- **Hover effects** - Cards lift and glow on hover

---

## 📈 Understanding the Charts

### 1. Performance Trends (Line Chart)
Shows how your model's performance evolves over time:
- **X-axis**: Question number (1, 2, 3...)
- **Y-axis**: Percentage (0-100%)
- **Blue line**: Exact Match rolling average
- **Green line**: F1 Score rolling average
- **Rolling window**: Last 10 questions

**What to look for:**
- Upward trend = Model is "learning" (unlikely but cool!)
- Flat line = Consistent performance
- Downward trend = Getting harder questions

### 2. Performance by Type (Bar Chart)
Compares performance across question types:
- **Bars**: Each question type
- **Purple**: Exact Match %
- **Green**: F1 Score (scaled to %)

**What to look for:**
- Which types are easiest/hardest
- Consistency across types
- Outliers that need attention

### 3. Latency Distribution (Histogram)
Shows the spread of processing times:
- **X-axis**: Time ranges (buckets)
- **Y-axis**: Number of questions
- **Bars**: Count per time bucket

**What to look for:**
- Normal distribution = consistent performance
- Long tail = some questions much slower
- Spikes = common processing times

---

## 💡 Pro Tips

### 1. **Watch Current Question**
The "Current Question" card shows exactly what's being processed. This helps you:
- Understand failure patterns
- See how predictions compare to ground truth
- Identify problematic question types

### 2. **Use Recent Questions Feed**
Scroll through the last 15 questions to:
- Spot patterns in failures
- See exact predictions
- Check F1 scores for partial matches

### 3. **Monitor Trends Chart**
The rolling average smooths out noise:
- Early questions might be lucky/unlucky
- The trend tells the real story
- Look for convergence

### 4. **Check Type Breakdown**
The table shows per-type metrics:
- Sort by count to see volume
- Compare EM% across types
- Identify weak areas

### 5. **Load Historical Reports**
Compare current run with past results:
- Enter path to saved JSON report
- Click "Load Report"
- Compare metrics side-by-side (open in new tab)

---

## 🔧 Technical Details

### Update Frequency
- **Dashboard refresh**: Every 1 second
- **Progress updates**: Instant (on every question)
- **Chart updates**: Real-time with animation disabled for smoothness

### Data Flow
```
Benchmark Runner → HTTP POST → Flask Server → Global State → Browser Polling → Chart.js
```

### Browser Requirements
- Modern browser (Chrome, Firefox, Safari, Edge)
- JavaScript enabled
- Local network access to port 5001

### Performance
- Minimal overhead (<1% of benchmark time)
- Updates are non-blocking
- Charts use efficient rendering
- Auto-cleanup of old data

---

## 🐛 Troubleshooting

### Dashboard Not Updating?
1. Check Flask server is running
2. Check browser console for errors
3. Try refreshing the page (F5)
4. Ensure benchmark is sending updates

### Charts Not Rendering?
1. Clear browser cache
2. Check internet connection (for CDN resources)
3. Try different browser
4. Check browser console for Chart.js errors

### "Connection Refused" Error?
1. Start visualization server first
2. Check port 5001 is not in use
3. Check firewall settings
4. Try `http://127.0.0.1:5001` instead

---

## 🎯 What to Look For

### Good Signs ✅
- Exact Match > 35%
- F1 Score > 0.5
- Latency < 10 seconds/question
- Consistent performance across types
- Smooth trend lines

### Warning Signs ⚠️
- Exact Match < 20%
- F1 Score < 0.3
- Latency > 15 seconds
- Large variance across types
- Downward trends

### Red Flags 🚩
- Exact Match < 10%
- F1 Score < 0.1
- Latency > 30 seconds
- Zero correct on some types
- Many empty predictions

---

## 🚀 Next Steps

1. **Run Small Test**: `--max-questions 5`
2. **Watch Live Updates**: See how dashboard responds
3. **Analyze Patterns**: Use charts to identify issues
4. **Run Full Benchmark**: 500 questions (~2-4 hours)
5. **Compare Results**: Load historical reports
6. **Optimize**: Use insights to improve Memory V4

---

## 📸 Screenshot Guide

### What You'll See

**Top Section (Status & Progress)**
- Live indicator (pulsing red dot)
- Progress fraction (3/10)
- Animated progress bar
- Big metric cards (EM, F1)

**Middle Section (Charts)**
- Performance trends over time
- Type comparison bars
- Current question details

**Bottom Section (Detailed View)**
- Live questions feed with predictions
- Type breakdown table
- Latency histogram

---

## 🎨 Customization

### Change Theme Colors
Edit `longmemeval_viz_enhanced.py`, CSS section:
```css
--primary: #667eea;      /* Purple */
--secondary: #764ba2;    /* Dark purple */
--success: #10b981;      /* Green */
--warning: #f59e0b;      /* Orange */
--error: #ef4444;        /* Red */
```

### Adjust Update Speed
Edit `longmemeval_viz_enhanced.py`, JavaScript section:
```javascript
setInterval(updateDashboard, 1000);  // Change 1000 to desired ms
```

### Change Chart Colors
Edit chart datasets in JavaScript:
```javascript
backgroundColor: 'rgba(102, 126, 234, 0.8)',  // Your color here
```

---

## 🏆 Comparison with Old Dashboard

| Feature | Old Dashboard | Enhanced Dashboard |
|---------|---------------|-------------------|
| Live Updates | ✅ | ✅ (1s refresh) |
| Current Question | ❌ | ✅ Full details |
| Recent Questions | ❌ | ✅ Last 15 with feed |
| Trends Chart | ❌ | ✅ Rolling average |
| Latency Chart | ❌ | ✅ Histogram |
| Type Breakdown | ✅ Basic table | ✅ Enhanced + chart |
| Dark Mode | ❌ | ✅ Professional |
| Animations | ✅ Basic | ✅ Smooth & polished |
| Responsive | ✅ | ✅ Better grid |
| Icons | ❌ | ✅ Font Awesome |

---

Enjoy the enhanced dashboard! 🎉 Watch your Memory V4 perform in real-time with beautiful, insightful visualizations.
