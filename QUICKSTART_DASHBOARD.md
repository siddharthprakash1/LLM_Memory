# 🚀 Quick Start: Enhanced Live Dashboard

Get your enhanced LongMemEval dashboard running in 60 seconds!

## Step-by-Step Instructions

### 1️⃣ Open Two Terminals

You need two terminal windows/tabs.

### 2️⃣ Terminal 1: Start Visualization Server

```bash
cd "/Users/siddharthprakash/Desktop/Personal/MyProjects/LLM Memory"
source venv/bin/activate
python benchmarks/longmemeval_viz.py
```

**Expected Output:**
```
================================================================================
LongMemEval Enhanced Live Dashboard
================================================================================

🔴 LIVE MODE - Real-time benchmark visualization

Features:
  ✓ Live question-by-question progress
  ✓ Real-time performance trends
  ✓ Interactive charts and graphs
  ✓ Recent questions feed
  ✓ Detailed type breakdown
  ✓ Latency distribution

Starting server on http://localhost:5001
Open your browser to view the live dashboard

Press Ctrl+C to stop
================================================================================
 * Running on http://0.0.0.0:5001
```

Your browser should automatically open to `http://localhost:5001`

**You should see:**
- Dark purple/blue themed dashboard
- "CONNECTING" status indicator
- Empty charts and metrics
- "No questions processed yet" messages

### 3️⃣ Terminal 2: Run Benchmark

```bash
cd "/Users/siddharthprakash/Desktop/Personal/MyProjects/LLM Memory"
source venv/bin/activate

# Small test (2 questions - good for testing)
python run_longmemeval.py --max-questions 2 --types single-session-user

# OR medium test (10 questions - ~5-10 minutes)
python run_longmemeval.py --max-questions 10

# OR full benchmark (500 questions - ~2-4 hours)
python run_longmemeval.py
```

### 4️⃣ Watch the Magic! ✨

**The dashboard will instantly start showing:**

1. **🔴 LIVE indicator** changes from "CONNECTING" to "LIVE" (pulsing red)
2. **Status** changes from "idle" to "running"
3. **Progress bar** starts filling: 0% → 50% → 100%
4. **Current question** shows what's being processed RIGHT NOW
5. **Metrics** update in real-time (Exact Match, F1 Score)
6. **Recent questions feed** populates with results
7. **Charts** build up as data comes in
8. **Type breakdown** table fills with statistics

---

## 🎥 What You'll See (Timeline)

### T+0s: Benchmark Starts
- Dashboard switches to LIVE mode
- Progress shows 0/2
- Status: "RUNNING"

### T+5s: First Question Processing
- Current Question card shows:
  - Question ID
  - Question text
  - Question type
- Progress bar animates

### T+30s: First Question Complete
- **Recent Questions** gets first entry
- Shows prediction vs ground truth
- Green/red badge (correct/incorrect)
- **Metrics** update: EM%, F1, Latency
- **Charts** start rendering

### T+35s: Second Question Processing
- Current Question updates
- Progress: 1/2 → 2/2
- Trends chart shows 1 data point

### T+60s: Second Question Complete
- Recent Questions shows 2 entries
- Charts have 2 data points
- Type breakdown shows statistics
- Progress: 2/2 (100%)

### T+61s: Benchmark Complete
- Status: "COMPLETED"
- All charts fully rendered
- Final metrics displayed
- Report saved notification

---

## 💡 Pro Tips for First Run

1. **Keep Both Terminals Visible**
   - Split screen or use tmux
   - Watch console output + dashboard

2. **Start with 2 Questions**
   - Tests connectivity
   - Shows all features
   - Fast feedback (~1 minute)

3. **Watch the "Current Question" Card**
   - This is the most interesting part
   - Shows real-time processing
   - Compare predictions vs truth

4. **Check Recent Questions Feed**
   - Auto-scrolls for newest
   - Hover for details
   - Click to see full text (future feature)

5. **Monitor Trends Chart**
   - Even with 2 questions, you'll see points
   - With 10+, you'll see actual trends
   - Rolling average smooths noise

---

## 🐛 Common Issues & Solutions

### Issue: Browser doesn't open automatically
**Solution:** Manually open `http://localhost:5001`

### Issue: Dashboard shows "CONNECTING" forever
**Solution:** 
1. Check Flask server is running (Terminal 1)
2. Restart the viz server
3. Refresh browser (F5)

### Issue: "Connection refused" error
**Solution:**
1. Make sure you started viz server FIRST
2. Check no other app using port 5001
3. Try: `lsof -i :5001` to see what's using the port

### Issue: Benchmark says "Visualization: Not connected"
**Solution:**
1. Start viz server first
2. Wait 2 seconds
3. Then start benchmark
4. Dashboard should say "Visualization: http://localhost:5001 ✓"

### Issue: Charts not rendering
**Solution:**
1. Check internet connection (needs CDN for Chart.js)
2. Clear browser cache
3. Try different browser
4. Check browser console (F12) for errors

---

## 📊 Understanding Your First Results

After your first 2-question test, you should see:

### Good First Results ✅
- **Exact Match**: >30% (at least one correct)
- **F1 Score**: >0.4
- **Latency**: <15 seconds per question
- **Predictions**: Non-empty strings
- **No errors**: In terminal output

### Needs Investigation ⚠️
- **Exact Match**: <20%
- **F1 Score**: <0.3
- **Latency**: >20 seconds
- **Empty predictions**: Check LLM connection

### Troubleshoot 🚨
- **Exact Match**: 0%
- **Errors**: In terminal
- **Timeouts**: Connection issues
- **No updates**: Dashboard not connected

---

## 🎯 Next Steps

1. ✅ **You just did**: Test with 2 questions
2. ⏭️ **Next**: Test with 10 questions (`--max-questions 10`)
3. ⏭️ **Then**: Test specific types (`--types temporal-reasoning --max-questions 5`)
4. ⏭️ **Finally**: Full 500-question benchmark (`run_longmemeval.py`)

---

## 📸 Screenshots to Expect

### Initial State (No Benchmark Running)
```
Status: IDLE
Progress: 0/0
Current Question: No question being processed
Recent Questions: (empty)
Charts: (empty)
```

### During Benchmark (Question 1/2)
```
Status: RUNNING 🔴
Progress: 1/2 (50%)
Current Question: "What degree did I graduate with?"
Recent Questions: (empty - first question still processing)
Charts: (empty)
```

### After First Question (Question 2/2)
```
Status: RUNNING 🔴
Progress: 1/2 (50%)
Current Question: "Where did I work last year?"
Recent Questions: 
  • Q1: Business Administration ✓ (correct)
Charts: 1 data point visible
```

### Benchmark Complete
```
Status: COMPLETED ✅
Progress: 2/2 (100%)
Current Question: (cleared)
Recent Questions: 2 items
Charts: Full with 2 data points
Metrics: Final scores displayed
```

---

## 🎉 You're Ready!

That's it! You now have a professional-grade live dashboard for benchmarking your Memory V4 system. 

**Enjoy watching your AI agent's memory in action!** 🧠✨

---

## 📚 Further Reading

- **Detailed Guide**: See `VISUALIZATION_GUIDE.md` for all features
- **Benchmark Guide**: See `LONGMEMEVAL.md` for dataset info
- **Main README**: See root `README.md` for system overview

---

**Questions?** Check the terminal output for helpful error messages!
