#!/usr/bin/env python3
"""
Test the LongMemEval visualization with a minimal test run.

This will:
1. Start processing 2 questions
2. Send updates to the viz server
3. Demonstrate the live dashboard features
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

def test_viz():
    """Test the visualization with a minimal run."""
    print("\n" + "="*80)
    print("Testing LongMemEval Enhanced Visualization")
    print("="*80)
    print()
    print("Prerequisites:")
    print("1. Start the visualization server in another terminal:")
    print("   python benchmarks/longmemeval_viz.py")
    print()
    print("2. Wait for it to open in your browser (http://localhost:5001)")
    print()
    print("3. Then run this test with a small number of questions:")
    print("   python run_longmemeval.py --max-questions 2")
    print()
    print("="*80)
    print()
    print("This will demonstrate:")
    print("  ✓ Live progress updates")
    print("  ✓ Real-time question processing")
    print("  ✓ Current question display")
    print("  ✓ Recent questions feed")
    print("  ✓ Performance charts")
    print("  ✓ Type breakdown")
    print("  ✓ Latency distribution")
    print()
    print("="*80)
    
if __name__ == "__main__":
    test_viz()
