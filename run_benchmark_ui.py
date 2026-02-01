"""
Benchmark Visualizer - Runs LOCOMO conversations through the Web UI.

This script acts as a user, sending conversation turns from the benchmark
to the running Web UI. This allows you to watch the memory graph and facts
build up in real-time in the browser.
"""

import json
import time
import requests
import os
from typing import List

# Configuration
UI_URL = "http://127.0.0.1:5000/api/chat"
CLEAR_URL = "http://127.0.0.1:5000/api/clear"
DATA_PATH = "benchmarks/locomo_data/data/locomo10.json"

def load_data():
    """Load LOCOMO dataset."""
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data file not found at {DATA_PATH}")
        return []
    
    with open(DATA_PATH, 'r') as f:
        return json.load(f)

def extract_turns(conversation: dict) -> List[dict]:
    """Extract conversation turns from LOCOMO format."""
    turns = []
    
    # LOCOMO format: dict with session_N keys
    if isinstance(conversation, dict):
        # Get all session keys
        session_keys = [k for k in conversation.keys() if k.startswith('session_') and not k.endswith('_date_time')]
        session_keys.sort(key=lambda x: int(x.split('_')[1]) if x.split('_')[1].isdigit() else 0)
        
        for session_key in session_keys:
            session_num = session_key.split('_')[1]
            date_key = f"session_{session_num}_date_time"
            date = conversation.get(date_key, f'Session {session_num}')
            
            session_turns = conversation.get(session_key, [])
            if isinstance(session_turns, list):
                for turn in session_turns:
                    if isinstance(turn, dict):
                        speaker = turn.get('speaker', 'Unknown')
                        text = turn.get('text', turn.get('utterance', ''))
                        turns.append({
                            'speaker': speaker,
                            'text': text,
                            'date': date,
                            'session': session_num,
                        })
    return turns

def run_conversation(turns: List[dict], delay: float = 2.0):
    """Run a conversation through the UI."""
    print(f"Starting conversation with {len(turns)} turns...")
    print("Watch the Web UI to see memory forming!")
    
    # Clear memory first
    requests.post(CLEAR_URL)
    print("Memory cleared.")
    time.sleep(1)
    
    for i, turn in enumerate(turns):
        speaker = turn['speaker']
        text = turn['text']
        date = turn['date']
        
        # Construct message
        # We format it so the agent knows who is speaking and when
        # The agent prompt expects user input, so we frame it as:
        # "[Date] Speaker: Message"
        message = f"[{date}] {speaker}: {text}"
        
        print(f"\n[{i+1}/{len(turns)}] Sending: {message[:50]}...")
        
        try:
            response = requests.post(UI_URL, json={"message": message})
            if response.status_code == 200:
                print("  ✅ Processed")
            else:
                print(f"  ❌ Error: {response.status_code}")
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            
        # Wait for user to see the update
        time.sleep(delay)

def main():
    data = load_data()
    if not data:
        return
    
    # Pick the first conversation (Caroline & Melanie)
    sample = data[0]
    print(f"Loaded conversation: {sample.get('sample_id', 'Unknown')}")
    print(f"Speakers: {sample['conversation'].get('speaker_a')} & {sample['conversation'].get('speaker_b')}")
    
    turns = extract_turns(sample['conversation'])
    
    # Run the first 20 turns for demo
    run_conversation(turns[:20], delay=3.0)
    
    print("\nDemo complete! Check the Web UI Graph and Timeline.")

if __name__ == "__main__":
    main()
