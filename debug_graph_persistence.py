"""
Diagnostic script for V5 Graph Memory.
Verifies that rule-based extraction populates the graph store.
"""
import os
import shutil
from llm_memory.memory_v5.memory_store_v5 import MemoryStoreV5
from llm_memory.memory_v5 import create_memory_v5

TEST_DIR = "test_debug_graph_v5"

def clean():
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)

def test_graph_population():
    print(f"Initializing V5 Memory in {TEST_DIR}...")
    memory = create_memory_v5(
        user_id="debug_user",
        persist_path=TEST_DIR,
        use_llm=False
    )
    
    # Test Input
    text = "Caroline lives in New York and works at Google."
    print(f"Adding turn: '{text}'")
    
    memory.add_conversation_turn(
        speaker="User",
        text=text,
        date="2023-01-01"
    )
    
    # Check Graph State
    triplet_count = len(memory.graph.triplets)
    print(f"\nGraph Triplet Count: {triplet_count}")
    
    if triplet_count > 0:
        print("✅ Graph populated successfully!")
        for t in memory.graph.triplets.values():
            print(f"  - {t.subject.name} [{t.predicate.relation_type.value}] {t.object.name}")
    else:
        print("❌ Graph is EMPTY!")
        
        # Check extraction directly
        print("\nChecking extraction logic directly...")
        extracted = memory._extract_rule_based(text, "User")
        print(f"Extracted: {extracted}")

if __name__ == "__main__":
    clean()
    try:
        test_graph_population()
    finally:
        clean()
