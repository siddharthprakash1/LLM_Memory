
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from llm_memory.memory_v5.memory_store_v5 import MemoryStoreV5

def test_openai_extraction():
    print("Testing OpenAI Extraction with GPT-4o Mini...")
    
    # API Key from environment variable
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: Set OPENAI_API_KEY environment variable")
        return
    
    # Init memory with OpenAI
    memory = MemoryStoreV5(
        user_id="test_gpt5",
        model_name="gpt-4o-mini",  # Using gpt-4o-mini as proxy/alias for "GPT-5 mini" if available, or just generic 4o-mini which is current SOTA small model
        openai_api_key=api_key,
        use_llm=True
    )
    
    # complex input to test extraction
    text = "On July 20, 2024, I moved to San Francisco to work at Anthropic as a researcher. I love hiking in the Presidio. My cat Luna has been with me for 5 years."
    
    print(f"\nProcessing text: '{text}'")
    turn = memory.add_conversation_turn("Siddharth", text)
    
    print("\n--- Extracted Entities ---")
    for ent in turn.extracted_entities:
        print(f"- {ent['name']} ({ent['type']})")
        
    print("\n--- Extracted Facts ---")
    for fact in turn.extracted_facts:
        print(f"- {fact['subject']} {fact['predicate']} {fact['object']}")
        
    print("\n--- Graph Check ---")
    subgraph = memory.graph.get_entity_subgraph(memory.graph.get_entity_by_name("Siddharth").entity_id)
    for triplet in subgraph:
        print(f"[Graph] {triplet.as_text()}")

if __name__ == "__main__":
    test_openai_extraction()
