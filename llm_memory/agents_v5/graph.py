"""
LangGraph Agent with Memory V5.

Implements a stateful agent with:
- Automatic memory loading before responses
- Graph-based context for complex questions
- Persistent conversation state
- Multi-hop reasoning support
"""

from typing import Literal
import sqlite3
import os

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from .state import AgentStateV5
from .tools import get_memory_tools_v5
from llm_memory.memory_v5.memory_store_v5 import MemoryStoreV5


SYSTEM_PROMPT_V5 = """You are an AI assistant with advanced long-term memory capabilities.

Your memory system includes:
1. **Graph Memory**: Entity-relationship knowledge graph for structured facts
2. **Tiered Memory**: Sensory → Short-term → Long-term memory hierarchy
3. **Multi-hop Reasoning**: Ability to connect facts across multiple hops

MEMORY CONTEXT (automatically loaded):
{memory_context}

GRAPH CONTEXT (entity relationships):
{graph_context}

INSTRUCTIONS:
- Use the memory context above to answer personal questions
- If you learn new important information, use 'save_memory' tool
- For complex questions requiring reasoning, use 'ask_memory' tool
- For relationship questions, use 'query_graph' tool
- Be conversational and remember details across the chat
- If information isn't in memory, say so honestly

Current session: {session_id}
"""


class MemoryAgentV5:
    """
    LangGraph Agent with Memory V5 integration.
    
    Features:
    - Automatic memory loading before each response
    - Graph-based context retrieval
    - Persistent conversation state via checkpointer
    - Tool-based memory interaction
    """
    
    def __init__(
        self,
        model_name: str = "qwen2.5:32b",
        ollama_url: str = "http://localhost:11434",
        memory_path: str = "./agent_memory_v5",
        user_id: str = "default",
    ):
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.memory_path = memory_path
        self.user_id = user_id
        
        # Initialize Memory V5
        self.memory = MemoryStoreV5(
            user_id=user_id,
            persist_path=memory_path,
            model_name=model_name,
            ollama_url=ollama_url,
        )
        
        # Initialize LLM
        self.llm = ChatOllama(
            model=model_name,
            temperature=0.7,
            base_url=ollama_url,
        )
        
        # Initialize Tools
        self.tools = get_memory_tools_v5(self.memory)
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
        # Build Graph
        self.graph = self._build_graph()
    
    def _build_graph(self):
        """Build the LangGraph workflow."""
        # Setup checkpointer for persistent state
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
            
            os.makedirs(self.memory_path, exist_ok=True)
            db_path = f"{self.memory_path}/checkpoints_v5.sqlite"
            conn = sqlite3.connect(db_path, check_same_thread=False)
            checkpointer = SqliteSaver(conn)
        except ImportError:
            checkpointer = None
        
        workflow = StateGraph(AgentStateV5)
        
        # Define nodes
        workflow.add_node("load_memory", self._load_memory_node)
        workflow.add_node("agent", self._agent_node)
        workflow.add_node("tools", ToolNode(self.tools))
        workflow.add_node("save_turn", self._save_turn_node)
        
        # Define edges
        workflow.set_entry_point("load_memory")
        workflow.add_edge("load_memory", "agent")
        
        # Conditional edge from agent
        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {
                "continue": "tools",
                "save": "save_turn",
            }
        )
        
        workflow.add_edge("tools", "agent")
        workflow.add_edge("save_turn", END)
        
        if checkpointer:
            return workflow.compile(checkpointer=checkpointer)
        return workflow.compile()
    
    def _load_memory_node(self, state: AgentStateV5) -> dict:
        """Load relevant memory before agent responds."""
        messages = state.get("messages", [])
        
        if not messages:
            return {
                "memory_context": "No conversation history yet.",
                "graph_context": "",
            }
        
        last_msg = messages[-1]
        if isinstance(last_msg, HumanMessage):
            query = last_msg.content
            
            # Get memory context using advanced retrieval
            memory_context = self.memory.query(query, top_k=15)
            
            # Get graph context for relationship questions
            graph_context = ""
            if self._is_relationship_query(query):
                graph_context = self.memory.query_graph(query)
            
            return {
                "memory_context": memory_context or "No relevant memories found.",
                "graph_context": graph_context,
            }
        
        return {}
    
    def _agent_node(self, state: AgentStateV5) -> dict:
        """Main agent reasoning node."""
        messages = state.get("messages", [])
        memory_context = state.get("memory_context", "No memory loaded.")
        graph_context = state.get("graph_context", "")
        session_id = state.get("session_id", "default")
        
        # Build system prompt with context
        system_content = SYSTEM_PROMPT_V5.format(
            memory_context=memory_context,
            graph_context=graph_context,
            session_id=session_id,
        )
        system_msg = SystemMessage(content=system_content)
        
        # Filter out old system messages
        filtered_msgs = [m for m in messages if not isinstance(m, SystemMessage)]
        
        # Invoke LLM
        response = self.llm_with_tools.invoke([system_msg] + filtered_msgs)
        
        return {"messages": [response]}
    
    def _save_turn_node(self, state: AgentStateV5) -> dict:
        """Save the conversation turn to memory."""
        messages = state.get("messages", [])
        
        if len(messages) >= 2:
            # Find last human and AI messages
            human_msg = None
            ai_msg = None
            
            for msg in reversed(messages):
                if isinstance(msg, AIMessage) and ai_msg is None:
                    ai_msg = msg
                elif isinstance(msg, HumanMessage) and human_msg is None:
                    human_msg = msg
                
                if human_msg and ai_msg:
                    break
            
            if human_msg:
                # Save user message to memory
                self.memory.add_conversation_turn(
                    speaker="User",
                    text=human_msg.content,
                    session_id=state.get("session_id"),
                )
        
        return {}
    
    def _should_continue(self, state: AgentStateV5) -> Literal["continue", "save"]:
        """Decide whether to use tools or finish."""
        messages = state.get("messages", [])
        
        if not messages:
            return "save"
        
        last_message = messages[-1]
        
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "continue"
        
        return "save"
    
    def _is_relationship_query(self, query: str) -> bool:
        """Check if query is about relationships/connections."""
        indicators = [
            'who', 'know', 'friend', 'family', 'work with',
            'relationship', 'connection', 'related',
            'where does', 'where did', 'what does .* like',
        ]
        
        query_lower = query.lower()
        return any(ind in query_lower for ind in indicators)
    
    def chat(self, user_input: str, thread_id: str = "1") -> str:
        """
        Process a chat message and return the response.
        
        Args:
            user_input: User's message
            thread_id: Conversation thread ID for state persistence
            
        Returns:
            Agent's response
        """
        config = {"configurable": {"thread_id": thread_id}}
        
        inputs = {
            "messages": [HumanMessage(content=user_input)],
            "user_id": self.user_id,
            "session_id": thread_id,
            "memory_context": "",
            "graph_context": "",
            "reasoning_log": [],
        }
        
        response_content = ""
        
        for event in self.graph.stream(inputs, config=config):
            for key, value in event.items():
                if key == "agent":
                    messages = value.get("messages", [])
                    if messages:
                        response_content = messages[0].content
        
        return response_content
    
    def get_memory_stats(self) -> dict:
        """Get memory statistics."""
        return self.memory.stats()


# Quick test
if __name__ == "__main__":
    print("Testing Memory Agent V5...")
    
    agent = MemoryAgentV5(
        model_name="qwen2.5:7b",
        memory_path="./test_agent_v5",
    )
    
    # Test conversation
    responses = []
    
    messages = [
        "Hi! My name is Alice and I'm a software engineer at Google.",
        "I love hiking and photography. I live in San Francisco.",
        "What do you know about me?",
    ]
    
    for msg in messages:
        print(f"\nUser: {msg}")
        response = agent.chat(msg)
        print(f"Agent: {response}")
        responses.append(response)
    
    print(f"\n\nMemory Stats: {agent.get_memory_stats()}")
