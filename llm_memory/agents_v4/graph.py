"""
LangGraph Agent Definition.

This defines the agent workflow using LangGraph.
"""

import operator
from typing import Annotated, Sequence, TypedDict, Union

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from llm_memory.agents_v4.state import AgentState
from llm_memory.agents_v4.tools import get_memory_tools
from llm_memory.memory_v4.memory_store import MemoryStoreV4


SYSTEM_PROMPT = """You are a helpful AI assistant with Long-Term Memory.

You have access to a powerful memory system that can:
1. Save important details (preferences, events, facts)
2. Search for past information
3. Answer complex questions about time and duration

MEMORY CONTEXT:
{memory_context}

INSTRUCTIONS:
- Always check your memory context before answering personal questions.
- If you learn something new and important about the user, use 'save_memory'.
- If you need to recall details not in context, use 'search_memory'.
- For "how long" or "when" questions, use 'ask_memory'.
- Be concise and helpful.
"""


class MemoryAgent:
    def __init__(
        self, 
        model_name: str = "qwen2.5:32b",
        ollama_url: str = "http://localhost:11434",
        memory_path: str = "./agent_memory"
    ):
        # Initialize Memory V4
        self.memory = MemoryStoreV4(
            user_id="default",
            persist_path=memory_path,
            model_name=model_name,
            ollama_url=ollama_url
        )
        
        # Initialize LLM
        self.llm = ChatOllama(
            model=model_name,
            temperature=0.7,
            base_url=ollama_url
        )
        
        # Initialize Tools
        self.tools = get_memory_tools(self.memory)
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
        # Build Graph
        self.graph = self._build_graph()
        
    def _build_graph(self):
        # Use SqliteSaver for persistent memory of the conversation state itself
        # Note: In newer LangGraph versions, checkpointer location might vary
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
        except ImportError:
            # Fallback for newer langgraph versions where it might be in a different path
            # or if the package name changed. 
            # Let's try the core checkpointer if sqlite specific one fails, 
            # but actually let's check what's available.
            # For now, let's assume it's available or we need to install langgraph-checkpoint-sqlite
            # But wait, we installed langgraph-checkpoint.
            from langgraph.checkpoint.sqlite import SqliteSaver
        
        import sqlite3
        
        # Ensure checkpoint directory exists
        import os
        os.makedirs(self.memory.persist_path, exist_ok=True)
        db_path = f"{self.memory.persist_path}/checkpoints.sqlite"
        conn = sqlite3.connect(db_path, check_same_thread=False)
        checkpointer = SqliteSaver(conn)
        
        workflow = StateGraph(AgentState)
        
        # Define Nodes
        workflow.add_node("load_memory", self._load_memory_node)
        workflow.add_node("agent", self._agent_node)
        workflow.add_node("tools", ToolNode(self.tools))
        
        # Define Edges
        workflow.set_entry_point("load_memory")
        workflow.add_edge("load_memory", "agent")
        
        # Conditional edge from agent
        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {
                "continue": "tools",
                "end": END
            }
        )
        
        workflow.add_edge("tools", "agent")
        
        return workflow.compile(checkpointer=checkpointer)

    def _load_memory_node(self, state: AgentState):
        """Implicitly load relevant memory before agent acts."""
        messages = state["messages"]
        if not messages:
            return {"memory_context": ""}
            
        last_msg = messages[-1]
        if isinstance(last_msg, HumanMessage):
            # Use the memory store's build_context_for_question method
            # which now includes the multi-hop reasoner logic
            context = self.memory.build_context_for_question(
                last_msg.content, 
                max_facts=20,
                include_episodes=True
            )
            return {"memory_context": context}
            
        return {}

    def _agent_node(self, state: AgentState):
        """The core agent node."""
        messages = state["messages"]
        memory_context = state.get("memory_context", "No relevant memory found.")
        
        # Inject system prompt with memory context
        system_msg = SystemMessage(content=SYSTEM_PROMPT.format(memory_context=memory_context))
        
        # Filter out old system messages to avoid duplication if we loop
        filtered_msgs = [m for m in messages if not isinstance(m, SystemMessage)]
        
        response = self.llm_with_tools.invoke([system_msg] + filtered_msgs)
        return {"messages": [response]}

    def _should_continue(self, state: AgentState):
        """Decide whether to use tools or end."""
        messages = state["messages"]
        last_message = messages[-1]
        
        if last_message.tool_calls:
            return "continue"
        return "end"
        
    def chat(self, user_input: str, thread_id: str = "1"):
        """Run a chat interaction."""
        config = {"configurable": {"thread_id": thread_id}}
        
        # Get current state to see if we have history
        current_state = self.graph.get_state(config)
        
        # If we have history, we don't need to re-initialize everything,
        # but LangGraph handles appending automatically if we pass messages.
        
        # Add user message to state
        inputs = {
            "messages": [HumanMessage(content=user_input)],
            "user_id": "default",
            # Don't overwrite memory_context here, let the graph update it
        }
        
        # Stream output
        for event in self.graph.stream(inputs, config=config):
            for key, value in event.items():
                if key == "agent":
                    msg = value["messages"][0]
                    print(f"\n🤖 Agent: {msg.content}")
                elif key == "tools":
                    # Tool output
                    pass
