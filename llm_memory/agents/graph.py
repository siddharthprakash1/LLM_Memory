"""
Main Graph Definition for Multi-Agent System.

This module defines the LangGraph workflow with:
- Supervisor-Worker routing
- Conditional edges
- Memory V3 integration
- Streaming support
"""

import logging
from typing import Any, Literal

from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver

from .state import AgentState, Message, create_initial_state
from .config import AgentConfig, DEFAULT_CONFIG
from ..memory_v3 import HierarchicalMemoryV3, create_memory_v3
from .supervisor import SupervisorAgent
from .research import ResearchAgent
from .code import CodeAgent
from .analysis import AnalysisAgent
from .writer import WriterAgent

logger = logging.getLogger(__name__)

# Global memory instance
_graph_memory: HierarchicalMemoryV3 | None = None


def create_graph(
    config: AgentConfig | None = None,
    user_id: str = "default",
) -> StateGraph:
    """
    Create the multi-agent graph with all nodes and edges.
    
    Args:
        config: Configuration object. If None, uses default config.
        user_id: User ID for memory persistence
        
    Returns:
        Compiled StateGraph ready for execution
    """
    global _graph_memory
    
    if config is None:
        config = DEFAULT_CONFIG
    
    # Initialize Memory V3 with Knowledge Graph
    _graph_memory = create_memory_v3(
        user_id=user_id,
        persist_path=config.memory_persist_path,
    )
    logger.info(f"Initialized Memory V3 (KG + Multi-hop) for user: {user_id}")
    
    # Initialize agents
    supervisor = SupervisorAgent(config)
    research_agent = ResearchAgent(config)
    code_agent = CodeAgent(config)
    analysis_agent = AnalysisAgent(config)
    writer_agent = WriterAgent(config)
    
    # Attach memory to all agents
    for agent in [supervisor, research_agent, code_agent, analysis_agent, writer_agent]:
        agent.set_memory(_graph_memory)
    
    # Create the graph
    workflow = StateGraph(AgentState)
    
    # =========================================
    # Define Node Functions
    # =========================================
    
    def supervisor_node(state: AgentState) -> dict[str, Any]:
        """Supervisor decides next action."""
        logger.info(">>> Supervisor Node")
        return supervisor.invoke(state)
    
    def research_node(state: AgentState) -> dict[str, Any]:
        """Research agent gathers information."""
        logger.info(">>> Research Node")
        return research_agent.invoke(state)
    
    def code_node(state: AgentState) -> dict[str, Any]:
        """Code agent writes/executes code."""
        logger.info(">>> Code Node")
        return code_agent.invoke(state)
    
    def analysis_node(state: AgentState) -> dict[str, Any]:
        """Analysis agent performs reasoning."""
        logger.info(">>> Analysis Node")
        return analysis_agent.invoke(state)
    
    def writer_node(state: AgentState) -> dict[str, Any]:
        """Writer agent creates content."""
        logger.info(">>> Writer Node")
        return writer_agent.invoke(state)
    
    def compile_response_node(state: AgentState) -> dict[str, Any]:
        """Compile final response from all agent work."""
        logger.info(">>> Compiling Final Response")
        final_response = supervisor.compile_final_response(state)
        
        # Store task and key findings in memory
        if _graph_memory:
            _graph_memory.add(
                f"Completed task: {state['task'][:300]}",
                memory_type="event",
                importance=0.6,
            )
            
            # Store key findings from artifacts
            artifacts = state.get("artifacts", {})
            
            if "research" in artifacts:
                research = artifacts["research"]
                if isinstance(research, dict) and research.get("summary"):
                    _graph_memory.add_fact(
                        f"Research finding: {research['summary'][:500]}",
                        importance=0.7,
                    )
            
            if "analysis" in artifacts:
                analysis = artifacts["analysis"]
                if isinstance(analysis, dict) and analysis.get("analysis"):
                    _graph_memory.add(
                        f"Analysis result: {analysis['analysis'][:500]}",
                        memory_type="fact",
                        importance=0.7,
                    )
            
            logger.info(f"Memory updated: {_graph_memory.stats()}")
        
        return {
            "final_response": final_response,
            "messages": Message(
                role="assistant",
                content=final_response,
                name="final"
            )
        }
    
    # =========================================
    # Add Nodes to Graph
    # =========================================
    
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("research", research_node)
    workflow.add_node("code", code_node)
    workflow.add_node("analysis", analysis_node)
    workflow.add_node("writer", writer_node)
    workflow.add_node("compile_response", compile_response_node)
    
    # =========================================
    # Define Routing Logic
    # =========================================
    
    def route_supervisor(state: AgentState) -> Literal["research", "code", "analysis", "writer", "compile_response"]:
        """Route based on supervisor's decision."""
        next_agent = state.get("next_agent", "").lower()
        
        # Check for max iterations
        if state.get("iteration", 0) >= state.get("max_iterations", 10):
            logger.warning("Max iterations - forcing finish")
            return "compile_response"
        
        # Route to appropriate agent
        if next_agent == "finish":
            return "compile_response"
        elif next_agent == "research":
            return "research"
        elif next_agent == "code":
            return "code"
        elif next_agent == "analysis":
            return "analysis"
        elif next_agent == "writer":
            return "writer"
        else:
            logger.warning(f"Unknown next_agent: {next_agent}, defaulting to compile")
            return "compile_response"
    
    def route_after_agent(state: AgentState) -> Literal["supervisor", "compile_response"]:
        """After any agent, return to supervisor for next decision."""
        # Check for errors
        if state.get("error"):
            return "compile_response"
        
        # Check for max iterations
        if state.get("iteration", 0) >= state.get("max_iterations", 10):
            return "compile_response"
        
        return "supervisor"
    
    # =========================================
    # Define Edges
    # =========================================
    
    # Entry point
    workflow.add_edge(START, "supervisor")
    
    # Supervisor routes to agents or finish
    workflow.add_conditional_edges(
        "supervisor",
        route_supervisor,
        {
            "research": "research",
            "code": "code",
            "analysis": "analysis",
            "writer": "writer",
            "compile_response": "compile_response",
        }
    )
    
    # All agents return to supervisor
    workflow.add_conditional_edges(
        "research",
        route_after_agent,
        {"supervisor": "supervisor", "compile_response": "compile_response"}
    )
    workflow.add_conditional_edges(
        "code",
        route_after_agent,
        {"supervisor": "supervisor", "compile_response": "compile_response"}
    )
    workflow.add_conditional_edges(
        "analysis",
        route_after_agent,
        {"supervisor": "supervisor", "compile_response": "compile_response"}
    )
    workflow.add_conditional_edges(
        "writer",
        route_after_agent,
        {"supervisor": "supervisor", "compile_response": "compile_response"}
    )
    
    # Final response ends the graph
    workflow.add_edge("compile_response", END)
    
    # =========================================
    # Compile with Memory/Checkpointing
    # =========================================
    
    if config.enable_memory:
        checkpointer = MemorySaver()
        logger.info("Using in-memory checkpointer")
    else:
        checkpointer = None
    
    compiled = workflow.compile(checkpointer=checkpointer)
    
    logger.info("Graph compiled successfully")
    return compiled


def run_graph(
    graph,
    task: str,
    config: AgentConfig | None = None,
    thread_id: str = "default",
    max_iterations: int = 10,
) -> dict[str, Any]:
    """
    Run the multi-agent graph on a task.
    
    Args:
        graph: Compiled graph from create_graph()
        task: The task/query to process
        config: Configuration object
        thread_id: Thread ID for memory persistence
        max_iterations: Maximum iterations before forcing completion
        
    Returns:
        Final state dictionary
    """
    initial_state = create_initial_state(task, max_iterations)
    run_config = {"configurable": {"thread_id": thread_id}}
    
    logger.info(f"Starting graph execution for task: {task[:100]}...")
    
    final_state = graph.invoke(initial_state, run_config)
    
    logger.info("Graph execution completed")
    return final_state


async def run_graph_async(
    graph,
    task: str,
    config: AgentConfig | None = None,
    thread_id: str = "default",
    max_iterations: int = 10,
) -> dict[str, Any]:
    """Run the multi-agent graph asynchronously."""
    initial_state = create_initial_state(task, max_iterations)
    run_config = {"configurable": {"thread_id": thread_id}}
    
    logger.info(f"Starting async graph execution for task: {task[:100]}...")
    
    final_state = await graph.ainvoke(initial_state, run_config)
    
    logger.info("Async graph execution completed")
    return final_state


def run_graph_stream(
    graph,
    task: str,
    config: AgentConfig | None = None,
    thread_id: str = "default",
    max_iterations: int = 10,
):
    """
    Run the multi-agent graph with streaming output.
    
    Yields state updates as the graph executes.
    
    Args:
        graph: Compiled graph from create_graph()
        task: The task/query to process
        config: Configuration object
        thread_id: Thread ID for memory persistence
        max_iterations: Maximum iterations
        
    Yields:
        Tuples of (node_name, state_update)
    """
    initial_state = create_initial_state(task, max_iterations)
    run_config = {"configurable": {"thread_id": thread_id}}
    
    logger.info(f"Starting streaming graph execution for task: {task[:100]}...")
    
    for event in graph.stream(initial_state, run_config, stream_mode="updates"):
        for node_name, state_update in event.items():
            logger.debug(f"Stream update from {node_name}")
            yield node_name, state_update
    
    logger.info("Streaming graph execution completed")


# =========================================
# Memory Access Functions
# =========================================

def get_graph_memory() -> HierarchicalMemoryV3 | None:
    """Get the current graph's Memory V3 system."""
    return _graph_memory


def search_memory(query: str, top_k: int = 10) -> list:
    """Search memory using advanced retrieval."""
    if _graph_memory:
        return _graph_memory.search(query, top_k=top_k, use_decomposition=True, use_reranking=True)
    return []


def get_knowledge_graph_facts(entity: str) -> list[str]:
    """Get facts from the Knowledge Graph about an entity."""
    if _graph_memory:
        return _graph_memory.knowledge_graph.get_facts_about(entity, as_text=True)
    return []


def add_to_memory(content: str, memory_type: str = "fact", importance: float = 0.5):
    """Add content to memory."""
    if _graph_memory:
        _graph_memory.add(content, memory_type=memory_type, importance=importance)


def get_memory_stats() -> dict:
    """Get memory statistics including KG stats."""
    if _graph_memory:
        return _graph_memory.stats()
    return {}


# =========================================
# Visualization Helper
# =========================================

def visualize_graph(graph, output_path: str = "graph.png"):
    """Generate a visualization of the graph structure."""
    try:
        mermaid = graph.get_graph().draw_mermaid()
        print("Graph Mermaid Diagram:")
        print(mermaid)
        
        try:
            png_bytes = graph.get_graph().draw_mermaid_png()
            with open(output_path, "wb") as f:
                f.write(png_bytes)
            logger.info(f"Graph visualization saved to {output_path}")
            return png_bytes
        except Exception as e:
            logger.warning(f"Could not render PNG: {e}")
            return mermaid
            
    except Exception as e:
        logger.error(f"Could not visualize graph: {e}")
        return None
