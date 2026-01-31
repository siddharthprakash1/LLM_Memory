"""
Code Agent - Programming and Execution Specialist.

Responsible for:
- Writing code
- Executing Python
- Debugging
- Data processing
"""

import logging
import sys
from io import StringIO
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from .base import BaseAgent
from .state import AgentState, Message
from .config import AgentConfig

logger = logging.getLogger(__name__)


# ============================================
# Code Tools
# ============================================

@tool
def execute_python(code: str) -> str:
    """
    Execute Python code and return the output.
    
    Args:
        code: Python code to execute
        
    Returns:
        Execution output or error message
    """
    # Capture stdout
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    
    try:
        # Create a safe execution environment
        exec_globals = {
            "__builtins__": __builtins__,
            "print": print,
        }
        
        # Execute the code
        exec(code, exec_globals)
        
        # Get output
        output = sys.stdout.getvalue()
        return f"Execution successful:\n{output}" if output else "Execution successful (no output)"
        
    except Exception as e:
        return f"Execution error: {type(e).__name__}: {str(e)}"
    finally:
        sys.stdout = old_stdout


@tool
def analyze_code(code: str, language: str = "python") -> str:
    """
    Analyze code for potential issues.
    
    Args:
        code: Code to analyze
        language: Programming language
        
    Returns:
        Analysis results
    """
    issues = []
    
    # Basic Python analysis
    if language == "python":
        lines = code.split('\n')
        
        for i, line in enumerate(lines, 1):
            # Check for common issues
            if 'eval(' in line or 'exec(' in line:
                issues.append(f"Line {i}: Potential security risk (eval/exec)")
            if 'import os' in line and 'os.system' in code:
                issues.append(f"Line {i}: System command execution detected")
            if len(line) > 120:
                issues.append(f"Line {i}: Line too long ({len(line)} chars)")
    
    if issues:
        return "Code analysis:\n" + "\n".join(f"- {issue}" for issue in issues)
    return "Code analysis: No obvious issues found"


CODE_TOOLS = [execute_python, analyze_code]


CODE_SYSTEM_PROMPT = """You are a Code Agent specializing in Python programming and execution.

## Your Tools

1. **execute_python** - Execute Python code and get output
2. **analyze_code** - Analyze code for potential issues

## Guidelines

1. Write clean, efficient Python code
2. Include error handling in your code
3. Test your code before presenting the final solution
4. Explain what the code does

## IMPORTANT

- Execute code to verify it works
- If there's an error, debug and fix it
- Keep code simple and readable
- Add comments for complex logic

## Output Format

After writing/executing code:
1. **Code** - The Python code (in code blocks)
2. **Output** - Execution result
3. **Explanation** - What the code does
"""


class CodeAgent(BaseAgent):
    """
    Code Agent for programming tasks.
    
    Uses Memory V3 to:
    - Remember past code solutions
    - Store reusable code patterns
    - Learn from past debugging sessions
    """
    
    def __init__(self, config: AgentConfig):
        """Initialize the Code Agent."""
        super().__init__(
            name="code",
            description="Writes and executes Python code, debugging, data processing",
            system_prompt=CODE_SYSTEM_PROMPT,
            config=config,
            tools=CODE_TOOLS,
        )
    
    def invoke(self, state: AgentState) -> dict[str, Any]:
        """Execute code task."""
        logger.info(f"Code Agent starting")
        
        current_subtask = state.get("agent_scratchpad", {}).get("current_subtask", state["task"])
        
        # Check memory for similar code solutions
        memory_results = self._search_memory(f"code: {current_subtask}", top_k=2)
        memory_context = ""
        if memory_results:
            memory_context = "\n\nSimilar past solutions from memory:\n"
            for item, score in memory_results[:2]:
                memory_context += f"- {item.content[:300]}\n"
        
        messages = self._convert_messages(state)
        
        messages.append(HumanMessage(content=f"""
Please complete this coding task:

**Task:** {current_subtask}

**Context:** {state['task']}
{memory_context}
Write Python code to solve this. Execute it to verify it works.
"""))
        
        tool_results = []
        code_output = None
        max_iterations = 3
        
        for i in range(max_iterations):
            response = self.llm_with_tools.invoke(messages)
            
            if hasattr(response, 'tool_calls') and response.tool_calls:
                results = self._handle_tool_calls(response)
                tool_results.extend(results)
                
                # Check if code execution was successful
                for r in results:
                    if r["tool"] == "execute_python" and r["success"]:
                        code_output = r["output"]
                
                messages.append(response)
                for result in results:
                    messages.append(HumanMessage(
                        content=f"Tool Result ({result['tool']}): {result['output']}"
                    ))
            else:
                break
        
        # Get final response
        final_response = self.llm.invoke(messages)
        
        # Store successful code in memory
        if code_output and "error" not in code_output.lower():
            self._remember(
                f"Code solution for '{current_subtask[:50]}': {final_response.content[:400]}",
                importance=0.7,
                memory_type="fact",
            )
        
        # Create code artifact
        code_artifact = {
            "subtask": current_subtask,
            "code": final_response.content,
            "output": code_output,
            "tool_results": tool_results,
            "success": code_output and "error" not in code_output.lower(),
        }
        
        agent_message = Message(
            role="assistant",
            content=f"[Code Agent]\n\n{final_response.content}",
            name="code",
            metadata={"tool_calls": len(tool_results)}
        )
        
        completed = state.get("completed_subtasks", [])
        if current_subtask and current_subtask not in completed:
            completed = completed + [current_subtask]
        
        return {
            "current_agent": "code",
            "artifacts": {"code": code_artifact},
            "messages": agent_message,
            "completed_subtasks": completed,
        }
