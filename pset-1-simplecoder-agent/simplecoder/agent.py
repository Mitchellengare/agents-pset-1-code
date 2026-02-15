
""" Agent - Main ReAct loop implementation.

 This module implements the core agent logic using a ReAct (Reasoning + Acting) loop.
 The agent alternates between reasoning about what to do next and executing tool calls until the task is complete.
"""


import json
import os
from typing import Any
import litellm
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from simplecoder.tools import get_tools, execute_tool
from simplecoder.rag import RAGSystem
from simplecoder.context import ContextManager
from simplecoder.planner import TaskPlanner
from simplecoder.permissions import PermissionManager


console = Console()


class Agent:
    #calls and executes
    
    def __init__(
        self,
        model: str = "gemini/gemini-2.0-flash-exp",
        max_iterations: int = 10,
        verbose: bool = False,
        use_planning: bool = False,
        use_rag: bool = False,
        rag_embedder: str = "gemini/text-embedding-004",
        rag_index_pattern: str = "**/*.py"
    ):
        self.model = model
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.use_planning = use_planning
        self.use_rag = use_rag
        
        # Initialize components
        self.permission_manager = PermissionManager()
        self.context_manager = ContextManager(max_tokens=100000)
        self.planner = TaskPlanner(model=model) if use_planning else None
        self.rag = None
        
        if use_rag:
            self.rag = RAGSystem(
                embedder_model=rag_embedder,
                index_pattern=rag_index_pattern
            )
            if self.verbose:
                console.print("[cyan]Building RAG index...[/cyan]")
            self.rag.index_codebase()
        
        # Get available tools
        self.tools = get_tools()
        
        # System prompt
        self.system_prompt = self._build_system_prompt()
    
    def _build_system_prompt(self) -> str:
        # Build the system prompt with tool descriptions
        tools_desc = "\n".join([
            f"- {tool['name']}: {tool['description']}"
            for tool in self.tools
        ])
        
        prompt = f"""You are SimpleCoder, a coding assistant.

Available tools:
{tools_desc}

If you want to call a tool, output a JSON object with this format:
{{"tool": "tool_name", "args": {{"arg1": "value1", "arg2": "value2"}}}}

After using a tool, you'll see the result and can decide what to do next.

After a tool runs, you'll get its result and can continue.

When you're done, return a normal response (no tool JSON).
"""
        
        if self.use_planning:
            prompt += "\nYou may draft a short plan for multi-step tasks."
        
        if self.use_rag:
            prompt += "\nYou may use search_code to locate relevant files/snippets."
        
        return prompt
    
    def run(self, task: str) -> str:
        """
        Execute the agent on a task using ReAct loop.
        
        Args:
            task: The user's task description
            
        Returns:
            The agent's final response
        """
        # Initialize context with system prompt and task
        self.context_manager.add_message("system", self.system_prompt)
        
        # If using planning, create a plan first
        if self.planner:
            if self.verbose:
                console.print("[cyan]Creating task plan...[/cyan]")
            plan = self.planner.create_plan(task)
            
            if self.verbose:
                console.print(Panel(
                    Markdown(f"**Plan:**\n{plan}"),
                    title="Task Plan",
                    border_style="cyan"
                ))
            
            # Add plan to context
            task_with_plan = f"{task}\n\n**Task Plan:**\n{plan}"
            self.context_manager.add_message("user", task_with_plan)
        else:
            self.context_manager.add_message("user", task)
        
        # ReAct loop
        for iteration in range(self.max_iterations):
            if self.verbose:
                console.print(f"\n[yellow]Iteration {iteration + 1}/{self.max_iterations}[/yellow]")
            
            # Get agent response
            messages = self.context_manager.get_messages()
            
            try:
                response = litellm.completion(
                    model=self.model,
                    messages=messages,
                    temperature=0.7
                )
                
                agent_message = response.choices[0].message.content
                
                if self.verbose:
                    console.print(Panel(
                        Markdown(agent_message),
                        title="Agent Thinking",
                        border_style="blue"
                    ))
                
                # Add to context
                self.context_manager.add_message("assistant", agent_message)
                
                # Check if agent is trying to use a tool
                tool_call = self._extract_tool_call(agent_message)
                
                if tool_call is None:
                    # No tool call - agent is done
                    return agent_message
                
                # Execute tool with permission check
                tool_name = tool_call["tool"]
                tool_args = tool_call.get("args", {})
                
                if self.verbose:
                    console.print(f"[green]Executing tool:[/green] {tool_name}")
                    console.print(f"[green]Args:[/green] {tool_args}")
                
                # Check permissions
                if not self._check_permission(tool_name, tool_args):
                    result = "Permission denied. User did not grant permission for this operation."
                else:
                    # Execute tool
                    try:
                        result = execute_tool(
                            tool_name,
                            tool_args,
                            rag_system=self.rag,
                            permission_manager=self.permission_manager
                        )
                    except Exception as e:
                        result = f"Error executing tool: {str(e)}"
                
                if self.verbose:
                    console.print(Panel(
                        str(result),
                        title="Tool Result",
                        border_style="green"
                    ))
                
                # Add tool result to context
                self.context_manager.add_message(
                    "user",
                    f"Tool '{tool_name}' result:\n{result}"
                )
                
                # Update plan if using planner
                if self.planner and tool_name in ["write_file", "edit_file"]:
                    self.planner.mark_subtask_complete(f"Used {tool_name}")
            
            except Exception as e:
                error_msg = f"Error in agent loop: {str(e)}"
                if self.verbose:
                    console.print(f"[red]{error_msg}[/red]")
                return error_msg
        
        return "Task incomplete. Reached maximum iterations."
    
    def _extract_tool_call(self, message: str) -> dict[str, Any] | None:
        """
        Extract tool call from agent message.
        
        Looks for JSON objects in the format:
        {"tool": "tool_name", "args": {...}}
        """
        # Try to find JSON in the message
        try:
            # Look for JSON code blocks
            if "```json" in message:
                json_start = message.find("```json") + 7
                json_end = message.find("```", json_start)
                json_str = message[json_start:json_end].strip()
            elif "```" in message:
                json_start = message.find("```") + 3
                json_end = message.find("```", json_start)
                json_str = message[json_start:json_end].strip()
            else:
                # Try to find raw JSON
                json_start = message.find("{")
                json_end = message.rfind("}") + 1
                if json_start == -1 or json_end == 0:
                    return None
                json_str = message[json_start:json_end]
            
            tool_call = json.loads(json_str)
            
            # Validate structure
            if "tool" in tool_call:
                return tool_call
            
        except (json.JSONDecodeError, ValueError):
            pass
        
        return None
    
    def _check_permission(self, tool_name: str, tool_args: dict) -> bool:
        """
        Check if user has granted permission for this operation.
        
        Args:
            tool_name: Name of the tool
            tool_args: Tool arguments
            
        Returns:
            True if permission granted, False otherwise
        """
        # Determine permission type needed
        if tool_name == "read_file":
            filepath = tool_args.get("filepath", "")
            return self.permission_manager.check_permission("read", filepath)
        
        elif tool_name in ["write_file", "edit_file"]:
            filepath = tool_args.get("filepath", "")
            return self.permission_manager.check_permission("write", filepath)
        
        elif tool_name == "list_files":
            directory = tool_args.get("directory", ".")
            return self.permission_manager.check_permission("read", directory)
        
        elif tool_name == "search_code":
            # Read-only operation
            return True
        
        # Default: allow
        return True