"""
SimpleCoder Task Planner - Task decomposition and planning.

This module handles breaking down complex tasks into manageable subtasks
and tracking their completion.
"""

import litellm
from typing import Any


class TaskPlanner:
    # Generates a subtask list for a request and tracks what we’ve checked off.
    
    def __init__(self, model: str = "gemini/gemini-2.0-flash-exp"):
        self.model = model
        self.current_plan: dict[str, Any] | None = None
        self.completed_subtasks: list[str] = []
    
    def create_plan(self, task: str) -> str:
        # Create a task plan with subtasks.

        planning_prompt = f"""You are a task planning assistant. Break down the following task into clear, actionable subtasks.

Task: {task}

Return:
- a brief note on what needs doing
- a numbered list of subtasks, in order
- any files/actions you expect for each step

Use this format:

**Analysis:**
...

**Subtasks:**
1. ...
2. ...
"""
        
        try:
            response = litellm.completion(
                model=self.model,
                messages=[{"role": "user", "content": planning_prompt}],
                temperature=0.7
            )
            
            plan_text = response.choices[0].message.content
            
            # Store the plan
            self.current_plan = {
                "task": task,
                "plan": plan_text,
                "subtasks": self._extract_subtasks(plan_text)
            }
            
            return plan_text
        
        except Exception as e:
            return f"Error creating plan: {str(e)}"
    
    def _extract_subtasks(self, plan_text: str) -> list[str]:
        # Extract subtasks from plan text.
        
        subtasks = []
        lines = plan_text.split('\n')
        
        in_subtasks = False
        for line in lines:
            line = line.strip()
            
            if "**Subtasks:**" in line or "Subtasks:" in line:
                in_subtasks = True
                continue
            
            if in_subtasks and line:
                # Look for numbered items
                if line[0].isdigit() and '.' in line[:3]:
                    # Remove the number prefix
                    subtask = line.split('.', 1)[1].strip()
                    subtasks.append(subtask)
        
        return subtasks
    
    def mark_subtask_complete(self, subtask_description: str) -> None:
        # Mark a subtask as complete.
        
        self.completed_subtasks.append(subtask_description)
    
    def get_progress(self) -> str:
        # Get current progress on the task plan.
        
        if not self.current_plan:
            return "No active plan"
        
        total = len(self.current_plan["subtasks"])
        completed = len(self.completed_subtasks)
        
        progress = f"**Progress:** {completed}/{total} subtasks completed\n\n"
        
        if self.current_plan["subtasks"]:
            progress += "**Subtasks:**\n"
            for i, subtask in enumerate(self.current_plan["subtasks"], 1):
                status = "✓" if i <= completed else "○"
                progress += f"{status} {i}. {subtask}\n"
        
        return progress
    
    def is_complete(self) -> bool:
        # Check if all subtasks are complete.
        
        if not self.current_plan:
            return False
        
        return len(self.completed_subtasks) >= len(self.current_plan["subtasks"])
    
    def reset(self) -> None:
        # Reset the planner state.
        self.current_plan = None
        self.completed_subtasks = []