"""
SimpleCoder Permission Manager - User permission management.

This module handles user permissions for file operations, supporting both
session-level and task-level permissions with persistent storage.
"""

import json
import os
from pathlib import Path
from typing import Literal
from rich.console import Console
from rich.prompt import Confirm


console = Console()


PermissionType = Literal["read", "write", "execute"]


class PermissionManager:
    """
    Manages user permissions for file operations.
    
    Supports:
    - Session-level permissions (persistent across tasks)
    - Task-level permissions (cleared after each task)
    - Interactive permission prompts
    - Permission storage/loading
    """
    
    def __init__(self, storage_path: str = ".simplecoder_permissions.json"):
        self.storage_path = Path.home() / storage_path
        
        # Session permissions (persistent)
        self.session_permissions: dict[str, set[str]] = {
            "read": set(),
            "write": set(),
            "execute": set()
        }
        
        # Task permissions (temporary)
        self.task_permissions: dict[str, set[str]] = {
            "read": set(),
            "write": set(),
            "execute": set()
        }
        
        # Load session permissions
        self._load_session_permissions()
    
    def check_permission(
        self,
        permission_type: PermissionType,
        path: str,
        interactive: bool = True
    ) -> bool:
        """
        Check if permission is granted for an operation on a path.
        
        Args:
            permission_type: Type of permission (read, write, execute)
            path: File or directory path
            interactive: If True, prompt user if permission not found
            
        Returns:
            True if permission granted, False otherwise
        """
        # Normalize path
        normalized_path = str(Path(path).resolve())
        
        # Check if permission already granted
        if self._has_permission(permission_type, normalized_path):
            return True
        
        # If not interactive, deny
        if not interactive:
            return False
        
        # Prompt user
        return self._prompt_permission(permission_type, normalized_path)
    
    def _has_permission(self, permission_type: PermissionType, path: str) -> bool:
        """
        Check if permission exists in session or task permissions.
        
        Args:
            permission_type: Type of permission
            path: Normalized path
            
        Returns:
            True if permission exists
        """
        # Check exact match
        if path in self.session_permissions[permission_type]:
            return True
        if path in self.task_permissions[permission_type]:
            return True
        
        # Check parent directory permissions
        path_obj = Path(path)
        for parent in path_obj.parents:
            parent_str = str(parent)
            if parent_str in self.session_permissions[permission_type]:
                return True
            if parent_str in self.task_permissions[permission_type]:
                return True
        
        return False
    
    def _prompt_permission(self, permission_type: PermissionType, path: str) -> bool:
        """
        Prompt user for permission.
        
        Args:
            permission_type: Type of permission
            path: Path requiring permission
            
        Returns:
            True if user grants permission
        """
        console.print(f"\n[yellow]Permission Request[/yellow]")
        console.print(f"Action: [bold]{permission_type}[/bold]")
        console.print(f"Path: [cyan]{path}[/cyan]")
        
        # Ask if user wants to grant permission
        granted = Confirm.ask("Grant permission?", default=False)
        
        if not granted:
            return False
        
        # Ask if this should be session-level (persistent)
        session_level = Confirm.ask(
            "Remember this permission for future tasks? (session-level)",
            default=False
        )
        
        if session_level:
            self.session_permissions[permission_type].add(path)
            self._save_session_permissions()
            console.print("[green]Permission granted for this session[/green]")
        else:
            self.task_permissions[permission_type].add(path)
            console.print("[green]Permission granted for this task[/green]")
        
        return True
    
    def grant_permission(
        self,
        permission_type: PermissionType,
        path: str,
        session_level: bool = False
    ) -> None:
        """
        Programmatically grant a permission.
        
        Args:
            permission_type: Type of permission
            path: Path to grant permission for
            session_level: If True, make it session-level
        """
        normalized_path = str(Path(path).resolve())
        
        if session_level:
            self.session_permissions[permission_type].add(normalized_path)
            self._save_session_permissions()
        else:
            self.task_permissions[permission_type].add(normalized_path)
    
    def revoke_permission(
        self,
        permission_type: PermissionType,
        path: str
    ) -> None:
        """
        Revoke a permission.
        
        Args:
            permission_type: Type of permission
            path: Path to revoke permission for
        """
        normalized_path = str(Path(path).resolve())
        
        self.session_permissions[permission_type].discard(normalized_path)
        self.task_permissions[permission_type].discard(normalized_path)
        self._save_session_permissions()
    
    def clear_task_permissions(self) -> None:
        """Clear all task-level permissions."""
        self.task_permissions = {
            "read": set(),
            "write": set(),
            "execute": set()
        }
    
    def clear_session_permissions(self) -> None:
        """Clear all session-level permissions."""
        self.session_permissions = {
            "read": set(),
            "write": set(),
            "execute": set()
        }
        self._save_session_permissions()
    
    def _save_session_permissions(self) -> None:
        """Save session permissions to disk."""
        try:
            data = {
                ptype: list(paths)
                for ptype, paths in self.session_permissions.items()
            }
            
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
        
        except Exception as e:
            console.print(f"[yellow]Warning: Could not save permissions: {e}[/yellow]")
    
    def _load_session_permissions(self) -> None:
        """Load session permissions from disk."""
        if not self.storage_path.exists():
            return
        
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
            
            self.session_permissions = {
                ptype: set(paths)
                for ptype, paths in data.items()
            }
        
        except Exception as e:
            console.print(f"[yellow]Warning: Could not load permissions: {e}[/yellow]")
    
    def get_permissions_summary(self) -> str:
        """
        Get a summary of current permissions.
        
        Returns:
            Formatted summary string
        """
        summary = "**Current Permissions:**\n\n"
        
        for ptype in ["read", "write", "execute"]:
            session_perms = self.session_permissions[ptype]
            task_perms = self.task_permissions[ptype]
            
            summary += f"**{ptype.capitalize()}:**\n"
            
            if session_perms:
                summary += "  Session-level:\n"
                for path in sorted(session_perms):
                    summary += f"    - {path}\n"
            
            if task_perms:
                summary += "  Task-level:\n"
                for path in sorted(task_perms):
                    summary += f"    - {path}\n"
            
            if not session_perms and not task_perms:
                summary += "  (none)\n"
            
            summary += "\n"
        
        return summary