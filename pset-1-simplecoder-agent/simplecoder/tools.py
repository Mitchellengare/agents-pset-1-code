"""
SimpleCoder Tools - File operations and code manipulation tools.

This module provides tools for listing, reading, searching, writing, and editing files.
"""

import os
import glob
from pathlib import Path
from typing import Any


def get_tools() -> list[dict[str, Any]]:
    # Get the list of available tools with their schemas.
    
    # Returns: List of tool definitions

    return [
        {
            "name": "list_files",
            "description": "List files in a directory with optional pattern matching",
            "parameters": {
                "directory": {"type": "string", "description": "Directory path (default: current)"},
                "pattern": {"type": "string", "description": "Glob pattern (e.g., '*.py')"}
            }
        },
        {
            "name": "read_file",
            "description": "Read the contents of a file",
            "parameters": {
                "filepath": {"type": "string", "description": "Path to the file", "required": True}
            }
        },
        {
            "name": "search_code",
            "description": "Search for code using semantic similarity (RAG)",
            "parameters": {
                "query": {"type": "string", "description": "Search query", "required": True},
                "top_k": {"type": "integer", "description": "Number of results (default: 5)"}
            }
        },
        {
            "name": "write_file",
            "description": "Write content to a file (creates or overwrites)",
            "parameters": {
                "filepath": {"type": "string", "description": "Path to the file", "required": True},
                "content": {"type": "string", "description": "Content to write", "required": True}
            }
        },
        {
            "name": "edit_file",
            "description": "Edit a file by replacing old content with new content",
            "parameters": {
                "filepath": {"type": "string", "description": "Path to the file", "required": True},
                "old_content": {"type": "string", "description": "Content to replace", "required": True},
                "new_content": {"type": "string", "description": "New content", "required": True}
            }
        }
    ]


def execute_tool(
    tool_name: str,
    args: dict[str, Any],
    rag_system: Any = None,
    permission_manager: Any = None
) -> str:
    # Execute a tool by name with given arguments.
    
    if tool_name == "list_files":
        return list_files(**args)
    elif tool_name == "read_file":
        return read_file(**args)
    elif tool_name == "search_code":
        if rag_system is None:
            return "Error: RAG system not initialized. Use --use-rag flag."
        return search_code(rag_system=rag_system, **args)
    elif tool_name == "write_file":
        return write_file(**args)
    elif tool_name == "edit_file":
        return edit_file(**args)
    else:
        return f"Error: Unknown tool '{tool_name}'"


def list_files(directory: str = ".", pattern: str = "*") -> str:
    # List files in a directory with optional pattern matching.
    
    try:
        path = Path(directory).resolve()
        
        if not path.exists():
            return f"Error: Directory '{directory}' does not exist"
        
        if not path.is_dir():
            return f"Error: '{directory}' is not a directory"
        
        # Use glob to find matching files
        search_pattern = str(path / pattern)
        files = glob.glob(search_pattern, recursive=False)
        
        # Also try recursive if pattern contains **
        if "**" in pattern:
            files = glob.glob(search_pattern, recursive=True)
        
        if not files:
            return f"No files found matching pattern '{pattern}' in '{directory}'"
        
        # Format output
        result = f"Files in '{directory}' matching '{pattern}':\n"
        for f in sorted(files):
            rel_path = os.path.relpath(f, directory)
            result += f"  - {rel_path}\n"
        
        return result.strip()
    
    except Exception as e:
        return f"Error listing files: {str(e)}"


def read_file(filepath: str) -> str:
    # Read the contents of a file.
    
    try:
        path = Path(filepath).resolve()
        
        if not path.exists():
            return f"Error: File '{filepath}' does not exist"
        
        if not path.is_file():
            return f"Error: '{filepath}' is not a file"
        
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return f"Contents of '{filepath}':\n```\n{content}\n```"
    
    except UnicodeDecodeError:
        return f"Error: File '{filepath}' is not a text file (binary content)"
    except Exception as e:
        return f"Error reading file: {str(e)}"


def search_code(rag_system: Any, query: str, top_k: int = 5) -> str:
    # Search for code using semantic similarity.
    
    try:
        results = rag_system.search(query, top_k=top_k)
        
        if not results:
            return f"No results found for query: '{query}'"
        
        output = f"Search results for '{query}':\n\n"
        
        for i, result in enumerate(results, 1):
            output += f"**Result {i}** (similarity: {result['score']:.3f})\n"
            output += f"File: {result['file']}\n"
            output += f"Type: {result['type']}\n"
            
            if result['type'] == 'function':
                output += f"Function: {result['name']}\n"
            elif result['type'] == 'class':
                output += f"Class: {result['name']}\n"
            
            output += f"```python\n{result['code']}\n```\n\n"
        
        return output.strip()
    
    except Exception as e:
        return f"Error searching code: {str(e)}"


def write_file(filepath: str, content: str) -> str:
    # Write content to a file (creates or overwrites).
    
    try:
        path = Path(filepath).resolve()
        
        # Create parent directories if they don't exist
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return f"Successfully wrote {len(content)} characters to '{filepath}'"
    
    except Exception as e:
        return f"Error writing file: {str(e)}"


def edit_file(filepath: str, old_content: str, new_content: str) -> str:
    # Edit a file by replacing old content with new content.
    
    try:
        path = Path(filepath).resolve()
        
        if not path.exists():
            return f"Error: File '{filepath}' does not exist"
        
        # Read current content
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if old_content exists
        if old_content not in content:
            return f"Error: Content to replace not found in '{filepath}'"
        
        # Replace content
        new_file_content = content.replace(old_content, new_content, 1)
        
        # Write back
        with open(path, 'w', encoding='utf-8') as f:
            f.write(new_file_content)
        
        return f"Successfully edited '{filepath}'"
    
    except Exception as e:
        return f"Error editing file: {str(e)}"