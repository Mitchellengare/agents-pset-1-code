# SimpleCoder - A CLI Coding Agent

A ReAct-style coding agent with tool use, RAG, context management, and task planning capabilities.

## Installation

```bash
pip install -e .
```

## Usage

```bash
# Set your API key
export GEMINI_API_KEY="your_key_here"


# Basic usage
simplecoder "create a hello.py file that prints hello world"

# With RAG for code search
simplecoder --use-rag "what does the Agent class do?"

# With planning for complex tasks
simplecoder --use-planning "create a simple web server with Flask"

# Interactive mode
simplecoder --interactive
```

## Architecture

### agent.py - ReAct Loop Implementation

**Design Rationale:**
The agent uses a ReAct (Reasoning + Acting) loop where it alternates between thinking about what to do and executing tools. This approach is chosen because:
- **Interpretability**: Each step shows clear reasoning before action
- **Error recovery**: Failed actions can be observed and corrected
- **Flexibility**: The agent can adapt its strategy based on tool results

The implementation extracts tool calls from JSON blocks in the agent's responses, which provides a clear interface while allowing the LLM to explain its reasoning in natural language before/after tool use.

### tools.py - File Operations

**Design Rationale:**
Tools are kept simple and focused on single responsibilities:
- **list_files**: Directory navigation with glob patterns
- **read_file**: Safe file reading with error handling
- **search_code**: Semantic search via RAG
- **write_file**: File creation with directory handling
- **edit_file**: Targeted content replacement

Each tool returns formatted strings rather than complex objects to make them easy for the LLM to interpret. Error messages are descriptive to help the agent understand what went wrong and how to fix it.

### rag.py - AST-Based Code Search

**Design Rationale:**
Instead of naive line-based chunking, we use Python's AST to extract semantic units (functions, classes). This is superior because:
- **Semantic coherence**: Each chunk is a complete, meaningful code unit
- **Better retrieval**: Searching for "authentication function" returns whole functions, not random line spans
- **Context preservation**: Function signatures and docstrings stay together

We use embedding-based similarity search (cosine similarity) rather than keyword matching to handle semantic queries like "code that handles user login" even when the actual code uses different terminology.

### context.py - Context Window Management

**Design Rationale:**
Long conversations eventually exceed token limits. Our approach:
- **Summarization over truncation**: We summarize old messages rather than just dropping them, preserving important information
- **Keep recent messages intact**: The last N messages are never summarized to maintain immediate context
- **Incremental summarization**: New summaries are merged with old ones to create a coherent history

This allows the agent to work on long-running tasks without losing track of what it's done. The 80% threshold triggers compacting before we hit hard limits.

### planner.py - Task Decomposition

**Design Rationale:**
Complex tasks benefit from upfront planning. The planner:
- **Structures ambiguous requests**: Turns "build a game" into concrete subtasks
- **Provides progress tracking**: Users can see what's been done and what's next
- **Guides agent execution**: The plan serves as a roadmap in the ReAct loop

We use an LLM for planning rather than hard-coded rules because task decomposition requires understanding project context and conventions. The planner extracts numbered subtasks from the LLM's response for tracking.

### permissions.py - Permission Management

**Design Rationale:**
File operations need user consent for safety. Our two-tier system:
- **Task-level permissions**: Cleared after each task, for one-off operations
- **Session-level permissions**: Persisted to disk, for frequently accessed paths

This balances security with usability - users can grant broad permissions for their project directory once, but destructive operations on new paths still require confirmation. Permission checks traverse parent directories so granting permission to `/project` covers `/project/src/file.py`.

Interactive prompts use rich formatting to make permission requests clear and hard to miss. The permission state is stored in the user's home directory for persistence across sessions.

## Design Philosophy

1. **Transparency**: Every operation is visible and explainable
2. **Safety**: Permissions protect users from accidental/malicious operations
3. **Efficiency**: RAG and context management enable work on large codebases
4. **Flexibility**: Tools are composable; the agent chooses how to use them

## Example Session

```bash
$ simplecoder --use-planning "create a simple text adventure game"

Creating task plan...

┌─ Task Plan ──────────────────────────────────────────┐
│ **Analysis:**                                         │
│ Build a Python text adventure with rooms, items,     │
│ and basic commands.                                   │
│                                                       │
│ **Subtasks:**                                         │
│ 1. Create main game file structure                   │
│ 2. Implement room system                             │
│ 3. Add item/inventory system                         │
│ 4. Create command parser                             │
│ 5. Add sample game content                           │
└───────────────────────────────────────────────────────┘

[Agent creates files, implements features, tests...]

┌─ Agent Response ─────────────────────────────────────┐
│ Game complete! Created:                               │
│ - adventure.py (main game)                            │
│ - rooms.json (game data)                              │
│ - README.md (how to play)                             │
│                                                       │
│ Run with: python adventure.py                         │
└───────────────────────────────────────────────────────┘
```

## Future Improvements

- **Git integration**: Commit changes automatically
- **Testing tools**: Run tests and analyze results
- **Refactoring tools**: Extract functions, rename variables
- **Debugging tools**: Set breakpoints, inspect variables
- **Multi-file edits**: Coordinate changes across files