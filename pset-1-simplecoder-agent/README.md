# AI Agents @ Dartmouth College
## Problem Set 1, Part III: SimpleCoder

In this assignment, you will build **SimpleCoder**, a CLI coding agent that can help you write code, navigate codebases, and complete various software engineering tasks. This agent will use several useful concepts from the modern AI Agent development stack: tool use, Retrieval-Augmented Generation (RAG), context management, and task planning, etc.

By the end of this assignment, you should have a working coding agent that you should use to build a demo project (we suggest a simple Python project — your choice).

## Overview

SimpleCoder is a ReAct-style agent that combines tool use, semantic code search, context management, and task planning.

## Getting Started

You are provided with:
- `simplecoder/main.py` - The CLI entry point (complete, do not modify)
- `pyproject.toml` - Package configuration (complete, do not modify)

You need to implement:
- Tool functions and schemas
- Semantic code RAG for code search
- Context management with compacting
- Task planning and decomposition
- Manage user permissions for file read/write access, etc. (must support a session-level persistence option)
- The main agent logic

## Installation

```bash
pip install -e .
```

### Set API Key

```bash
export GEMINI_API_KEY="your_key_here" # You can also switch to a different provider, in case you prefer
```

### Intended Usage


Your implementation should support the following task inputs:
```bash
# Basic usage
simplecoder "create a hello.py file"

# With RAG
simplecoder --use-rag "what does the Agent class do?"

# With planning
simplecoder --use-planning "create a web server with routes for home and about"

# Interactive mode
simplecoder --interactive

# Options
simplecoder --help
```

I used claude to debug, and i used openAI API key