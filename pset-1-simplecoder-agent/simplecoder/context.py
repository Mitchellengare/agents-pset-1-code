"""
SimpleCoder Context Manager - Manages conversation context with compacting.

This module handles context window management by tracking token usage and
summarizing conversation history when limits are approached.
"""

import litellm
from typing import Any


class ContextManager:
    """
    Manages conversation context with automatic summarization.
    
    When the context approaches the token limit, older messages are summarized
    to keep the conversation within bounds while preserving recent context.
    """
    
    def __init__(
        self,
        max_tokens: int = 100000,
        summary_model: str = "gemini/gemini-2.0-flash-exp",
        keep_recent: int = 5
    ):
        """
        Initialize context manager.
        
        Args:
            max_tokens: Maximum context window size
            summary_model: Model to use for summarization
            keep_recent: Number of recent messages to keep intact
        """
        self.max_tokens = max_tokens
        self.summary_model = summary_model
        self.keep_recent = keep_recent
        
        self.messages: list[dict[str, str]] = []
        self.current_tokens = 0
        self.summary: str | None = None
    
    def add_message(self, role: str, content: str) -> None:
        """
        Add a message to the context.
        
        Args:
            role: Message role (system, user, assistant)
            content: Message content
        """
        message = {"role": role, "content": content}
        self.messages.append(message)
        
        # Estimate tokens (rough approximation: 4 chars per token)
        estimated_tokens = len(content) // 4
        self.current_tokens += estimated_tokens
        
        # Check if we need to compact
        if self.current_tokens > self.max_tokens * 0.8:  # 80% threshold
            self._compact_context()
    
    def get_messages(self) -> list[dict[str, str]]:
        """
        Get the current message list for the LLM.
        
        Returns:
            List of messages, possibly with summarized history
        """
        if self.summary is None:
            return self.messages
        
        # If we have a summary, include it before recent messages
        system_msg = self.messages[0] if self.messages and self.messages[0]["role"] == "system" else None
        recent_messages = self.messages[-self.keep_recent:]
        
        result = []
        
        if system_msg:
            result.append(system_msg)
        
        # Add summary as a user message
        result.append({
            "role": "user",
            "content": f"**Previous conversation summary:**\n{self.summary}\n\n**Recent messages follow:**"
        })
        
        result.extend(recent_messages)
        
        return result
    
    def _compact_context(self) -> None:
        """
        Compact the context by summarizing older messages.
        """
        if len(self.messages) <= self.keep_recent + 1:  # +1 for system message
            return
        
        # Separate system message, messages to summarize, and recent messages
        system_msg = None
        if self.messages and self.messages[0]["role"] == "system":
            system_msg = self.messages[0]
            messages_to_summarize = self.messages[1:-self.keep_recent]
            recent_messages = self.messages[-self.keep_recent:]
        else:
            messages_to_summarize = self.messages[:-self.keep_recent]
            recent_messages = self.messages[-self.keep_recent:]
        
        if not messages_to_summarize:
            return
        
        # Create summary prompt
        conversation_text = "\n\n".join([
            f"{msg['role'].upper()}: {msg['content']}"
            for msg in messages_to_summarize
        ])
        
        summary_prompt = f"""Summarize the following conversation history concisely, preserving key information about:
- Tasks attempted and their outcomes
- Files created, read, or modified
- Important decisions or findings
- Current state of work

Conversation history:
{conversation_text}

Provide a concise summary (max 500 words):"""
        
        try:
            response = litellm.completion(
                model=self.summary_model,
                messages=[{"role": "user", "content": summary_prompt}],
                temperature=0.3
            )
            
            new_summary = response.choices[0].message.content
            
            # Combine with existing summary if present
            if self.summary:
                combined_prompt = f"""Combine these two summaries into one concise summary:

Previous summary:
{self.summary}

New summary:
{new_summary}

Combined summary (max 500 words):"""
                
                response = litellm.completion(
                    model=self.summary_model,
                    messages=[{"role": "user", "content": combined_prompt}],
                    temperature=0.3
                )
                
                self.summary = response.choices[0].message.content
            else:
                self.summary = new_summary
            
            # Update message list
            self.messages = [system_msg] + recent_messages if system_msg else recent_messages
            
            # Recalculate token count
            self.current_tokens = sum(len(msg["content"]) // 4 for msg in self.messages)
            if self.summary:
                self.current_tokens += len(self.summary) // 4
        
        except Exception as e:
            # If summarization fails, just truncate old messages
            self.messages = [system_msg] + recent_messages if system_msg else recent_messages
            self.current_tokens = sum(len(msg["content"]) // 4 for msg in self.messages)
    
    def get_token_usage(self) -> dict[str, int]:
        """
        Get current token usage statistics.
        
        Returns:
            Dictionary with token usage info
        """
        return {
            "current_tokens": self.current_tokens,
            "max_tokens": self.max_tokens,
            "usage_percent": int((self.current_tokens / self.max_tokens) * 100)
        }
    
    def clear(self) -> None:
        """Clear all context."""
        self.messages = []
        self.current_tokens = 0
        self.summary = None