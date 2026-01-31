import tiktoken
from typing import List
from app.core.schemas import Message, SessionSummary

# Initialize encoding
_encoding = None

def get_encoding():
    """Get or initialize the tiktoken encoding."""
    global _encoding
    if _encoding is None:
        _encoding = tiktoken.get_encoding("cl100k_base")
    return _encoding


def count_tokens(text: str) -> int:
    """
    Count tokens in a single text string.
    
    Args:
        text: The text to count tokens for
        
    Returns:
        Number of tokens
    """
    if not text:
        return 0
    
    encoding = get_encoding()
    return len(encoding.encode(text))


def count_messages_tokens(messages: List[Message]) -> int:
    """
    Count tokens in a list of messages.
    
    Uses the same format as OpenAI ChatCompletion API:
    - Each message has overhead (role + formatting)
    - 4 tokens per message for metadata
    - 3 tokens for priming (assistant reply)
    
    Args:
        messages: List of Message objects
        
    Returns:
        Total number of tokens
    """
    if not messages:
        return 0
    
    encoding = get_encoding()
    num_tokens = 0
    
    for message in messages:
        # 4 tokens per message for metadata
        # Breakdown: <|im_start|> (1) + role_wrapper (1) + <|im_end|> (1) + newline (1)
        num_tokens += 4
        
        # Count tokens in role
        num_tokens += len(encoding.encode(message.role))
        
        # Count tokens in content
        num_tokens += len(encoding.encode(message.content))
    
    # Add 3 tokens for priming assistant reply
    # Format: <|im_start|>assistant (prepares next turn)
    num_tokens += 3
    
    return num_tokens


def count_summary_tokens(summary: SessionSummary) -> int:
    """
    Count tokens in a SessionSummary.
    
    Serializes all fields to text and counts total tokens.
    
    Args:
        summary: SessionSummary object
        
    Returns:
        Total number of tokens
    """
    if not summary:
        return 0
    
    # Build text representation of summary
    text_parts = []
    
    # User profile
    if summary.user_profile:
        if summary.user_profile.prefs:
            text_parts.append(f"Preferences: {', '.join(summary.user_profile.prefs)}")
        if summary.user_profile.constraints:
            text_parts.append(f"Constraints: {', '.join(summary.user_profile.constraints)}")
        if summary.user_profile.background:
            text_parts.append(f"Background: {summary.user_profile.background}")
    
    # Current goal
    if summary.current_goal:
        text_parts.append(f"Goal: {summary.current_goal}")
    
    # Topics
    if summary.topics:
        text_parts.append(f"Topics: {', '.join(summary.topics)}")
    
    # Key facts
    if summary.key_facts:
        text_parts.append(f"Facts: {'; '.join(summary.key_facts)}")
    
    # Decisions
    if summary.decisions:
        text_parts.append(f"Decisions: {'; '.join(summary.decisions)}")
    
    # Open questions
    if summary.open_questions:
        text_parts.append(f"Questions: {'; '.join(summary.open_questions)}")
    
    # Todos
    if summary.todos:
        text_parts.append(f"Todos: {'; '.join(summary.todos)}")
    
    # Join all parts and count tokens
    full_text = "\n".join(text_parts)
    return count_tokens(full_text)