"""
Pydantic schemas for ChatTMT - Chat Assistant with Session Memory.

This module defines all structured outputs used throughout the pipeline.
Schemas are organized by their purpose in the system.

Schema Mapping:
    - Message          → session.py, summarizer.py
    - SessionSummary   → summarizer, augmenter, decision
    - SessionState     → pipeline.py
    - QueryDecision    → decision.py
    - Clarification    → clarifier.py
    - Answer           → answer.py
    - PromptPayload    → prompt_builder, UI
"""

from datetime import datetime
from typing import List, Optional, Literal, Dict
from pydantic import BaseModel, Field


# =============================================================================
# 1. MESSAGE SCHEMA (Raw History)
# =============================================================================

class Message(BaseModel):
    """
    Single conversation message.
    Used for: raw_messages, chat rendering, summarization input.
    """
    role: Literal["user", "assistant"]
    content: str
    timestamp: Optional[datetime] = None  # Optional for tracking


# =============================================================================
# 2. SESSION SUMMARY SCHEMA (Long-term Memory)
# =============================================================================

class SessionSummary(BaseModel):
    """
    Compressed representation of conversation history.
    
    This is the CENTER of context augmentation.
    - All modules READ from this
    - Only Summarizer WRITES to this
    """
    known_facts: List[str] = Field(default_factory=list, description="Important facts extracted from conversation")
    user_intents: List[str] = Field(default_factory=list, description="User's goals and intentions")
    decisions: List[str] = Field(default_factory=list, description="Decisions made during session")
    open_questions: List[str] = Field(default_factory=list, description="Unresolved questions")


class SummarizationResult(BaseModel):
    """
    Output from Summarizer module.
    Wraps SessionSummary with metadata about what was summarized.
    """
    session_summary: SessionSummary
    message_range_summarized: Dict[str, int] = Field(
        description="Range of messages summarized, e.g., {'from': 0, 'to': 42}"
    )
    token_count_before: int
    token_count_after: int
    summarized_at: datetime = Field(default_factory=datetime.now)


# =============================================================================
# 3. SESSION STATE SCHEMA (What the System Remembers)
# =============================================================================

class SessionState(BaseModel):
    """
    Complete session state - what the system "remembers".
    
    Contains ONLY memory-related data:
    - NO stats
    - NO config  
    - NO runtime info
    """
    raw_messages: List[Message] = Field(default_factory=list)
    summary: Optional[SessionSummary] = None


# =============================================================================
# 4. DECISION SCHEMA (System Brain)
# =============================================================================

class ContextUsage(BaseModel):
    """
    Specifies which parts of session memory to use for augmentation.
    """
    use_known_facts: bool = False
    use_user_intents: bool = False
    use_decisions: bool = False
    use_open_questions: bool = False


class QueryDecision(BaseModel):
    """
    Output from Decision module - the BRAIN of the system.
    
    This is the MOST IMPORTANT schema.
    If reviewer reads only one schema, they read THIS.
    
    Determines:
    - Whether to proceed with answering or ask for clarification
    - How to rewrite ambiguous queries
    - What context to pull from memory
    """
    action: Literal["PROCEED", "ASK_BACK"]
    is_ambiguous: bool = Field(description="Whether the original query was ambiguous")
    reason: str = Field(description="Explanation for the decision")
    rewritten_query: Optional[str] = Field(
        default=None, 
        description="Clarified version of query (if ambiguous)"
    )
    context_usage: ContextUsage = Field(
        default_factory=ContextUsage,
        description="Which memory fields to use for augmentation"
    )


# =============================================================================
# 5. CLARIFICATION SCHEMA (Ask-back)
# =============================================================================

class Clarification(BaseModel):
    """
    Output when system needs more information from user.
    Single clarifying question - no redundancy.
    """
    clarifying_question: str


# =============================================================================
# 6. ANSWER SCHEMA (Final Output)
# =============================================================================

class Answer(BaseModel):
    """
    Final response to user.
    Clean output - no chain-of-thought exposed.
    """
    answer: str
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional confidence score (0.0 - 1.0)"
    )


# =============================================================================
# 7. PROMPT PAYLOAD SCHEMA (Debug/Inspection)
# =============================================================================

class PromptPayload(BaseModel):
    """
    Debug schema for prompt inspection.
    
    Used for:
    - UI inspector
    - Logging
    - Demo visualization
    """
    system_prompt: str
    augmented_context: str
    user_query: str
    full_prompt: Optional[str] = Field(
        default=None,
        description="Combined prompt sent to LLM"
    )
