from datetime import datetime
from typing import List, Optional, Literal, Dict
from pydantic import BaseModel, Field

# 1. MESSAGE SCHEMA (Raw History)
class Message(BaseModel):
    """
    Single conversation message.
    """
    role: Literal["user", "assistant"]
    content: str
    timestamp: Optional[datetime] = None

# 2. SESSION SUMMARY SCHEMA
class UserProfile(BaseModel):
    """
    User preferences and constraints extracted from conversation.
    """
    prefs: List[str] = Field(
        default_factory=list,
        description="User preferences (e.g., 'detailed explanations', 'concise answers')"
    )
    constraints: List[str] = Field(
        default_factory=list,
        description="User constraints/limitations"
    )
    background: Optional[str] = Field(
        default=None,
        description="User's background/context (e.g., 'software engineer', 'business owner', 'student')"
    )
class SessionSummary(BaseModel):
    user_profile: UserProfile = Field(default_factory=UserProfile)
    current_goal: Optional[str] = Field(
        default=None,
        description="User's main objective in this session"
    )
    topics: List[str] = Field(
        default_factory=list,
        description="Key entities/subjects discussed"
    )
    key_facts: List[str] = Field(default_factory=list, description="Important facts extracted from conversation")
    decisions: List[str] = Field(default_factory=list, description="Decisions made during session")
    open_questions: List[str] = Field(default_factory=list, description="Unresolved questions")
    todos: List[str] = Field(default_factory=list, description="Action items to follow up")

class SummarizationResult(BaseModel):
    """
    Output from Summarizer module.
    """
    session_summary: SessionSummary
    summarized_up_to_turn: Optional[int] = Field(
        default=None,
        description="Cumulative: summary covers Turn 1 up to this turn"
    )
    token_count_before: int = Field(
        description="Token count before operation"
    )
    token_count_after: int = Field(
        description="Token count after operation"
    )
    was_compressed: bool = Field(
        default=False,
        description="True = COMPRESSION (summary→smaller), False = SUMMARIZATION (messages→summary)"
    )

# 3. SESSION STATE SCHEMA
class SessionState(BaseModel):
    """
    Complete session state.
    """
    raw_messages: List[Message] = Field(default_factory=list)
    summary: Optional[SessionSummary] = None
    total_turns: int = Field(
        default=0,
        description="Total conversation turns (increments each turn)"
    )
    clarification_count: int = Field(
        default=0,
        description="Track consecutive clarification rounds to prevent infinite loops"
    )


# 4. QUERY UNDERSTANDING PIPELINE
class ContextUsage(BaseModel):
    """
    Specifies which parts of session memory to use for augmentation.
    """
    use_user_profile: bool = False
    use_key_facts: bool = False
    use_decisions: bool = False
    use_open_questions: bool = False
    use_todos: bool = False


# Step 1: Rewrite/Paraphrase 
class RewriteResult(BaseModel):
    """
    Uses LIGHT CONTEXT (last N (1-3) messages) for:
    - Resolving pronouns: "it", "that", "this"
    - Resolving references: "the above", "like before"
    
    Then detects if query is ambiguous and rewrites if needed.
    """
    original_query: str = Field(description="The original user query")
    is_ambiguous: bool = Field(description="Whether the original query was ambiguous")
    rewritten_query: Optional[str] = Field(
        default=None,
        description="Clarified version of query"
    )
    referenced_messages: List[Message] = Field(
        default_factory=list,
        description="N (1-3) recent messages used to resolve references (light context)"
    )
    context_usage: ContextUsage = Field(
        default_factory=ContextUsage,
        description="Which memory fields are needed for augmentation"
    )

#Step 2: Context Augmentation
class AugmentedContext(BaseModel):
    """
    Build an augmented context by combining:
    - The most recent N conversation messages
    - Relevant fields from short-term session memory (based on ContextUsage flags)
    """
    recent_messages: List[Message] = Field(
        default_factory=list,
        description="Most recent N messages from conversation"
    )
    memory_fields_used: List[str] = Field(
        default_factory=list,
        description="Memory fields used based on ContextUsage (e.g., ['user_profile', 'key_facts'])"
    )
    memory_context: str = Field(
        default="",
        description="Relevant fields from session summary formatted as string"
    )
    final_augmented_context: str = Field(
        description="Complete combined context ready for LLM"
    )

# Step 3: Clarifying Questions
class ClarificationResult(BaseModel):
    """
    If the query remains unclear AFTER rewriting and augmentation,
    generate 1-3 clarifying questions for the user.
    """
    needs_clarification: bool = Field(
        description="Whether clarification is needed after augmentation"
    )
    clarifying_questions: List[str] = Field(
        default_factory=list,
        description="1-3 clarifying questions (empty if needs_clarification=False)"
    )

# Final Query Understanding Result (full pipeline output)
class QueryUnderstandingResult(BaseModel):
    rewrite: RewriteResult
    augment: AugmentedContext
    clarify: ClarificationResult

# 5. ANSWER SCHEMA (Final Output)
class Answer(BaseModel):
    """
    Final response to user.
    """
    answer: str

# 6. PROMPT PAYLOAD SCHEMA
class PromptPayload(BaseModel):
    """
    Debug schema for prompt inspection.
    """
    system_prompt: str
    augmented_context: str
    user_query: str