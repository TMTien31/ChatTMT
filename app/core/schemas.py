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
    prefs: List[str] = Field(default_factory=list, description="User preferences")
    constraints: List[str] = Field(default_factory=list, description="User constraints/limitations")

class SessionSummary(BaseModel):
    """
    Compressed representation of conversation history.
    """
    user_profile: UserProfile = Field(default_factory=UserProfile)
    key_facts: List[str] = Field(default_factory=list, description="Important facts extracted from conversation")
    decisions: List[str] = Field(default_factory=list, description="Decisions made during session")
    open_questions: List[str] = Field(default_factory=list, description="Unresolved questions")
    todos: List[str] = Field(default_factory=list, description="Action items to follow up")

class SummarizationResult(BaseModel):
    """
    Output from Summarizer module.
    """
    session_summary: SessionSummary
    message_range_summarized: Dict[str, int] = Field(
        description="Range of messages summarized, e.g., {'from': 0, 'to': 42}"
    )
    token_count_before: int
    token_count_after: int


# 3. SESSION STATE SCHEMA
class SessionState(BaseModel):
    """
    Complete session state.
    """
    raw_messages: List[Message] = Field(default_factory=list)
    summary: Optional[SessionSummary] = None


# 4. QUERY UNDERSTANDING PIPELINE
class ContextUsage(BaseModel):
    """
    Specifies which parts of session memory to use for augmentation.
    Used by Augmenter to know what to pull from session.summary.
    """
    use_user_profile: bool = False
    use_key_facts: bool = False
    use_decisions: bool = False
    use_open_questions: bool = False
    use_todos: bool = False


# Step 1: Rewrite/Paraphrase → Step 2: Context Augmentation → Step 3: Clarifying Questions
class RewriteResult(BaseModel):
    """
    Output from Step 1: Rewrite/Paraphrase.
    
    Uses LIGHT CONTEXT (last 1-3 messages) for:
    - Resolving pronouns: "it", "that", "this"
    - Resolving references: "the above", "like before"
    - Linguistic disambiguation only (NOT knowledge augmentation)
    
    Then detects if query is ambiguous and rewrites if needed.
    """
    original_query: str = Field(description="The original user query")
    is_ambiguous: bool = Field(description="Whether the original query was ambiguous")
    rewritten_query: Optional[str] = Field(
        default=None,
        description="Clarified version of query (with references resolved)"
    )
    referenced_messages: List[Message] = Field(
        default_factory=list,
        description="1-3 recent messages used to resolve references (light context)"
    )
    context_usage: ContextUsage = Field(
        default_factory=ContextUsage,
        description="Which memory fields are needed for Step 2 augmentation"
    )

class AugmentedContext(BaseModel):
    """
    Output from Step 2: Context Augmentation.
    
    Build an augmented context by combining:
    - The most recent N conversation messages
    - Relevant fields from short-term session memory
    """
    recent_messages: List[Message] = Field(
        default_factory=list,
        description="Most recent N messages from conversation"
    )
    memory_context: str = Field(
        default="",
        description="Relevant fields from session summary formatted as string"
    )
    final_augmented_context: str = Field(
        description="Complete combined context ready for LLM"
    )

class ClarificationResult(BaseModel):
    """
    Output from Step 3: Clarifying Questions.
    
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

class QueryUnderstandingResult(BaseModel):
    """
    Final output from Query Understanding Pipeline.
    """
    # Step 1: Rewrite
    original_query: str
    is_ambiguous: bool
    rewritten_query: Optional[str] = None
    
    # Step 2: Augmentation
    needed_context_from_memory: List[str] = Field(
        default_factory=list,
        description="Memory fields used (e.g., ['user_profile.prefs', 'key_facts'])"
    )
    final_augmented_context: str = Field(
        default="",
        description="Complete context after augmentation"
    )
    
    # Step 3: Clarification
    clarifying_questions: List[str] = Field(
        default_factory=list,
        description="1-3 clarifying questions (if still unclear after Steps 1-2)"
    )

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