# Comprehensive Example: Schema Flow

## Configuration
```python
TOKEN_THRESHOLD_RAW = 10000      # Trigger summarization when raw_messages exceeds this
SUMMARY_TOKEN_THRESHOLD = 2000   # Trigger compression when summary exceeds this
KEEP_RECENT_N = 10               # Messages to keep after summarization
MAX_CLARIFICATION_ROUNDS = 2     # Max clarification attempts before giving up
```

---

## Turn 1: Straightforward Query

**User**: "I want to build a REST API for an e-commerce platform"

```python
# Initial state
session_state = SessionState(
    raw_messages=[], 
    summary=None, 
    total_turns=0,  # Track turn number
    clarification_count=0
)

# Step 0: Check tokens (0 < 10k) → No summarization needed

# Step 1: Rewrite
rewrite_result = RewriteResult(
    original_query="I want to build a REST API for an e-commerce platform",
    is_ambiguous=False,
    rewritten_query=None,
    referenced_messages=[],
    context_usage=ContextUsage()  # All flags False - no memory needed
)

# Step 2: Augmentation (summary=None → empty context)
augmented_context = AugmentedContext(
    recent_messages=[],
    memory_fields_used=[],
    memory_context="",
    final_augmented_context=""
)

# Step 3: Clarification → Not needed

# Generate answer
answer = Answer(answer="I recommend FastAPI or Django REST Framework...")

# Update session
session_state.raw_messages.append(Message(role="user", content="I want to build..."))
session_state.raw_messages.append(Message(role="assistant", content=answer.answer))
session_state.total_turns += 1  # Increment turn counter
```

---

## Turn 5: Ambiguous but Resolvable

**User**: "What about it?"

```python
# State: 8 messages (~500 tokens), summary=None
# session_state.total_turns = 4

# Step 1: Rewrite - Pronoun "it" detected, resolved via light context
light_context = session_state.raw_messages[-3:]  # Last 3 messages mention FastAPI

rewrite_result = RewriteResult(
    original_query="What about it?",
    is_ambiguous=True,
    rewritten_query="What about FastAPI's async support?",  # Resolved
    referenced_messages=light_context,
    context_usage=ContextUsage()  # No memory needed - recent context enough
)

# Step 2: Augmentation
augmented_context = AugmentedContext(
    recent_messages=light_context,
    memory_fields_used=[],
    memory_context="",
    final_augmented_context="Recent context: [FastAPI discussions]"
)

# Step 3: Clarification not needed
clarification_result = ClarificationResult(
    needs_clarification=False,
    clarifying_questions=[]
)

# Full pipeline result (for logging/debugging)
pipeline_result = QueryUnderstandingResult(
    rewrite=rewrite_result,
    augment=augmented_context,
    clarify=clarification_result
)

# Generate answer
session_state.total_turns += 1  # Now = 5
```

---

## Turn 10: Ambiguous and Unresolvable → Ask Clarification

**User**: "How do I set it up?"

```python
# State: 18 messages (~2000 tokens), summary=None
# Recent messages discuss FastAPI, PostgreSQL, Docker, deployment...

# Step 1: Rewrite - "it" is ambiguous even with light context
## Turn 10: Ambiguous and Unresolvable → Ask Clarification

**User**: "How do I set it up?"

```python
# State: 18 messages (~2000 tokens), summary=None
# session_state.total_turns = 9
# Recent messages discuss FastAPI, PostgreSQL, Docker, deployment...

# Step 1: Rewrite - "it" is ambiguous even with light context
rewrite_result = RewriteResult(
    original_query="How do I set it up?",
    is_ambiguous=True,
    rewritten_query="How do I set up [deployment/database/FastAPI]?",  # Still unclear
    referenced_messages=light_context,
    context_usage=ContextUsage(use_key_facts=True, use_decisions=True)
)

# Step 2: Augmentation - But summary=None, so no memory available
augmented_context = AugmentedContext(
    recent_messages=session_state.raw_messages[-5:],
    memory_fields_used=[],  # Wanted key_facts but summary=None
    memory_context="",
    final_augmented_context="Recent: [FastAPI, PostgreSQL, Docker, deployment discussions]"
)

# Step 3: Clarification needed
clarification_result = ClarificationResult(
    needs_clarification=True,
    clarifying_questions=[
        "What do you want to set up?",
        "1) FastAPI project? 2) PostgreSQL? 3) Docker? 4) Deployment?"
    ]
)

session_state.clarification_count += 1  # = 1
session_state.total_turns += 1  # Now = 10

# Full pipeline result
pipeline_result = QueryUnderstandingResult(
    rewrite=rewrite_result,
    augment=augmented_context,
    clarify=clarification_result
)

# Return clarification to user
return clarification_result.clarifying_questions
```

---

## Turn 11: User Clarifies

**User**: "I mean the Docker setup"

```python
# session_state.total_turns = 10
# Query now clear → Generate answer
answer = Answer(answer="For Docker setup, you need...")

session_state.clarification_count = 0  # Reset after successful answer
session_state.total_turns += 1  # Now = 11
```

---

## Turn 20: Trigger Summarization

**User**: "What's the best way to handle authentication?"

```python
# State: 38 messages (~10,500 tokens) → Exceeds threshold!
# session_state.total_turns = 19 (after Turn 19 completed)

# Step 0: SUMMARIZATION triggered
# Summarize ALL messages up to current turn
summarized_up_to = session_state.total_turns  # 19 - summarize all Turn 1-19

new_summary = SessionSummary(
    user_profile=UserProfile(
        prefs=["Detailed explanations", "Code examples"],
        constraints=["3 developers", "3-month timeline"],
        background="Software engineer"
    ),
    current_goal="Build production REST API for e-commerce with FastAPI",
    topics=["FastAPI", "PostgreSQL", "Docker", "Deployment"],
    key_facts=["Using FastAPI async", "PostgreSQL for 100k users", "Docker containerization"],
    decisions=["FastAPI over Django", "PostgreSQL", "Docker", "AWS"],
    open_questions=["Authentication approach?"],
    todos=["Docker compose", "Database schema", "Auth middleware"]
)

summarization_result = SummarizationResult(
    session_summary=new_summary,
    summarized_up_to_turn=19,  # Summary covers Turn 1-19
    token_count_before=10500,
    token_count_after=800,
    was_compressed=False
)

# Keep recent messages (overlap is intentional - see note below)
# KEEP_RECENT_N = 10, so keep last 10 messages (Turn 10-19)
session_state.raw_messages = session_state.raw_messages[-10:]
session_state.summary = new_summary
session_state.summarized_up_to_turn = 19  # Track cumulative summarization
session_state.total_turns += 1  # Now = 20

# Step 1: Rewrite with summary available
rewrite_result = RewriteResult(
    original_query="What's the best way to handle authentication?",
    is_ambiguous=False,
    context_usage=ContextUsage(
        use_user_profile=True,
        use_key_facts=True,
        use_decisions=True
    )
)

# Step 2: Augmentation now pulls from summary
augmented_context = AugmentedContext(
    recent_messages=session_state.raw_messages[-5:],
    memory_fields_used=["user_profile", "key_facts", "decisions"],
    memory_context="User: software engineer. Stack: FastAPI + PostgreSQL + Docker + AWS",
    final_augmented_context="[Combined context for LLM]"
)

# Step 3: Clarification not needed
clarification_result = ClarificationResult(
    needs_clarification=False,
    clarifying_questions=[]
)

# Full pipeline result
pipeline_result = QueryUnderstandingResult(
    rewrite=rewrite_result,
    augment=augmented_context,
    clarify=clarification_result
)

# Generate answer using augmented context
answer = Answer(answer="For authentication with FastAPI...")
```

### Why Overlap is Intentional

**Overlap**: Summary covers Turn 1-19, but `raw_messages` keeps Turn 10-19 (10 recent messages).

Summary and recent messages serve **different purposes**:
- **Summary**: Compressed knowledge (facts, decisions) - loses exact wording
- **Recent messages**: Exact context - needed for pronoun resolution ("it", "that")

Overlap cost (~500 tokens for Turn 10-19) is worth the benefit of accurate reference resolution.

---

## Turn 50: Trigger Compression

**User**: "Can you summarize our discussion?"

```python
# State: raw_messages ~550 tokens (OK), summary ~2300 tokens (exceeds 2k!)
# session_state.total_turns = 49

# COMPRESSION: Condense summary only, DO NOT touch raw_messages
compressed_summary = summarizer.compress_summary(session_state.summary)
# LLM condenses: 20 topics → 8, 35 facts → 14, 18 decisions → 8
# Keep the same summarized_up_to_turn from previous summarization

summarization_result = SummarizationResult(
    session_summary=compressed_summary,
    summarized_up_to_turn=session_state.summarized_up_to_turn,  # Keep tracking value (e.g., 19)
    token_count_before=2300,
    token_count_after=1400,
    was_compressed=True
)

session_state.summary = compressed_summary
# session_state.summarized_up_to_turn: UNCHANGED (still 19)
session_state.total_turns += 1  # Now = 50
# raw_messages: UNCHANGED
```

---

## Turn 60-61: Clarification Loop Limit

**Turn 60: User**: "help"

```python
# session_state.total_turns = 59
clarification_result = ClarificationResult(
    needs_clarification=True,
    clarifying_questions=["What do you need help with?"]
)
session_state.clarification_count += 1  # = 1
session_state.total_turns += 1  # Now = 60
```

**Turn 61: User**: "anything"

```python
# session_state.total_turns = 60
session_state.clarification_count += 1  # = 2

if session_state.clarification_count >= MAX_CLARIFICATION_ROUNDS:
    return Answer(answer="Please describe your specific question or problem.")
    session_state.clarification_count = 0
session_state.total_turns += 1  # Now = 61
```

---

## Summary

### Token Flow
```
Turn 1-19:   raw_messages: 0 → 10.5k, summary: None, summarized_up_to_turn: None
Turn 20:     SUMMARIZATION → raw_messages: 550, summary: 800, summarized_up_to_turn: 19
Turn 21-49:  raw_messages grows, summary grows to 2.3k
Turn 50:     COMPRESSION → raw_messages: 550, summary: 1.4k, summarized_up_to_turn: 19 (unchanged)
Turn 51-69:  raw_messages: 550 → 10.8k, summary: 1.4k
Turn 70:     SUMMARIZATION → raw_messages: 550, summary: 2.0k, summarized_up_to_turn: 70 (cumulative)
```

### Operation Comparison

| Aspect | SUMMARIZATION | COMPRESSION |
|--------|---------------|-------------|
| Trigger | raw_messages > 10k | summary > 2k |
| Input | raw_messages | summary only |
| Affects raw_messages? | Yes (keep N recent) | No |
| summarized_up_to_turn | Set to current turn | Keep previous value |
| was_compressed | False | True |

### Edge Cases

| Case | Turn | Handling |
|------|------|----------|
| Straightforward query | 1 | Direct answer |
| Ambiguous, resolvable | 5 | Rewrite → answer |
| Ambiguous, unresolvable | 10 | Ask clarification |
| User clarifies | 11 | Answer, reset counter |
| No summary yet | 1-19 | Recent messages only |
| Trigger summarization | 20 | Extract → summary, keep N |
| Trigger compression | 50 | Compress summary only |
| Clarification loop | 60-61 | Max 2 rounds |
