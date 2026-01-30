# Comprehensive Example: Chat Flow với Schemas

## Setup
```python
TOKEN_THRESHOLD_RAW = 10000  # Trigger summarization
SUMMARY_TOKEN_THRESHOLD = 2000  # Trigger compression
KEEP_RECENT_N = 10  # Keep messages after summarize
MAX_CLARIFICATION_ROUNDS = 2
```

---

## 🟢 TURN 1: Câu hỏi bình thường (Straightforward)

**User Input**: "I want to build a REST API for an e-commerce platform"

### Step 0: Check Token + Summarize
```python
session_state = SessionState(
    raw_messages=[],
    summary=None,  # Chưa có summary
    clarification_count=0
)

token_count = count_tokens(session_state.raw_messages)  # = 0
# 0 < 10000 → No summarization needed
```

### Step 1: Rewrite (Decision Module)
```python
# Input: query + light context (last 1-3 messages)
light_context = session_state.raw_messages[-3:]  # = [] (empty, first turn)

rewrite_result = RewriteResult(
    original_query="I want to build a REST API for an e-commerce platform",
    is_ambiguous=False,  # Clear query, no pronouns
    rewritten_query=None,  # No rewrite needed
    referenced_messages=[],  # No light context yet
    context_usage=ContextUsage(  # Detect: no memory needed (new topic)
        use_user_profile=False,
        use_key_facts=False,
        use_decisions=False,
        use_open_questions=False,
        use_todos=False
    )
)
```

### Step 2: Augmentation (Augmenter Module)
```python
# Input: context_usage flags + session_state

# Edge Case 2: Handle summary=None
if session_state.summary is None:
    memory_context = ""  # ✅ Gracefully handle
else:
    memory_context = build_memory_context(session_state.summary, context_usage)

augmented_context = AugmentedContext(
    recent_messages=[],  # No previous messages
    memory_fields_used=[],  # No flags set
    memory_context="",  # No summary yet
    final_augmented_context=""  # Empty context
)
```

### Step 3: Clarification (Clarifier Module)
```python
# Input: rewritten_query + augmented_context

clarification_result = ClarificationResult(
    needs_clarification=False,  # Query is clear
    clarifying_questions=[]
)
```

### Generate Answer
```python
# No clarification needed → Generate answer
answer = Answer(
    answer="To build a REST API for e-commerce, I recommend using FastAPI or Django REST Framework. What's your tech stack preference?"
)

# Reset clarification counter (successful answer)
session_state.clarification_count = 0  # ✅ Edge Case 4
```

### Update Session
```python
session_state.raw_messages.append(Message(
    role="user",
    content="I want to build a REST API for an e-commerce platform",
    timestamp=datetime.now()
))
session_state.raw_messages.append(Message(
    role="assistant", 
    content=answer.answer,
    timestamp=datetime.now()
))
# raw_messages: 2 messages, ~100 tokens
```

---

## 🟡 TURN 5: Câu hỏi mơ hồ NHƯNG trả lời được (Ambiguous but Resolvable)

**User Input**: "What about it?"

```python
# Current state:
session_state.raw_messages = [
    # Turn 1-4: Discussion about FastAPI vs Django
    Message(user, "I want to build REST API..."),
    Message(assistant, "I recommend FastAPI or Django..."),
    Message(user, "I prefer Python with async support"),
    Message(assistant, "Then FastAPI is perfect. It has native async/await..."),
    # Turn 5:
]
# Tokens: ~500 (still < 10k threshold)
# summary: None (not triggered yet)
```

### Step 0: No summarization (500 < 10k)

### Step 1: Rewrite
```python
# Get light context
light_context = session_state.raw_messages[-3:]  # Last 3 messages

rewrite_result = RewriteResult(
    original_query="What about it?",
    is_ambiguous=True,  # ✅ Pronoun "it" detected
    rewritten_query="What about FastAPI's async support?",  # ✅ Resolved via light context
    referenced_messages=light_context,
    context_usage=ContextUsage(  # Still no need for memory (recent context enough)
        use_user_profile=False,
        use_key_facts=False,
        use_decisions=False,
        use_open_questions=False,
        use_todos=False
    )
)
```

### Step 2: Augmentation
```python
# summary still None → memory_context = ""
augmented_context = AugmentedContext(
    recent_messages=session_state.raw_messages[-5:],  # Last 5 for context
    memory_fields_used=[],
    memory_context="",
    final_augmented_context="Recent: [4 previous messages about FastAPI]"
)
```

### Step 3: Clarification
```python
clarification_result = ClarificationResult(
    needs_clarification=False,  # Resolved via rewrite + recent messages
    clarifying_questions=[]
)
```

### Answer
```python
answer = Answer(
    answer="FastAPI's async support allows you to handle thousands of concurrent requests efficiently using async/await syntax..."
)
session_state.clarification_count = 0  # Reset
```

---

## 🔴 TURN 10: Câu hỏi mơ hồ KHÔNG trả lời được → Cần hỏi lại

**User Input**: "How do I set it up?"

```python
# Current state:
session_state.raw_messages = [
    # Turn 1-9: Discussions about FastAPI, PostgreSQL, Docker, deployment...
]
# Tokens: ~2000 (still < 10k)
# summary: None
```

### Step 1: Rewrite
```python
light_context = session_state.raw_messages[-3:]  # Last 3: about deployment

rewrite_result = RewriteResult(
    original_query="How do I set it up?",
    is_ambiguous=True,  # "it" is ambiguous
    rewritten_query="How do I set up [deployment/database/FastAPI project]?",  # ❌ Still unclear!
    referenced_messages=light_context,
    context_usage=ContextUsage(
        use_key_facts=True,  # Need broader context to understand "it"
        use_decisions=True,
        use_user_profile=False,
        use_open_questions=False,
        use_todos=False
    )
)
```

### Step 2: Augmentation
```python
# summary=None → memory_context = "" (no help from memory)
augmented_context = AugmentedContext(
    recent_messages=session_state.raw_messages[-5:],
    memory_fields_used=[],  # Wanted key_facts but summary=None
    memory_context="",
    final_augmented_context="Recent: [discussing FastAPI, PostgreSQL, Docker deployment]"
)
```

### Step 3: Clarification
```python
# Even with augmentation, "it" is ambiguous (could be FastAPI project, PostgreSQL, Docker, deployment pipeline)

clarification_result = ClarificationResult(
    needs_clarification=True,  # ✅ Cannot resolve
    clarifying_questions=[
        "What specifically do you want to set up?",
        "1) FastAPI project structure?",
        "2) PostgreSQL database?", 
        "3) Docker containerization?",
        "4) Deployment pipeline?"
    ]
)
```

### Return Clarification (NO answer yet)
```python
# Increment clarification counter
session_state.clarification_count += 1  # = 1 (Edge Case 4)

# Return questions to user (not Answer)
return clarification_result.clarifying_questions
```

---

## 🟢 TURN 11: User clarifies

**User Input**: "I mean the Docker setup"

### Step 1: Rewrite
```python
rewrite_result = RewriteResult(
    original_query="I mean the Docker setup",
    is_ambiguous=False,  # Now clear
    rewritten_query=None,
    referenced_messages=[],
    context_usage=ContextUsage(
        use_decisions=True,  # May need past decisions about Docker
        use_key_facts=True,
        use_user_profile=False,
        use_open_questions=False,
        use_todos=False
    )
)
```

### Step 2-3: Augmentation + No clarification needed

### Answer
```python
answer = Answer(
    answer="For Docker setup, here's the Dockerfile for your FastAPI app..."
)

# ✅ Reset clarification counter (successful answer after 1 round)
session_state.clarification_count = 0
```

---

## 🟠 TURN 20: Context dài → Trigger Summarization

**User Input**: "What's the best way to handle authentication?"

```python
# Current state:
session_state.raw_messages = [
    # Turn 1-19: 40 messages total (user + assistant)
]
tokens = count_tokens(session_state.raw_messages)  # = 10,500 tokens
# 10,500 > 10,000 → ✅ TRIGGER SUMMARIZATION!
```

### Step 0: Summarization (Before Pipeline)
```python
# Call Summarizer module
summarizer = Summarizer()
summarization_result = summarizer.summarize(
    messages=session_state.raw_messages[:-1]  # All except current query
)

new_summary = SessionSummary(
    user_profile=UserProfile(
        prefs=["Detailed technical explanations", "Code examples"],
        constraints=["Team of 3 developers", "3-month timeline"],
        background="Software engineer building first production API"
    ),
    current_goal="Build production-ready REST API for e-commerce with FastAPI",
    topics=["FastAPI", "PostgreSQL", "Docker", "Authentication", "Deployment"],
    key_facts=[
        "Using FastAPI with async support",
        "PostgreSQL database for 100k expected users",
        "Docker containerization decided",
        "Planning AWS deployment",
        "Need JWT-based authentication"
    ],
    decisions=[
        "Use FastAPI (not Django)",
        "PostgreSQL as database",
        "Docker for containerization",
        "AWS for hosting"
    ],
    open_questions=[
        "How to handle authentication?",  # Current question
        "Best practices for error handling?"
    ],
    todos=[
        "Set up Docker compose file",
        "Design database schema",
        "Implement auth middleware"
    ]
)

# Check summary tokens
summary_tokens = count_tokens(new_summary)  # = 800 tokens
# 800 < 2000 → No compression needed

summarization_result = SummarizationResult(
    session_summary=new_summary,
    message_range_summarized={"from": 0, "to": 38},  # Summarized first 38 messages
    token_count_before=10500,
    token_count_after=800,
    was_compressed=False  # No compression this time
)

# ✅ Edge Case 1: Keep N recent messages
session_state.raw_messages = session_state.raw_messages[-10:]  # Keep last 10
session_state.summary = new_summary

# New token count: 10 messages (~500 tokens) + summary (800) = 1300 total memory
```

### Step 1: Rewrite (with summary available now!)
```python
rewrite_result = RewriteResult(
    original_query="What's the best way to handle authentication?",
    is_ambiguous=False,
    rewritten_query=None,
    referenced_messages=[],
    context_usage=ContextUsage(
        use_user_profile=True,  # Need to know user's skill level
        use_key_facts=True,  # Know we're using FastAPI
        use_decisions=True,  # Know tech stack decisions
        use_open_questions=False,
        use_todos=False
    )
)
```

### Step 2: Augmentation (NOW with memory!)
```python
# ✅ Edge Case 2: summary is NOT None now
memory_fields = []
memory_context_parts = []

if context_usage.use_user_profile and session_state.summary:
    memory_fields.append("user_profile")
    memory_context_parts.append(f"User Profile: {session_state.summary.user_profile}")

if context_usage.use_key_facts and session_state.summary:
    memory_fields.append("key_facts")
    memory_context_parts.append(f"Key Facts: {session_state.summary.key_facts}")

if context_usage.use_decisions and session_state.summary:
    memory_fields.append("decisions")
    memory_context_parts.append(f"Decisions: {session_state.summary.decisions}")

augmented_context = AugmentedContext(
    recent_messages=session_state.raw_messages[-5:],  # Last 5 from the 10 kept
    memory_fields_used=["user_profile", "key_facts", "decisions"],
    memory_context="\n".join(memory_context_parts),
    final_augmented_context="""
    User Profile: Software engineer, prefers detailed explanations
    Key Facts: Using FastAPI async, PostgreSQL, Docker, AWS deployment
    Decisions: FastAPI framework, PostgreSQL DB, Docker containers
    Recent Messages: [Last 5 messages about Docker and deployment]
    """
)
```

### Step 3: No clarification needed → Answer
```python
answer = Answer(
    answer="""For authentication in FastAPI, I recommend JWT tokens with the following setup:
    
    1. Use python-jose for JWT handling
    2. Implement OAuth2 password flow
    3. Store hashed passwords with passlib
    
    Here's a complete example...
    [detailed code with context from user's FastAPI + PostgreSQL stack]
    """
)
```

---

## 🔥 TURN 50: Summary cần Compression

**User Input**: "Can you summarize our entire discussion?"

```python
# Current state:
session_state.raw_messages = [10 recent messages, ~500 tokens]
session_state.summary = {
    topics: [20 topics],  # Accumulated over 50 turns
    key_facts: [35 facts],
    decisions: [18 decisions],
    open_questions: [12 questions],
    todos: [15 todos]
}

summary_tokens = count_tokens(session_state.summary)  # = 2300 tokens
# 2300 > 2000 → ✅ COMPRESSION NEEDED!
```

### Before Pipeline: Update Summary
```python
# Extract new info from recent 10 messages
new_info = summarizer.extract_info(session_state.raw_messages)

# Merge with old summary (simple append)
merged_summary = SessionSummary(
    user_profile=new_info.user_profile,  # Update
    current_goal=new_info.current_goal,  # Update
    topics=session_state.summary.topics + new_info.topics,  # 20 + 5 = 25 topics
    key_facts=session_state.summary.key_facts + new_info.key_facts,  # 35 + 8 = 43 facts!
    decisions=session_state.summary.decisions + new_info.decisions,  # 18 + 3 = 21
    open_questions=new_info.open_questions,  # Replace (always fresh)
    todos=new_info.todos  # Replace (always fresh)
)

# Check tokens
merged_tokens = count_tokens(merged_summary)  # = 2800 tokens
# 2800 > 2000 → COMPRESS!

# Call LLM to compress
compressed_summary = summarizer.compress_summary(merged_summary)
# LLM receives:
# - 25 topics → output 10 most relevant
# - 43 key_facts → output 15 most important
# - 21 decisions → output 10 key decisions
# - Keep open_questions/todos as-is (already fresh)

compressed_summary = SessionSummary(
    user_profile=merged_summary.user_profile,  # Keep
    current_goal=merged_summary.current_goal,  # Keep
    topics=[
        "FastAPI Development",
        "PostgreSQL Database Design",
        "Docker Containerization",
        "AWS Deployment",
        "JWT Authentication",
        # ... 5 more condensed topics
    ],  # 10 topics (merged/condensed from 25)
    key_facts=[
        "Building production e-commerce API with FastAPI + PostgreSQL",
        "Team: 3 developers, 3-month timeline",
        "Expected scale: 100k users",
        "Using async/await for performance",
        "JWT-based authentication implemented",
        # ... 10 more condensed facts
    ],  # 15 facts (condensed from 43)
    decisions=[
        "Tech stack: FastAPI + PostgreSQL + Docker + AWS",
        "Authentication: JWT with OAuth2 password flow",
        "Database: PostgreSQL with SQLAlchemy ORM",
        # ... 7 more key decisions
    ],  # 10 decisions (condensed from 21)
    open_questions=merged_summary.open_questions,  # Keep as-is
    todos=merged_summary.todos  # Keep as-is
)

compressed_tokens = count_tokens(compressed_summary)  # = 1400 tokens

# ✅ Edge Case 3: Double-check compression result
if compressed_tokens > 2000:
    logger.warning("Summary still large after compression!")
    # Could compress again more aggressively, but 1400 < 2000 so OK

summarization_result = SummarizationResult(
    session_summary=compressed_summary,
    message_range_summarized={"from": 40, "to": 50},
    token_count_before=2800,
    token_count_after=1400,
    was_compressed=True  # ✅ Track compression event
)

session_state.summary = compressed_summary
```

### Continue with Pipeline (using compressed summary)
```python
# Step 1-3 proceed normally with compressed_summary
# Augmentation now uses condensed but complete context
```

---

## 🔴 TURN 60: Clarification Loop Edge Case

**User Input**: "help"  (extremely vague)

```python
session_state.clarification_count = 0  # Reset from previous success
```

### Pipeline detects ambiguity → Ask clarification
```python
clarification_result = ClarificationResult(
    needs_clarification=True,
    clarifying_questions=["What do you need help with?"]
)

session_state.clarification_count += 1  # = 1
return clarifying_questions
```

**TURN 61 User**: "anything"  (still vague!)

### Pipeline again detects ambiguity
```python
clarification_result = ClarificationResult(
    needs_clarification=True,
    clarifying_questions=["Can you be more specific? Are you asking about code, deployment, or architecture?"]
)

session_state.clarification_count += 1  # = 2
# ✅ Edge Case 4: Check limit
if session_state.clarification_count >= 2:  # Max reached!
    # Give up, return best-effort answer
    return Answer(
        answer="I need more specific details to help effectively. Please describe what you're trying to achieve or what problem you're facing."
    )
    session_state.clarification_count = 0  # Reset
```

---

## Summary of Edge Cases Handled:

✅ **Edge Case 1**: Keep 10 messages after summarize (Turn 20)  
✅ **Edge Case 2**: Handle summary=None in augmentation (Turn 1-19)  
✅ **Edge Case 3**: Check compression result (Turn 50)  
✅ **Edge Case 4**: Limit clarification loops to 2 rounds (Turn 60-61)

## Token Flow Timeline:

```
Turn 1-19:  raw_messages grows: 0 → 10,500 tokens, summary=None
Turn 20:    SUMMARIZE → raw_messages: 500 (10 kept), summary: 800 tokens
Turn 21-49: raw_messages grows: 500 → 10,200, summary: 800 → 2300 tokens
Turn 50:    SUMMARIZE + COMPRESS → raw_messages: 500 (10 kept), summary: 1400 tokens
Turn 51+:   Continue cycle...
```

## All Scenarios Covered:

| Scenario | Turn | Result |
|----------|------|--------|
| ✅ Straightforward query | 1 | Direct answer |
| ✅ Ambiguous but resolvable (light context) | 5 | Rewrite → answer |
| ✅ Ambiguous unresolvable → clarify | 10 | Ask clarification |
| ✅ User clarifies | 11 | Answer after clarification |
| ✅ Normal context (no summary) | 1-19 | Use recent messages only |
| ✅ Trigger summarization | 20 | Create summary, keep 10 messages |
| ✅ Use summary in augmentation | 20+ | Pull memory fields |
| ✅ Trigger compression | 50 | Compress summary (List→List smaller) |
| ✅ Clarification loop limit | 60-61 | Max 2 rounds, then give up |

Schemas hoạt động hoàn hảo! 🎉
