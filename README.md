# ChatTMT - Chat Assistant with Session Memory

A modular chat assistant backend that supports **session memory via auto-summarization** and **intelligent query understanding pipeline**.

---

## 📋 Table of Contents
- [Features](#features)
- [High-Level Architecture](#high-level-architecture)
- [Pipeline Design](#pipeline-design)
- [Project Structure](#project-structure)
- [Schemas (Structured Outputs)](#schemas-structured-outputs)
- [Setup Instructions](#setup-instructions)
- [How to Run](#how-to-run)
- [Demo Flows](#demo-flows)
- [Assumptions & Limitations](#assumptions--limitations)

---

## ✨ Features

### A. Session Memory via Summarization
- Automatic summarization when conversation context exceeds configurable threshold (~10k tokens)
- Token counting using tiktoken (or heuristic word/char counting as fallback)
- Structured summary output following defined schema
- Session memory persistence (file-based storage)

### B. Query Understanding Pipeline
- **Step 1 - Rewrite/Paraphrase**: Detect ambiguous queries and generate clarified versions
- **Step 2 - Context Augmentation**: Combine recent messages + session memory
- **Step 3 - Clarifying Questions**: Generate 1-3 questions if intent remains unclear

---

## 🏗️ High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER INPUT                                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CHAT PIPELINE (Orchestrator)                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │  1. Receive user message                                                ││
│  │  2. Add to conversation history                                         ││
│  │  3. Check context size → Trigger summarization if exceeded              ││
│  │  4. Run Query Understanding Pipeline                                    ││
│  │  5. Build final prompt & generate response                              ││
│  └─────────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
            ┌───────────────────────┼───────────────────────┐
            ▼                       ▼                       ▼
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  SESSION MEMORY  │    │ QUERY UNDERSTAND │    │   LLM PROVIDER   │
│  (Summarizer)    │    │    PIPELINE      │    │   (OpenAI/Mock)  │
└──────────────────┘    └──────────────────┘    └──────────────────┘
```

---

## 🔄 Pipeline Design

### Main Flow: Chat Input → Session Memory → Query Understanding → Response

```
┌─────────────┐     ┌─────────────────┐     ┌──────────────────┐     ┌─────────────┐
│  User Input │────▶│  Session Check  │────▶│ Query Understanding│────▶│  Response   │
└─────────────┘     └─────────────────┘     └──────────────────┘     └─────────────┘
                           │                         │
                           ▼                         ▼
                    ┌─────────────┐          ┌──────────────────┐
                    │ Summarizer  │          │ Augmenter        │
                    │ (if needed) │          │ Clarifier        │
                    └─────────────┘          │ Prompt Builder   │
                                             └──────────────────┘
```

### Flow A: Session Memory Trigger

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        SESSION MEMORY FLOW                                │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────┐    ┌──────────────────┐    ┌─────────────────────┐ │
│  │ Conversation    │───▶│ Token Counter    │───▶│ Check Threshold     │ │
│  │ History         │    │ (Tokenizer)      │    │ (e.g., 10k tokens)  │ │
│  └─────────────────┘    └──────────────────┘    └─────────────────────┘ │
│                                                          │               │
│                              ┌────────────────┬──────────┴───────┐      │
│                              ▼                ▼                  │      │
│                    ┌─────────────────┐ ┌─────────────────┐       │      │
│                    │ Under Threshold │ │ Over Threshold  │       │      │
│                    │ (No action)     │ │ (Trigger)       │       │      │
│                    └─────────────────┘ └────────┬────────┘       │      │
│                                                 ▼                │      │
│                                    ┌─────────────────────┐       │      │
│                                    │ Summarizer Module   │       │      │
│                                    │ - Extract key facts │       │      │
│                                    │ - User preferences  │       │      │
│                                    │ - Decisions made    │       │      │
│                                    │ - Open questions    │       │      │
│                                    └──────────┬──────────┘       │      │
│                                               ▼                  │      │
│                                    ┌─────────────────────┐       │      │
│                                    │ Session Storage     │       │      │
│                                    │ (JSON file)         │       │      │
│                                    └─────────────────────┘       │      │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### Flow B: Query Understanding Pipeline

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    QUERY UNDERSTANDING PIPELINE                           │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │ STEP 1: AMBIGUITY DETECTION & REWRITE (Decision Module)             ││
│  │ ┌─────────────┐    ┌──────────────────┐    ┌──────────────────────┐ ││
│  │ │ User Query  │───▶│ Analyze Query    │───▶│ is_ambiguous: bool   │ ││
│  │ └─────────────┘    │ - Missing refs   │    │ rewritten_query: str │ ││
│  │                    │ - Vague terms    │    └──────────────────────┘ ││
│  │                    │ - Pronouns       │                             ││
│  │                    └──────────────────┘                             ││
│  └─────────────────────────────────────────────────────────────────────┘│
│                                    │                                     │
│                                    ▼                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │ STEP 2: CONTEXT AUGMENTATION (Augmenter Module)                     ││
│  │ ┌──────────────────┐    ┌───────────────────────────────────────┐   ││
│  │ │ Recent N messages│───▶│         AUGMENTED CONTEXT             │   ││
│  │ └──────────────────┘    │                                       │   ││
│  │ ┌──────────────────┐    │  - Rewritten query                    │   ││
│  │ │ Session Memory   │───▶│  - Relevant conversation history      │   ││
│  │ │ (Summary)        │    │  - User preferences from memory       │   ││
│  │ └──────────────────┘    │  - Key facts & decisions              │   ││
│  │                         └───────────────────────────────────────┘   ││
│  └─────────────────────────────────────────────────────────────────────┘│
│                                    │                                     │
│                                    ▼                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │ STEP 3: CLARIFYING QUESTIONS (Clarifier Module)                     ││
│  │ ┌─────────────────────┐    ┌──────────────────────────────────────┐ ││
│  │ │ Still Unclear?      │───▶│ Generate 1-3 clarifying questions   │ ││
│  │ │ - Check if rewrite  │    │ "Did you mean X or Y?"              │ ││
│  │ │   resolved ambiguity│    │ "Which project are you referring?"  │ ││
│  │ └─────────────────────┘    └──────────────────────────────────────┘ ││
│  └─────────────────────────────────────────────────────────────────────┘│
│                                    │                                     │
│                                    ▼                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │ FINAL: PROMPT CONSTRUCTION (Prompt Builder Module)                  ││
│  │ ┌─────────────────────────────────────────────────────────────────┐ ││
│  │ │ Combine: System prompt + Augmented context + Final query        │ ││
│  │ └─────────────────────────────────────────────────────────────────┘ ││
│  └─────────────────────────────────────────────────────────────────────┘│
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
ChatTMT/
├── README.md                    # This documentation
├── requirements.txt             # Python dependencies
├── main.py                      # Entry point (CLI demo)
├── app/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── pipeline.py          # Main orchestrator - coordinates all modules
│   │   ├── schemas.py           # Pydantic models for structured outputs
│   │   ├── session.py           # Session management & storage
│   │   └── tokenizer.py         # Token counting utilities
│   │
│   ├── llms/
│   │   ├── __init__.py
│   │   ├── base.py              # Abstract LLM interface
│   │   ├── openai_client.py     # OpenAI API implementation
│   │   └── mock_client.py       # Mock LLM for testing
│   │
│   ├── modules/
│   │   ├── __init__.py
│   │   ├── summarizer.py        # Session summarization logic
│   │   ├── decision.py          # Query ambiguity detection & rewriting
│   │   ├── augmenter.py         # Context augmentation
│   │   ├── clarifier.py         # Clarifying questions generation
│   │   ├── prompt_builder.py    # Final prompt assembly
│   │   └── answer.py            # Response generation
│   │
│   ├── ui/
│   │   ├── __init__.py
│   │   └── ui_app.py            # Streamlit/Gradio UI (optional)
│   │
│   └── utils/
│       ├── __init__.py
│       ├── config.py            # Configuration settings
│       └── logger.py            # Logging utilities
│
├── tests/
│   ├── __init__.py
│   ├── test_pipeline.py         # Pipeline unit tests
│   ├── test_summarizer.py       # Summarizer tests
│   └── test_query_understanding.py
│
└── data/
    ├── test_conversations/      # Test conversation logs
    │   ├── long_conversation.jsonl     # Demo: Session memory trigger
    │   ├── ambiguous_queries.jsonl     # Demo: Ambiguous query handling
    │   └── mixed_scenario.jsonl        # Demo: Combined scenarios
    └── sessions/                # Session storage (generated)
        └── .gitkeep
```

---

## 📐 Schemas (Structured Outputs)

### 1. Session Summary Schema

```python
class UserProfile(BaseModel):
    preferences: List[str] = []      # User preferences discovered
    constraints: List[str] = []      # User constraints/limitations

class SessionSummary(BaseModel):
    user_profile: UserProfile
    key_facts: List[str] = []        # Important facts from conversation
    decisions: List[str] = []        # Decisions made during session
    open_questions: List[str] = []   # Unresolved questions
    todos: List[str] = []            # Action items identified

class SummarizationResult(BaseModel):
    session_summary: SessionSummary
    message_range_summarized: Dict[str, int]  # {"from": 0, "to": 42}
    token_count_before: int
    token_count_after: int
    timestamp: datetime
```

### 2. Query Understanding Schema

```python
class QueryUnderstandingResult(BaseModel):
    original_query: str
    is_ambiguous: bool
    ambiguity_reasons: List[str] = []        # Why it's ambiguous
    rewritten_query: Optional[str] = None    # Clarified version
    needed_context_from_memory: List[str] = []  # Memory fields used
    clarifying_questions: List[str] = []     # 1-3 questions if unclear
    final_augmented_context: str             # Combined context
    confidence_score: float                  # 0.0 - 1.0
```

### 3. Message Schema

```python
class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: Optional[datetime] = None
    metadata: Optional[Dict] = None
```

---

## 🚀 Setup Instructions

### Prerequisites
- Python 3.10+
- OpenAI API key (or use mock client for testing)

### Installation

```bash
# Clone repository
git clone <repository-url>
cd ChatTMT

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

---

## 🎮 How to Run

### CLI Demo (Main)

```bash
# Interactive chat mode
python main.py

# Demo Flow 1: Session Memory Trigger
python main.py --demo session-memory --input data/test_conversations/long_conversation.jsonl

# Demo Flow 2: Ambiguous Query Handling  
python main.py --demo query-understanding --input data/test_conversations/ambiguous_queries.jsonl

# Use mock LLM (no API key needed)
python main.py --mock
```

### Streamlit UI (Optional)

```bash
streamlit run app/ui/ui_app.py
```

---

## 🎯 Demo Flows

### Flow 1: Session Memory Trigger
1. Load `long_conversation.jsonl` (pre-built conversation with ~15k tokens)
2. Watch context size increase as messages are processed
3. When threshold (10k tokens) is exceeded:
   - Summarizer automatically triggers
   - Generates structured summary
   - Stores in session memory
   - Trims conversation history
4. Logs show summary output

### Flow 2: Ambiguous Query Handling
1. Load test query: "What about that thing we discussed?"
2. System detects ambiguity (pronouns, vague references)
3. Attempts rewrite using session memory context
4. If still unclear, generates clarifying questions
5. Shows full QueryUnderstandingResult output

---

## ⚠️ Assumptions & Limitations

### Assumptions
1. Single-user session model (no multi-user support)
2. English language only for query understanding
3. OpenAI GPT-4o-mini as default LLM (configurable)
4. File-based session storage (suitable for demo)

### Limitations
1. No long-term memory persistence across sessions
2. Summarization may lose nuanced context
3. Ambiguity detection relies on LLM capabilities
4. No RAG/vector search integration (out of scope)

### Configuration Defaults
- Token threshold: 10,000 tokens
- Recent messages for context: 5
- Max clarifying questions: 3
- Session storage: `data/sessions/`

---

## 🧪 Test Data

Three conversation logs are provided in `data/test_conversations/`:

| File | Purpose | Description |
|------|---------|-------------|
| `long_conversation.jsonl` | Session Memory | 50+ messages exceeding token limit |
| `ambiguous_queries.jsonl` | Query Understanding | Contains vague/ambiguous queries |
| `mixed_scenario.jsonl` | Combined Demo | Real-world conversation simulation |

---

## 📝 License

MIT License - See [LICENSE](LICENSE) for details.