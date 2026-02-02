# ChatTMT - Conversational Agent with Session Memory

An intelligent conversational agent that implements session memory management through structured summarization, enabling long conversations without context loss.

---

## Table of Contents

- [Setup Instructions](#setup-instructions)
- [How to Run the Demo](#how-to-run-the-demo)
- [High-Level Design](#high-level-design)
- [Assumptions & Limitations](#assumptions--limitations)

---

## Setup Instructions

### 1. Clone and Create Virtual Environment

**Option A: Using Python venv**
```bash
cd ChatTMT
python -m venv venv

# Windows
.\venv\Scripts\Activate.ps1

# Linux/macOS
source venv/bin/activate
```

**Option B: Using Conda**
```bash
cd ChatTMT
conda create -n chattmt python=3.11
conda activate chattmt
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Copy the example environment file and edit it:

```bash
# Windows
copy .env.example .env

# Linux/macOS
cp .env.example .env
```

Open `.env` and configure your settings. The file is fully commented with explanations for each variable.

> **Important**: Set your `OPENAI_API_KEY` in the `.env` file before running.

### 4. Verify Installation

```bash
# Run unit tests
pytest tests/ -v --ignore=tests/test_e2e.py

# Run all tests including end to end
pytest tests/ -v
```

---

## How to Run the Demo

### Streamlit UI (Recommended)

```bash
streamlit run app/ui/ui_app.py
```

**Features:**
- Interactive chat interface
- Session management (create, save, load sessions)
- **Debug Panel** showing:
  - Session state and metadata
  - Summary schema (structured JSON)
  - Token usage with progress bars
  - Threshold monitoring (history & summary)

### CLI Interface

```bash
python main.py
```

**Commands:**
- Type your questions naturally
- `/exit` or `Ctrl+C` - Save session and exit

---

## High-Level Design

### Architecture Overview

```
User Query → QueryPipeline → Response
                  │
    ┌─────────────┼─────────────┐
    ▼             ▼             ▼
[Step 0]      [Step 1-3]    [Step 4]
 Memory        Query        Generate
 Check       Understanding   Answer
```

### Core Pipeline (4 Steps)

| Step | Module | Purpose |
|------|--------|---------|
| **0** | Summarizer | Check token thresholds → Summarize/Compress if needed |
| **1** | Rewriter | Resolve pronouns ("it", "that") and detect ambiguity |
| **2** | Augmenter | Inject relevant memory fields into context |
| **3** | Clarifier | Ask clarifying questions if query still unclear |
| **4** | Answer | Generate final response using augmented context |

### Key Features

#### 1. Two-Level Memory Compression

| Trigger | Action | Effect |
|---------|--------|--------|
| `raw_messages` exceeds threshold | **Summarization** | Messages → Structured Summary |
| `summary` exceeds threshold | **Compression** | Summary → Smaller Summary |

This ensures conversations can continue indefinitely without hitting context limits. Thresholds are configurable in `.env`.

#### 2. Structured Session Summary (Pydantic Schema)

The summary is **not just a text blob** - it's a structured JSON schema that allows selective retrieval of relevant context.

```json
{
  "user_profile": {
    "prefs": [],           // User preferences (e.g., "detailed explanations")
    "constraints": [],     // User limitations (e.g., "beginner level")
    "background": ""       // User context (e.g., "software engineer")
  },
  "current_goal": "",      // Main objective in this session
  "topics": [],            // Key entities/subjects discussed
  "key_facts": [],         // Important information extracted
  "decisions": [],         // Decisions made during session
  "open_questions": [],    // Unresolved questions
  "todos": []              // Action items to follow up
}
```

#### 3. Intelligent Context Augmentation

The Rewriter module decides which memory fields are relevant for the current query. This **selective augmentation** prevents context pollution and keeps prompts focused.

```json
{
  "use_user_profile": false,    // Include user preferences?
  "use_current_goal": false,    // Include main objective?
  "use_topics": false,          // Include discussed topics?
  "use_key_facts": false,       // Include important facts?
  "use_decisions": false,       // Include past decisions?
  "use_open_questions": false,  // Include pending questions?
  "use_todos": false            // Include action items?
}
```

#### 4. Query Understanding Pipeline

```
Original Query: "What about the other one?"
                     ↓
[Rewriter] - Uses last N messages (light context)
           - Resolves: "other one" → "React framework"
                     ↓
Rewritten: "What about the React framework?"
                     ↓
[Augmenter] - Injects: topics=["Vue", "React"], key_facts=[...]
                     ↓
[Clarifier] - Query clear? → Generate answer
            - Still vague? → Ask clarifying questions
```

#### 5. Clarification Loop with Safety Limit

- System asks up to **2 clarifying questions** (configurable)
- After max rounds, forces a best-effort answer
- Prevents infinite clarification loops

#### 6. Session Persistence

- Sessions saved as JSON files in `data/sessions/`
- Full conversation history with timestamps
- Can resume sessions from UI or programmatically

### Project Structure

```
ChatTMT/
├── app/
│   ├── core/           # Core logic
│   │   ├── pipeline.py    # Orchestrates the 4-step pipeline
│   │   ├── session.py     # Session state management
│   │   └── schemas.py     # Pydantic models (all schemas)
│   ├── modules/        # Pipeline modules
│   │   ├── rewriter.py    # Step 1: Query rewriting
│   │   ├── augmenter.py   # Step 2: Context augmentation
│   │   ├── clarifier.py   # Step 3: Clarification check
│   │   ├── answer.py      # Step 4: Response generation
│   │   └── summarizer.py  # Step 0: Memory compression
│   ├── llms/           # LLM abstraction
│   │   ├── base.py        # Base interface
│   │   └── openai_client.py
│   ├── ui/             # User interfaces
│   │   └── ui_app.py      # Streamlit application
│   └── utils/          # Utilities
│       ├── config.py      # Environment configuration
│       ├── logger.py      # Logging setup
│       └── tokenizer.py   # Token counting (tiktoken)
├── data/sessions/      # Saved session files
├── logs/               # Application logs
├── tests/              # Test suite (123 tests)
├── .env.example        # Environment template (copy to .env)
├── requirements.txt    # Dependencies
└── main.py             # CLI entry point
```

---

## Assumptions & Limitations

### Assumptions

1. **OpenAI API Access**: Requires a valid OpenAI API key with access to the configured model
2. **Model Compatibility**: Designed for GPT class models
3. **Single User**: Each session assumes a single user; no multi-user support
4. **Text Only**: Handles text conversations only; no image/audio support

### Limitations

1. **No RAG/External Knowledge**: Does not retrieve external documents; relies only on conversation context and LLM knowledge
2. **No Streaming**: Responses are returned in full (no token-by-token streaming)
3. **Session Isolation**: Sessions are independent; no cross-session memory
4. **Token Estimation**: Uses tiktoken for count tokens; may vary slightly for other models
5. **Summarization Quality**: Summary quality depends on LLM; very long or complex conversations may lose nuance

---