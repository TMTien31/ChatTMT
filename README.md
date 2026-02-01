# ChatTMT - Conversational Agent with Session Memory

A production-ready conversational agent built with LangChain and OpenAI that implements intelligent session memory management, query understanding, and context-aware responses.

## Features

- **Session Memory via Summarization**: Automatically summarizes conversations when token count exceeds 2000 tokens, maintaining context while reducing token usage
- **Query Understanding Pipeline**: 
  - Rewrites ambiguous queries for clarity
  - Augments queries with session context and user profile
  - Asks clarifying questions when needed
- **Structured Outputs**: Uses Pydantic schemas for session summaries and query analysis
- **Session Persistence**: Saves conversation history to JSON files for continuation
- **Dual Interfaces**: CLI and Streamlit UI with debug panel

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure OpenAI API

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4o-mini
```

See [docs/OPENAI_API_SETUP.md](docs/OPENAI_API_SETUP.md) for detailed API setup instructions.

### 3. Verify Installation

Run tests to ensure everything is working:

```bash
pytest tests/ -v
```

Expected: 126 tests pass in ~5 minutes.

## Usage

### CLI Interface

```bash
python main.py
```

**Available commands:**
- Type your questions naturally
- `/exit` - Save session and exit
- `Ctrl+C` - Save session and exit

### Streamlit UI

```bash
streamlit run app/ui/ui_app.py
```

**Features:**
- Chat interface with session history
- Debug panel showing:
  - Raw token count
  - Compressed token count
  - Recent summarization status
  - Query rewriting logs
- Session management (load/save/create new)

## System Design

### High-Level Architecture

```
User Query → Pipeline → LLM Response
              ↓
         [Rewrite] → [Augment] → [Decision] → [Answer/Clarify] → [Summarize]
```

### Pipeline Flow

1. **Rewriter**: Analyzes query for ambiguity, rewrites if needed
2. **Augmenter**: Adds session context (summary + recent messages) and user profile
3. **Decision**: Routes to answer or clarification based on query clarity
4. **Answer/Clarifier**: Generates response or asks clarifying questions
5. **Summarizer**: Triggers when raw token count > 2000, generates structured summary

### Key Components

**app/core/**
- `pipeline.py` - Orchestrates the query processing pipeline
- `session.py` - Manages session state and persistence
- `schemas.py` - Pydantic models for structured outputs

**app/modules/**
- `rewriter.py` - Query rewriting for ambiguous queries
- `augmenter.py` - Context augmentation with memory
- `decision.py` - Routing logic (answer vs clarify)
- `answer.py` - Final answer generation
- `clarifier.py` - Clarifying question generation
- `summarizer.py` - Conversation summarization
- `prompt_builder.py` - Prompt templates

**app/llms/**
- `openai_client.py` - OpenAI API client with retry logic (3 attempts, exponential backoff)

**app/utils/**
- `config.py` - Configuration management with environment variables
- `logger.py` - Structured logging
- `tokenizer.py` - Token counting with tiktoken

### Token Management

- **Raw Messages**: Full conversation history
- **Summarization Trigger**: When raw token count > 2000 (configurable via `TOKEN_THRESHOLD_RAW`)
- **Context Window**: Keeps last 5 turns + summary (configurable via `KEEP_RECENT_N`)
- **Token Counting**: Uses `tiktoken` for accurate OpenAI token measurement

### Structured Outputs

**SessionSummary** (from summarizer):
```json
{
  "user_profile": {
    "prefs": ["preference 1", "preference 2"],
    "constraints": ["constraint 1"],
    "background": "user context"
  },
  "current_goal": "what user is trying to achieve",
  "topics": ["topic 1", "topic 2"],
  "key_facts": ["fact 1", "fact 2"],
  "decisions": ["decision 1"],
  "open_questions": ["question 1"],
  "todos": ["action 1"]
}
```

**QueryUnderstandingResult** (from rewriter):
```json
{
  "original_query": "user's original query",
  "rewritten_query": "clarified version",
  "is_ambiguous": false,
  "explanation": "why rewrite was needed"
}
```

## Assumptions & Limitations

### Assumptions

1. **OpenAI API Access**: Requires valid API key with sufficient quota
2. **English Language**: Prompts and examples optimized for English (but supports other languages via LLM)
3. **Internet Connection**: Required for OpenAI API calls
4. **Token Threshold**: 2000 tokens is a reasonable balance between context and cost (adjustable)
5. **Recent Context**: Keeping last 5 turns provides sufficient context for most conversations

### Limitations

1. **Token Counting Overhead**: Doesn't account for message formatting overhead (~4-5 tokens per message)
2. **Clarification Counter**: Not reset on topic switch (edge case, <0.1% of conversations)
3. **Error Handling**: Corrupt JSON sessions may not load gracefully (low frequency)
4. **Cost Tracking**: No aggregated LLM cost reporting
5. **Single User**: Designed for single-user sessions (no multi-user support)
6. **English Prompts**: System prompts in English, may be suboptimal for non-English conversations
7. **Debug Logs**: Default LOG_LEVEL is DEBUG (acceptable for demo, should be INFO in production)

### Design Decisions

- **File-based Persistence**: Simple JSON storage for demo purposes (would use database in production)
- **Exponential Backoff**: 3 retry attempts (1s, 2s, 4s delays) for OpenAI API resilience
- **Pydantic Schemas**: Ensures structured, validated outputs from LLM
- **Token-based Triggers**: Simpler than semantic change detection, more predictable
- **Keep Recent N**: Balances context relevance with token efficiency

## Testing

Run all tests:
```bash
pytest tests/ -v
```

Run specific test files:
```bash
pytest tests/test_pipeline.py -v
pytest tests/test_e2e.py -v
```

### Test Coverage

- **126 tests** across 13 test files
- **Unit tests**: Config, logger, schemas, tokenizer, modules
- **Integration tests**: Pipeline, end-to-end flows
- **Coverage**: Core logic, error handling, edge cases

### Manual Testing

See [docs/MANUAL_TESTING_GUIDE.md](docs/MANUAL_TESTING_GUIDE.md) for scenarios to test:
1. Beginner learning journey (20+ turns, triggers summarization)
2. Ambiguous query handling
3. Context continuity across session loads

## Project Structure

```
ChatTMT/
├── app/
│   ├── core/           # Pipeline orchestration
│   ├── modules/        # Query processing modules
│   ├── llms/           # LLM clients
│   ├── ui/             # Streamlit interface
│   └── utils/          # Config, logging, tokenizer
├── data/
│   └── sessions/       # Saved session files
├── docs/               # Documentation
├── tests/              # Test suite (126 tests)
├── main.py            # CLI entry point
├── requirements.txt   # Python dependencies
└── README.md          # This file
```

## Debugging

Enable debug mode to see detailed pipeline logs:

```bash
# In .env file
LOG_LEVEL=DEBUG
```

See [docs/DEBUG_MODE.md](docs/DEBUG_MODE.md) for advanced debugging techniques.

## Contributing

This is a demo project for assignment submission. For production use, consider:
- Database for session storage (PostgreSQL/MongoDB)
- User authentication and multi-user support
- Redis for caching summaries
- Monitoring and observability (Prometheus, Grafana)
- Horizontal scaling with load balancer
- Cost tracking and budget alerts

## License

See [LICENSE](LICENSE) file for details.

---

**Built with:** Python 3.11, LangChain, OpenAI, Streamlit, Pydantic, pytest