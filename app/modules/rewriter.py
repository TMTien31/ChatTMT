import json
from typing import List, Optional
from app.core.schemas import Message, LLMMessage, RewriteResult, ContextUsage, SessionSummary
from app.llms.base import BaseLLM
from app.utils.logger import get_logger
from app.utils.config import get_config

logger = get_logger(__name__)

def rewrite_query(
    query: str,
    recent_messages: List[Message],
    llm_client: BaseLLM,
    summary: Optional[SessionSummary] = None
) -> RewriteResult:
    """
    Rewrite query using light context.
    
    Detects ambiguity and decides which memory fields are needed.
    """
    logger.info(f"Rewriting query: '{query}' with {len(recent_messages)} recent messages")
    
    # Build prompt
    prompt = _build_rewrite_prompt(query, recent_messages, summary)
    
    # Create LLM messages
    llm_messages = [
        LLMMessage(role="system", content=prompt),
        LLMMessage(role="user", content="Output JSON:")
    ]
    
    # Call LLM
    logger.debug("Calling LLM for query rewriting")
    response = llm_client.chat(llm_messages, temperature=0.2, max_tokens=1000)
    
    # Parse response
    logger.debug(f"LLM raw response: {response[:200]}")
    try:
        data = json.loads(response)
        logger.debug(f"Successfully parsed rewrite response: {data}")
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse rewrite response: {e}")
        # Fallback: assume query is clear, no rewrite needed
        return RewriteResult(
            original_query=query,
            is_ambiguous=False,
            rewritten_query=None,
            referenced_messages=[],
            context_usage=ContextUsage()
        )
    
    # Build RewriteResult
    result = _dict_to_rewrite_result(data, query, recent_messages)
    
    logger.info(f"Rewrite complete: ambiguous={result.is_ambiguous}, "
                f"rewritten={result.rewritten_query is not None}")
    
    return result


def _build_rewrite_prompt(
    query: str, 
    recent_messages: List[Message],
    summary: Optional[SessionSummary]
) -> str:
    """Build prompt for query rewriting with light context."""
    config = get_config()
    
    # Format recent messages (light context)
    # Note: 1 turn = 2 messages (user + assistant)
    # LIGHT_CONTEXT_SIZE = max messages to use (e.g., 8 messages = 4 turns)
    context_text = ""
    if recent_messages:
        light_context = recent_messages[-config.LIGHT_CONTEXT_SIZE:]
        num_turns = len(light_context) // 2
        context_text = f"RECENT CONVERSATION (Last {num_turns} turns, {len(light_context)} messages):\n"
        for msg in light_context:
            context_text += f"{msg.role.upper()}: {msg.content}\n"
    else:
        context_text = "RECENT CONVERSATION: (empty - first message in session)\n"
    
    # Optional: Include summary for additional context
    summary_text = ""
    if summary:
        summary_text = f"\nSESSION SUMMARY (for reference):\n"
        summary_text += f"- Current goal: {summary.current_goal or 'None'}\n"
        summary_text += f"- Topics discussed: {', '.join(summary.topics) if summary.topics else 'None'}\n"
        summary_text += f"- User preferences: {', '.join(summary.user_profile.prefs) if summary.user_profile.prefs else 'None'}\n"
    
    prompt = f"""You are a query understanding assistant. Your task is to:

              1. **Detect if query is AMBIGUOUS** (contains pronouns/references needing resolution)
              2. **Rewrite query** if ambiguous by resolving pronouns using RECENT CONVERSATION
              3. **Determine context needs** from session memory

              {context_text}{summary_text}

              CURRENT USER QUERY: "{query}"

              DECISION RULES:

              **is_ambiguous = TRUE** if query contains:
              - Pronouns referring to previous context: "it", "that", "this", "them", "its"
              - Vague references: "the above", "like before", "similar approach", "the same"
              - Continuation phrases without context: "continue", "keep going", "next step" (when recent conversation is empty)

              **is_ambiguous = FALSE** if query:
              - Is self-contained and clear (e.g., "What is Python?", "How to install FastAPI?")
              - Asks about NEW topic unrelated to recent conversation
              - Contains all necessary information to answer

              **rewritten_query**:
              - If is_ambiguous=true AND can resolve from RECENT CONVERSATION: provide resolved query
              - If is_ambiguous=true BUT cannot resolve (empty context): set to null
              - If is_ambiguous=false: set to null

              **context_usage flags**: Set to TRUE if query needs that information to answer properly:
              - use_current_goal: Query about continuing work, next steps, "where we left off"
              - use_topics: Query needs to know what was discussed before
              - use_todos: Query about tasks, what to do next
              - use_decisions: Query about past choices, "should I use that approach"
              - use_key_facts: Query needs specific facts from history
              - use_user_profile: Query needs user's background/preferences
              - use_open_questions: Query about outstanding questions
                - use_todos: User's todo list needed?

              OUTPUT FORMAT (JSON):
              {{
                "is_ambiguous": true/false,
                "rewritten_query": "Clarified query" or null,
                "context_usage": {{
                  "use_user_profile": true/false,
                  "use_current_goal": true/false,
                  "use_topics": true/false,
                  "use_key_facts": true/false,
                  "use_decisions": true/false,
                  "use_open_questions": true/false,
                  "use_todos": true/false
                }}
              }}

              EXAMPLES:

              Example 1: Pronoun resolution
              Query: "What about it?"
              Recent: "USER: Tell me about FastAPI\\nASSISTANT: FastAPI is a modern Python framework..."
              Analysis: "it" refers to FastAPI
              Output: {{"is_ambiguous": true, "rewritten_query": "What about FastAPI?", "context_usage": {{"use_user_profile": false, "use_current_goal": false, "use_topics": false, "use_key_facts": false, "use_decisions": false, "use_open_questions": false, "use_todos": false}}}}

              Example 2: Continuation query (needs context)
              Query: "Continue where we left off"
              Recent: (empty - first message in session)
              Analysis: Needs goal/topics/todos from memory
              Output: {{"is_ambiguous": true, "rewritten_query": null, "context_usage": {{"use_user_profile": false, "use_current_goal": true, "use_topics": true, "use_key_facts": false, "use_decisions": false, "use_open_questions": false, "use_todos": true}}}}

              Example 3: Clear, self-contained query
              Query: "What is Flask?"
              Recent: "USER: What is Django?\\nASSISTANT: Django is a Python web framework."
              Analysis: New topic, no pronouns, self-contained
              Output: {{"is_ambiguous": false, "rewritten_query": null, "context_usage": {{"use_user_profile": false, "use_current_goal": false, "use_topics": false, "use_key_facts": false, "use_decisions": false, "use_open_questions": false, "use_todos": false}}}}

              Example 4: Reference resolution with decision context
              Query: "Should I use that approach?"
              Recent: "USER: Is Django or FastAPI better?\\nASSISTANT: FastAPI is better for modern APIs..."
              Analysis: "that approach" = FastAPI, also needs decision history
              Output: {{"is_ambiguous": true, "rewritten_query": "Should I use FastAPI?", "context_usage": {{"use_user_profile": false, "use_current_goal": false, "use_topics": false, "use_key_facts": false, "use_decisions": true, "use_open_questions": false, "use_todos": false}}}}

              Now analyze the CURRENT USER QUERY above and output ONLY valid JSON, nothing else."""
    
    return prompt


def _dict_to_rewrite_result(
    data: dict,
    original_query: str,
    recent_messages: List[Message]
) -> RewriteResult:
    """Convert LLM JSON response to RewriteResult object."""
    
    # Parse context_usage
    context_usage_data = data.get("context_usage", {})
    context_usage = ContextUsage(
        use_user_profile=context_usage_data.get("use_user_profile", False),
        use_current_goal=context_usage_data.get("use_current_goal", False),
        use_topics=context_usage_data.get("use_topics", False),
        use_key_facts=context_usage_data.get("use_key_facts", False),
        use_decisions=context_usage_data.get("use_decisions", False),
        use_open_questions=context_usage_data.get("use_open_questions", False),
        use_todos=context_usage_data.get("use_todos", False)
    )
    
    # Determine which messages were actually used (up to LIGHT_CONTEXT_SIZE)
    config = get_config()
    if data.get("is_ambiguous"):
        referenced_messages = recent_messages[-config.LIGHT_CONTEXT_SIZE:]
    else:
        referenced_messages = []
    
    return RewriteResult(
        original_query=original_query,
        is_ambiguous=data.get("is_ambiguous", False),
        rewritten_query=data.get("rewritten_query"),
        referenced_messages=referenced_messages,
        context_usage=context_usage
    )