import json
from typing import List, Optional
from app.core.schemas import Message, LLMMessage, SessionSummary, UserProfile
from app.llms.base import BaseLLM
from app.utils.logger import get_logger

logger = get_logger(__name__)

def summarize_messages(
    messages: List[Message],
    llm_client: BaseLLM,
    existing_summary: Optional[SessionSummary] = None
) -> SessionSummary:
    """Summarize messages into SessionSummary using LLM."""
    logger.info(f"Summarizing {len(messages)} messages")
    
    # Build prompt for LLM
    prompt = _build_summarization_prompt(messages, existing_summary)
    
    # Create messages for LLM API (use LLMMessage for system role)
    llm_messages = [
        LLMMessage(role="system", content=prompt),
        LLMMessage(role="user", content="Please summarize the conversation above in JSON format.")
    ]
    
    # Call LLM
    logger.debug("Calling LLM for summarization")
    response = llm_client.chat(llm_messages, temperature=0.2, max_tokens=2000)
    
    # Parse response
    try:
        summary_dict = json.loads(response)
        logger.debug("Successfully parsed LLM response")
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM response: {e}")
        # Fallback to empty summary
        summary_dict = {}
    
    # Convert to SessionSummary
    summary = _dict_to_session_summary(summary_dict)
    
    logger.info("Summarization complete")
    return summary


def compress_summary(
    old_summary: SessionSummary,
    new_messages: List[Message],
    llm_client: BaseLLM
) -> SessionSummary:
    """Compress old summary + new messages into updated summary."""
    logger.info(f"Compressing summary with {len(new_messages)} new messages")
    
    # Build compression prompt
    prompt = _build_compression_prompt(old_summary, new_messages)
    
    # Create messages for LLM API (use LLMMessage for system role)
    llm_messages = [
        LLMMessage(role="system", content=prompt),
        LLMMessage(role="user", content="Please compress and update the summary in JSON format.")
    ]
    
    # Call LLM
    logger.debug("Calling LLM for compression")
    response = llm_client.chat(llm_messages, temperature=0.2, max_tokens=2000)
    
    # Parse response
    try:
        summary_dict = json.loads(response)
        logger.debug("Successfully parsed compression response")
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse compression response: {e}")
        # Return old summary as fallback
        return old_summary
    
    # Convert to SessionSummary
    compressed_summary = _dict_to_session_summary(summary_dict)
    
    logger.info("Compression complete")
    return compressed_summary


def _build_summarization_prompt(messages: List[Message], existing_summary: Optional[SessionSummary]) -> str:
    """Build detailed prompt for LLM to summarize messages."""
    conversation = "\n".join([f"{msg.role.upper()}: {msg.content}" for msg in messages])
    
    existing_context = ""
    if existing_summary:
        existing_context = f"\n\nEXISTING SUMMARY:\n{_summary_to_text(existing_summary)}\n"
    
    prompt = f"""You are an expert at summarizing conversations. 
              Analyze the following conversation and extract structured information.{existing_context}

              CONVERSATION TO SUMMARIZE:
              {conversation}

              Extract and return ONLY a JSON object with these fields:
              {{
                  "user_profile": {{
                      "prefs": ["user preferences list"],
                      "constraints": ["user constraints/limitations"],
                      "background": "user background info or null"
                  }},
                  "current_goal": "What the user is trying to achieve",
                  "topics": ["topic1", "topic2"],
                  "key_facts": ["important fact 1", "important fact 2"],
                  "decisions": ["decision 1", "decision 2"],
                  "open_questions": ["question 1", "question 2"],
                  "todos": ["todo 1", "todo 2"]
              }}

              Return ONLY the JSON, no other text."""
    
    return prompt


def _build_compression_prompt(old_summary: SessionSummary, new_messages: List[Message]) -> str:
    """Build detailed prompt for LLM to compress summary."""
    old_text = _summary_to_text(old_summary)
    new_text = "\n".join([f"{msg.role.upper()}: {msg.content}" for msg in new_messages])
    
    prompt = f"""You are an expert at compressing and updating conversation summaries.

              OLD SUMMARY:
              {old_text}

              NEW MESSAGES:
              {new_text}

              Merge the old summary with new information. Keep only the most important details.
              Remove redundant or less important information to reduce token usage.

              Return ONLY a JSON object with these fields:
              {{
                  "user_profile": {{"prefs": [...], "constraints": [...], "background": "..."}},
                  "current_goal": "...",
                  "topics": [...],
                  "key_facts": [...],
                  "decisions": [...],
                  "open_questions": [...],
                  "todos": [...]
              }}

              Return ONLY the JSON, no other text."""
    
    return prompt


def _summary_to_text(summary: SessionSummary) -> str:
    """Convert SessionSummary to readable text for prompts."""
    lines = []
    
    if summary.user_profile:
        if summary.user_profile.prefs:
            lines.append(f"Preferences: {', '.join(summary.user_profile.prefs)}")
        if summary.user_profile.constraints:
            lines.append(f"Constraints: {', '.join(summary.user_profile.constraints)}")
        if summary.user_profile.background:
            lines.append(f"Background: {summary.user_profile.background}")
    
    if summary.current_goal:
        lines.append(f"Goal: {summary.current_goal}")
    
    if summary.topics:
        lines.append(f"Topics: {', '.join(summary.topics)}")
    
    if summary.key_facts:
        lines.append("Key Facts:")
        for fact in summary.key_facts:
            lines.append(f"  - {fact}")
    
    if summary.decisions:
        lines.append("Decisions:")
        for decision in summary.decisions:
            lines.append(f"  - {decision}")
    
    if summary.open_questions:
        lines.append("Open Questions:")
        for question in summary.open_questions:
            lines.append(f"  - {question}")
    
    if summary.todos:
        lines.append("TODOs:")
        for todo in summary.todos:
            lines.append(f"  - {todo}")
    
    return "\n".join(lines)


def _dict_to_session_summary(data: dict) -> SessionSummary:
    """Convert dict from LLM response to SessionSummary object."""
    
    # Extract user_profile
    user_profile = UserProfile()
    if "user_profile" in data and data["user_profile"]:
        profile_data = data["user_profile"]
        user_profile = UserProfile(
            prefs=profile_data.get("prefs", []),
            constraints=profile_data.get("constraints", []),
            background=profile_data.get("background")
        )
    
    # Create SessionSummary
    return SessionSummary(
        user_profile=user_profile,
        current_goal=data.get("current_goal"),
        topics=data.get("topics", []),
        key_facts=data.get("key_facts", []),
        decisions=data.get("decisions", []),
        open_questions=data.get("open_questions", []),
        todos=data.get("todos", [])
    )