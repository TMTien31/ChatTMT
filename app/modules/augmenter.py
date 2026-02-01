from typing import List, Optional
from app.core.schemas import Message, SessionSummary, ContextUsage, AugmentedContext
from app.utils.logger import get_logger
from app.utils.config import get_config

logger = get_logger(__name__)

def augment_context(
    recent_messages: List[Message],
    context_usage: ContextUsage,
    summary: Optional[SessionSummary] = None
) -> AugmentedContext:
    """
    Build augmented context by combining recent messages with selected memory fields.
    
    Args:
        recent_messages: Recent conversation messages (typically last N turns)
        context_usage: Flags indicating which memory fields are needed
        summary: Session summary containing memory fields
    
    Returns:
        AugmentedContext with recent messages and selected memory context
    """
    config = get_config()
    
    # Take last RECENT_CONTEXT_SIZE messages
    # Note: 1 turn = 2 messages (user + assistant)
    # Default 10 messages = last 5 turns
    recent = recent_messages[-config.RECENT_CONTEXT_SIZE:] if recent_messages else []
    
    logger.info(f"Augmenting context: {len(recent)} recent messages, "
                f"summary={'present' if summary else 'absent'}")
    
    # Build memory context string based on flags
    memory_fields_used = []
    memory_context_parts = []
    
    if summary:
        if context_usage.use_user_profile and summary.user_profile is not None:
            memory_fields_used.append("user_profile")
            parts = []
            if hasattr(summary.user_profile, 'prefs') and summary.user_profile.prefs:
                parts.append(f"Preferences: {', '.join(summary.user_profile.prefs)}")
            if hasattr(summary.user_profile, 'constraints') and summary.user_profile.constraints:
                parts.append(f"Constraints: {', '.join(summary.user_profile.constraints)}")
            if hasattr(summary.user_profile, 'background') and summary.user_profile.background:
                parts.append(f"Background: {summary.user_profile.background}")
            if parts:
                memory_context_parts.append("USER PROFILE:\n" + "\n".join(parts))
        
        # Current goal
        if context_usage.use_current_goal and summary.current_goal:
            memory_fields_used.append("current_goal")
            memory_context_parts.append(f"CURRENT GOAL: {summary.current_goal}")
        
        # Topics discussed
        if context_usage.use_topics and summary.topics:
            memory_fields_used.append("topics")
            memory_context_parts.append(f"TOPICS DISCUSSED: {', '.join(summary.topics)}")
        
        # Key facts
        if context_usage.use_key_facts and summary.key_facts:
            memory_fields_used.append("key_facts")
            facts_text = "\n".join([f"- {fact}" for fact in summary.key_facts])
            memory_context_parts.append(f"KEY FACTS:\n{facts_text}")
        
        # Decisions made
        if context_usage.use_decisions and summary.decisions:
            memory_fields_used.append("decisions")
            decisions_text = "\n".join([f"- {decision}" for decision in summary.decisions])
            memory_context_parts.append(f"DECISIONS MADE:\n{decisions_text}")
        
        # Open questions
        if context_usage.use_open_questions and summary.open_questions:
            memory_fields_used.append("open_questions")
            questions_text = "\n".join([f"- {q}" for q in summary.open_questions])
            memory_context_parts.append(f"OPEN QUESTIONS:\n{questions_text}")
        
        # Todos
        if context_usage.use_todos and summary.todos:
            memory_fields_used.append("todos")
            todos_text = "\n".join([f"- {todo}" for todo in summary.todos])
            memory_context_parts.append(f"TODOS:\n{todos_text}")
    
    # Combine memory context parts
    memory_context = "\n\n".join(memory_context_parts) if memory_context_parts else ""
    
    logger.info(f"Memory fields used: {memory_fields_used if memory_fields_used else 'none'}")
    logger.debug(f"Memory context length: {len(memory_context)} chars")
    
    # Build final augmented context (recent messages + memory)
    final_parts = []
    
    # Add recent messages
    if recent:
        messages_text = "RECENT CONVERSATION:\n"
        for msg in recent:
            messages_text += f"{msg.role.upper()}: {msg.content}\n"
        final_parts.append(messages_text.strip())
    
    # Add memory context
    if memory_context:
        final_parts.append(f"MEMORY CONTEXT:\n{memory_context}")
    
    final_augmented_context = "\n\n".join(final_parts) if final_parts else ""
    
    return AugmentedContext(
        recent_messages=recent,
        memory_fields_used=memory_fields_used,
        memory_context=memory_context,
        final_augmented_context=final_augmented_context
    )


def format_augmented_context(augmented: AugmentedContext) -> str:
    """
    Format AugmentedContext into a readable text for LLM prompts.
    
    Args:
        augmented: AugmentedContext object
    
    Returns:
        Formatted string with recent messages and memory context
    """
    parts = []
    
    # Recent conversation
    if augmented.recent_messages:
        num_turns = len(augmented.recent_messages) // 2
        parts.append(f"RECENT CONVERSATION (Last {num_turns} turns):")
        for msg in augmented.recent_messages:
            parts.append(f"{msg.role.upper()}: {msg.content}")
    
    # Memory context
    if augmented.memory_context:
        parts.append("")  # Empty line separator
        parts.append("RELEVANT MEMORY:")
        parts.append(augmented.memory_context)
    
    return "\n".join(parts)