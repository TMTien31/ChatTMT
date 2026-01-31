"""
Answer Generation Module

Generates final answers to user queries using augmented context.
"""

import json
from typing import Optional

from app.core.schemas import AugmentedContext, LLMMessage
from app.llms.base import BaseLLM
from app.utils.logger import get_logger
from app.utils.config import get_config

logger = get_logger(__name__)
config = get_config()


def generate_answer(
    query: str,
    augmented_context: AugmentedContext,
    llm: BaseLLM,
    temperature: Optional[float] = None
) -> str:
    """
    Generate answer to user query using augmented context.
    
    Args:
        query: User's query to answer
        augmented_context: Context extracted from memory and recent messages
        llm: LLM client to use
        temperature: Sampling temperature (defaults to config.ANSWER_TEMPERATURE)
        
    Returns:
        Generated answer as string
    """
    if temperature is None:
        temperature = config.ANSWER_TEMPERATURE
        
    logger.info(f"Generating answer for query: '{query}'")
    
    # Build prompt with context
    prompt = _build_answer_prompt(query, augmented_context)
    
    # Prepare messages for LLM
    messages = [
        LLMMessage(role="system", content=prompt),
        LLMMessage(role="user", content=query)
    ]
    
    # Generate answer
    try:
        response = llm.chat(
            messages, 
            temperature=temperature, 
            max_tokens=config.ANSWER_MAX_TOKENS
        )
        answer = response.strip()
        
        logger.info("Answer generated successfully")
        return answer
        
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        raise


def _build_answer_prompt(
    query: str,
    augmented_context: AugmentedContext
) -> str:
    """Build system prompt for answer generation."""
    
    context_text = augmented_context.final_augmented_context
    
    prompt = f"""You are a helpful AI assistant with access to conversation history and user memory.

⚠️ CRITICAL: You MUST read and use the context provided below. DO NOT ask for information that is already in the context.

AVAILABLE CONTEXT:
{context_text}

MANDATORY RULES:
1. ⚠️ **READ THE CONTEXT FIRST**: Before answering, carefully check what information is already provided
2. ⚠️ **USE WHAT YOU HAVE**: If context contains preferences, past discussions, or relevant details → USE THEM
3. ⚠️ **DO NOT RE-ASK**: NEVER ask for information that's already in the context (preferences, topics, decisions, etc.)
4. **Be specific with context**: When context provides details, reference them explicitly
5. **Be conversational**: Respond naturally, as if continuing an ongoing conversation
6. **Be helpful**: Provide actionable information and examples
7. **Remember continuity**: Maintain awareness of ongoing topics and tasks

EXAMPLE CORRECT BEHAVIOR:
- If context says "PREFS: Prefers Python" and user asks "Suggest a language" → Recommend Python
- If context shows "TOPICS: Database setup" and user asks "What next?" → Continue with database topic
- If context has "TODOS: Write tests" and user asks "What to do?" → Reference the test writing task

RESPONSE STYLE:
- Use a friendly, professional tone
- Format with markdown when helpful (lists, code blocks, emphasis)
- Explicitly reference context (e.g., "Based on your preference for Python..." or "As we discussed...")
- If context is truly empty/insufficient, THEN you may ask clarifying questions

Now answer the user's query using the context provided above."""

    return prompt


def generate_contextual_response(
    query: str,
    augmented_context: AugmentedContext,
    llm: BaseLLM,
    include_metadata: bool = False,
    temperature: Optional[float] = None
) -> dict:
    """
    Generate answer with additional metadata.
    
    Args:
        query: User's query
        augmented_context: Context from memory
        llm: LLM client
        include_metadata: If True, return metadata about context usage
        temperature: Sampling temperature (defaults to config.ANSWER_TEMPERATURE)
        
    Returns:
        Dict with 'answer' and optionally 'metadata'
    """
    answer = generate_answer(query, augmented_context, llm, temperature)
    
    result = {"answer": answer}
    
    if include_metadata:
        result["metadata"] = {
            "memory_fields_used": augmented_context.memory_fields_used,
            "recent_message_count": len(augmented_context.recent_messages),
            "has_memory_context": bool(augmented_context.memory_context)
        }
    
    return result