import json
from typing import Optional
from app.core.schemas import LLMMessage, ClarificationResult, AugmentedContext
from app.llms.base import BaseLLM
from app.utils.logger import get_logger
from app.utils.config import get_config

logger = get_logger(__name__)
config = get_config()

def check_clarification_needed(
    original_query: str,
    augmented_context: AugmentedContext,
    llm_client: BaseLLM
) -> ClarificationResult:
    """
    Determine if clarification is needed and generate questions if so.
    
    Args:
        original_query: User's original query
        augmented_context: Context with recent messages + selected memory
        llm_client: LLM client for decision making
    
    Returns:
        ClarificationResult with decision and questions (if needed)
    """
    logger.info(f"Checking clarification for query: '{original_query}'")
    
    # Build prompt
    prompt = _build_clarification_prompt(original_query, augmented_context)
    
    # Create LLM messages
    llm_messages = [
        LLMMessage(role="system", content=prompt),
        LLMMessage(role="user", content="Output JSON:")
    ]
    
    # Call LLM
    logger.debug("Calling LLM for clarification decision")
    response = llm_client.chat(
        llm_messages, 
        temperature=config.CLARIFIER_TEMPERATURE, 
        max_tokens=config.CLARIFIER_MAX_TOKENS
    )
    
    # Parse response
    logger.debug(f"LLM raw response: {response[:200]}")
    try:
        data = json.loads(response)
        logger.debug(f"Successfully parsed clarification response")
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse clarification response: {e}")
        # Fallback: assume can answer (no clarification needed)
        return ClarificationResult(
            needs_clarification=False,
            clarifying_questions=[]
        )
    
    # Build result
    needs_clarification = data.get("needs_clarification", False)
    questions = data.get("clarifying_questions", [])
    
    # Validate: limit to 3 questions
    if len(questions) > 3:
        logger.warning(f"LLM returned {len(questions)} questions, limiting to 3")
        questions = questions[:3]
    
    if needs_clarification and questions:
        logger.info(f"Clarification needed, questions: {questions}")
    else:
        logger.info(f"Clarification not needed")
    
    return ClarificationResult(
        needs_clarification=needs_clarification,
        clarifying_questions=questions
    )


def _build_clarification_prompt(
    query: str,
    augmented_context: AugmentedContext
) -> str:
    """Build prompt for clarification decision."""
    
    # Format augmented context for prompt
    context_text = augmented_context.final_augmented_context
    
    prompt = f"""You are a clarification decision assistant. Your task is to determine whether you have ENOUGH INFORMATION to answer the user's query, or if you need to ask clarifying questions.

            **IMPORTANT: Default to answering. Only ask clarifying questions when CRITICAL information is missing.**

            AVAILABLE CONTEXT:
            {context_text}

            USER QUERY: "{query}"

            CRITICAL DISTINCTION:
            - **INFORMATIONAL QUERY** (question seeking knowledge) → ALWAYS answer
            * Pattern: "What is...", "How does...", "Why...", "Can you explain..."
            * Example: "What is a database?" → Answer with explanation
            
            - **ACTION COMMAND** (imperative requesting execution) → Need specifics
            * Pattern: "Set up...", "Fix...", "Install...", "Debug...", "Deploy..."
            * Example: "Set up the database" → MUST ask: which database?

            DECISION RULES:

            **needs_clarification = FALSE** (You CAN and SHOULD answer) when:
            - Query is a **general knowledge question** that can be answered objectively
            * "What is X?", "How does X work?", "Explain X", "Why X?"
            * These ALWAYS can be answered - don't ask for clarification!
            - Context provides enough information to give a helpful answer
            - Query asks about something already discussed in recent messages or memory
            - You can make reasonable assumptions or give general guidance
            - Query is a greeting, acknowledgment, or casual conversation
            - Query asks "continue" or "next step" and context shows clear progression

            **needs_clarification = TRUE** (You NEED more info) ONLY when:
            - **IMPERATIVE COMMAND with missing specifics**: Query starts with action verb commanding you to DO something specific
            * **CRITICAL PATTERNS requiring clarification:**
                * "Set up [THE/A] X" → MUST clarify: WHICH specific X?
                * "Fix [THE] X" → MUST clarify: WHAT X? WHERE?
                * "Debug [THE] X" → MUST clarify: WHAT X exactly?
                * "Install X" → MUST clarify: WHICH X? WHERE?
                * "Deploy X" → MUST clarify: WHERE? HOW?
            * **Examples that ALWAYS need clarification:**
                * "Set up the database" → NEED: PostgreSQL? MySQL? MongoDB?
                * "Fix the bug" → NEED: what bug? in which file?
                * "Debug the error" → NEED: what error message?
                * "Install the dependencies" → NEED: which requirements.txt?
                * "Deploy the app" → NEED: where? production? staging?
            
            - **POSSESSIVE REFERENCE to unknown**: Query refers to "my X", "the X", "that X" not in context
            * "Fix MY code" → code not provided
            * "Use THAT approach" → approach not mentioned
            * "Continue with THE tutorial" → no tutorial in context
            
            - **PERSONALIZED CHOICE without context**: Query asks for recommendation FOR USER without user info
            * "Which framework should I use?" → need project details
            * "What should I learn next?" → need user level/goals

            **CONTRAST TABLE (must follow exactly):**
            | Query Type | Example | Decision |
            |------------|---------|----------|
            | Question about concept | "What is a database?" | needs_clarification=FALSE (answer) |
            | Command to setup | "Set up the database" | needs_clarification=TRUE (which db?) |
            | Question about fixing | "How do I fix bugs?" | needs_clarification=FALSE (answer) |
            | Command to fix | "Fix the bug" | needs_clarification=TRUE (what bug?) |
            | General how-to | "How to deploy apps?" | needs_clarification=FALSE (answer) |
            | Command to deploy | "Deploy the app" | needs_clarification=TRUE (where?) |

            **KEY PRINCIPLE: Question words (What/How/Why) = answer. Imperative verbs (Set up/Fix/Deploy) = clarify.**

            CLARIFYING QUESTIONS GUIDELINES (only if needs_clarification=true):
            - Ask ONLY when critical information is truly missing
            - Base questions on CONTEXT (don't ask what's already known)
            - Ask 1-3 specific, actionable questions
            - DON'T ask for preferences/options when you can give a complete answer
            - DON'T ask "would you like X or Y?" - just explain both!

            OUTPUT FORMAT (JSON):
            {{
            "needs_clarification": true/false,
            "clarifying_questions": ["Question 1", "Question 2"] or []
            }}

            EXAMPLES:

            Example 1: General knowledge - ALWAYS answer
            Query: "What is FastAPI?"
            Context: Any context (or empty)
            Decision: {{"needs_clarification": false, "clarifying_questions": []}}
            Reason: General knowledge question - answer with explanation, don't ask for clarification

            Example 2: General knowledge with context - ALWAYS answer
            Query: "What are the main features of it?"
            Context: "Recent: USER asked about FastAPI, ASSISTANT explained it's a Python framework"
            Decision: {{"needs_clarification": false, "clarifying_questions": []}}
            Reason: "it" = FastAPI from context, features are general knowledge - answer directly

            Example 3: Clear continuation - answer
            Query: "Continue where we left off"
            Context: "TODOS: Learn functions, Current goal: Learn Python, Recent: covered variables"
            Decision: {{"needs_clarification": false, "clarifying_questions": []}}
            Reason: Next step is clear from todos/context - proceed with functions

            Example 4: Vague specific request - NEED clarification
            Query: "Set up the database"
            Context: "Topics: [Web development], No database mentioned"
            Decision: {{"needs_clarification": true, "clarifying_questions": ["Which database system are you using? (PostgreSQL, MySQL, MongoDB, SQLite)"]}}
            Reason: User asking to set up THEIR database - need to know which one

            Example 5: Reference to unknown specific bug - NEED clarification
            Query: "Fix the bug"
            Context: "Recent: discussing project, no bug/error mentioned"
            Decision: {{"needs_clarification": true, "clarifying_questions": ["What bug or error are you experiencing?", "Can you describe the issue or share error messages?"]}}
            Reason: User asking to fix THEIR specific bug - need details

            Example 6: General how-to - CAN answer
            Query: "How do I set up a database?"
            Context: Any context
            Decision: {{"needs_clarification": false, "clarifying_questions": []}}
            Reason: General question about database setup process - can explain generally
            Query: "How should I structure my project?"
            Context: "Topics: [Python], User building web app"
            Decision: {{"needs_clarification": false, "clarifying_questions": []}}
            Reason: Can provide general project structure guidance for Python web apps

            Example 7: Need user-specific info - need clarification
            Query: "Which framework is best for me?"
            Context: "No user background, no project requirements, no constraints"
            Decision: {{"needs_clarification": true, "clarifying_questions": ["What type of project are you building?", "What's your experience level with Python?"]}}
            Reason: "Best for me" requires personal context not available

            Remember: **Bias towards answering!** General knowledge questions NEVER need clarification.

            Now analyze the CURRENT USER QUERY with the AVAILABLE CONTEXT above and output ONLY valid JSON, nothing else."""
                
    return prompt
