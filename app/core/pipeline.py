"""
Query Understanding Pipeline

Orchestrates: Rewrite → Augment → Clarify/Answer flow.
"""

from dataclasses import dataclass
from typing import Optional, Union

from app.core.schemas import (
    Message, 
    RewriteResult, 
    AugmentedContext, 
    ClarificationResult,
    SessionSummary
)
from app.core.session import SessionManager
from app.llms.base import BaseLLM
from app.modules.rewriter import rewrite_query
from app.modules.augmenter import augment_context
from app.modules.clarifier import check_clarification_needed
from app.modules.answer import generate_answer
from app.utils.logger import get_logger
from app.utils.config import get_config

logger = get_logger(__name__)
config = get_config()


@dataclass
class PipelineResult:
    """Result from pipeline processing."""
    
    # Pipeline stages results
    rewrite_result: RewriteResult
    augmented_context: AugmentedContext
    clarification_result: Optional[ClarificationResult] = None
    
    # Final output (either answer or clarifying questions)
    needs_clarification: bool = False
    response: str = ""  # Either final answer or formatted clarification questions
    
    # Metadata
    was_summarized: bool = False
    was_compressed: bool = False


class QueryPipeline:
    """
    Orchestrates the query understanding and response pipeline.
    
    Flow:
    1. Step 0: Check tokens → Summarize/Compress if needed
    2. Step 1: Rewrite query (resolve pronouns, detect ambiguity, decide context needs)
    3. Step 2: Augment context (extract relevant memory fields)
    4. Step 3: Decision
       - If clarification needed AND under MAX_CLARIFICATION_ROUNDS → Ask questions
       - If MAX_CLARIFICATION_ROUNDS reached → Force answer
       - Else → Generate answer
    """
    
    def __init__(
        self,
        session_manager: SessionManager,
        llm_client: BaseLLM
    ):
        """
        Initialize pipeline.
        
        Args:
            session_manager: Session manager with current session state
            llm_client: LLM client for all LLM calls
        """
        self.session = session_manager
        self.llm = llm_client
        
        # Ensure session has LLM client for summarization
        if not self.session.llm_client:
            self.session.llm_client = llm_client
    
    def process(self, query: str) -> PipelineResult:
        """
        Process a user query through the full pipeline.
        
        Args:
            query: User's query text
            
        Returns:
            PipelineResult with all stage outputs and final response
        """
        logger.info(f"Processing query: '{query[:50]}...' at turn {self.session.total_turns + 1}")
        
        # Step 0: Check and perform summarization/compression if needed
        was_summarized = self.session.check_and_summarize()
        
        # Step 1: Rewrite query
        light_context = self.session.get_light_context()
        rewrite_result = rewrite_query(
            query=query,
            recent_messages=light_context,
            llm_client=self.llm,
            summary=self.session.summary
        )
        
        logger.info(f"Rewrite: ambiguous={rewrite_result.is_ambiguous}, "
                   f"rewritten={'Yes' if rewrite_result.rewritten_query else 'No'}")
        
        # Step 2: Augment context
        augmented_context = augment_context(
            recent_messages=self.session.get_recent_messages(),
            context_usage=rewrite_result.context_usage,
            summary=self.session.summary
        )
        
        logger.info(f"Augmented: memory_fields={augmented_context.memory_fields_used}, "
                   f"recent_msgs={len(augmented_context.recent_messages)}")
        
        # Step 3: Decision - Clarify or Answer?
        # Use rewritten query if available, else original
        effective_query = rewrite_result.rewritten_query or query
        
        # Check clarification
        clarification_result = check_clarification_needed(
            original_query=effective_query,
            augmented_context=augmented_context,
            llm_client=self.llm
        )
        
        # Decision logic
        if clarification_result.needs_clarification:
            # Check if we've hit max clarification rounds
            current_count = self.session.increment_clarification()
            
            if current_count >= config.MAX_CLARIFICATION_ROUNDS:
                logger.info(f"Max clarification rounds ({config.MAX_CLARIFICATION_ROUNDS}) reached - forcing answer")
                self.session.reset_clarification()
                
                # Force answer with generic helpful response
                answer = self._generate_forced_answer(effective_query, augmented_context)
                
                return PipelineResult(
                    rewrite_result=rewrite_result,
                    augmented_context=augmented_context,
                    clarification_result=clarification_result,
                    needs_clarification=False,
                    response=answer,
                    was_summarized=was_summarized
                )
            else:
                # Ask clarification
                logger.info(f"Clarification round {current_count}/{config.MAX_CLARIFICATION_ROUNDS}")
                questions = self._format_clarification_questions(
                    clarification_result.clarifying_questions
                )
                
                return PipelineResult(
                    rewrite_result=rewrite_result,
                    augmented_context=augmented_context,
                    clarification_result=clarification_result,
                    needs_clarification=True,
                    response=questions,
                    was_summarized=was_summarized
                )
        else:
            # Generate answer
            self.session.reset_clarification()
            answer = generate_answer(
                query=effective_query,
                augmented_context=augmented_context,
                llm=self.llm
            )
            
            logger.info(f"Generated answer: {len(answer)} chars")
            
            return PipelineResult(
                rewrite_result=rewrite_result,
                augmented_context=augmented_context,
                clarification_result=clarification_result,
                needs_clarification=False,
                response=answer,
                was_summarized=was_summarized
            )
    
    def process_and_record(self, query: str) -> PipelineResult:
        """
        Process query AND record the turn to session.
        
        Use this for complete turn handling (query + response stored).
        
        Args:
            query: User's query text
            
        Returns:
            PipelineResult
        """
        result = self.process(query)
        
        # Only record if we got an answer (not clarification)
        if not result.needs_clarification:
            self.session.add_turn(query, result.response)
            logger.debug(f"Recorded turn {self.session.total_turns}")
        
        return result
    
    def _format_clarification_questions(self, questions: list) -> str:
        """Format clarification questions for display."""
        if not questions:
            return "Could you please provide more details about your request?"
        
        if len(questions) == 1:
            return questions[0]
        
        # Format as numbered list
        formatted = "I need a bit more information:\n"
        for i, q in enumerate(questions, 1):
            formatted += f"{i}. {q}\n"
        return formatted.strip()
    
    def _generate_forced_answer(
        self, 
        query: str, 
        augmented_context: AugmentedContext
    ) -> str:
        """
        Generate an answer when max clarification rounds exceeded.
        
        Provides a helpful but generic response acknowledging the ambiguity.
        """
        # Try to generate best-effort answer
        try:
            answer = generate_answer(
                query=query,
                augmented_context=augmented_context,
                llm=self.llm
            )
            return answer
        except Exception as e:
            logger.error(f"Error generating forced answer: {e}")
            return ("I understand you need help, but I'm having trouble understanding "
                   "the specific request. Could you try rephrasing your question with "
                   "more details about what you're trying to accomplish?")