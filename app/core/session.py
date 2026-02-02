import json
import os
from datetime import datetime
from typing import Optional
from uuid import uuid4

from app.core.schemas import SessionState, SessionSummary, Message
from app.utils.tokenizer import count_tokens
from app.llms.base import BaseLLM
from app.modules.summarizer import summarize_messages, compress_summary
from app.utils.logger import get_logger
from app.utils.config import get_config

logger = get_logger(__name__)
config = get_config()


class SessionManager:
    """
    Manages a single session's state and lifecycle.
    
    Responsibilities:
    - Create/load/save session state
    - Add messages and track turns
    - Trigger summarization when raw_messages exceeds TOKEN_THRESHOLD_RAW
    - Trigger compression when summary exceeds SUMMARY_TOKEN_THRESHOLD
    """
    
    def __init__(
        self, 
        session_id: Optional[str] = None,
        llm_client: Optional[BaseLLM] = None
    ):
        """
        Initialize session manager.
        
        Args:
            session_id: Existing session ID to load, or None to create new
            llm_client: LLM client for summarization (required for summarize/compress)
        """
        self.llm_client = llm_client
        
        if session_id:
            self.state = self._load_session(session_id)
            logger.info(f"Loaded session: {session_id}")
        else:
            self.state = SessionState(session_id=str(uuid4()))
            logger.info(f"Created new session: {self.state.session_id}")
    
    @property
    def session_id(self) -> str:
        return self.state.session_id
    
    @property
    def raw_messages(self) -> list:
        return self.state.raw_messages
    
    @property
    def summary(self) -> Optional[SessionSummary]:
        return self.state.summary
    
    @property
    def total_turns(self) -> int:
        return self.state.total_turns
    
    @property
    def clarification_count(self) -> int:
        return self.state.clarification_count
    
    @property
    def summarized_up_to_turn(self) -> Optional[int]:
        return self.state.summarized_up_to_turn
    
    def add_turn(self, user_message: str, assistant_message: str) -> None:
        """
        Add a complete turn (user + assistant messages) to session.
        
        Args:
            user_message: User's message content
            assistant_message: Assistant's response content
        """
        now = datetime.now()
        self.state.raw_messages.append(Message(role="user", content=user_message, timestamp=now))
        self.state.raw_messages.append(Message(role="assistant", content=assistant_message, timestamp=now))
        self.state.total_turns += 1
        self.state.last_updated = now
        
        logger.debug(f"Added turn {self.state.total_turns}: "
                    f"user={len(user_message)} chars, assistant={len(assistant_message)} chars")
    
    def increment_clarification(self) -> int:
        """Increment and return clarification count."""
        self.state.clarification_count += 1
        return self.state.clarification_count
    
    def reset_clarification(self) -> None:
        """Reset clarification count after successful answer."""
        self.state.clarification_count = 0
    
    def check_and_summarize(self) -> bool:
        """
        Check if summarization/compression is needed and perform if so.
        
        Returns:
            True if summarization/compression was performed
        """
        if not self.llm_client:
            logger.warning("No LLM client - cannot summarize")
            return False
        
        # Check raw_messages tokens
        raw_text = " ".join(m.content for m in self.state.raw_messages)
        raw_tokens = count_tokens(raw_text)
        
        # Check summary tokens
        summary_tokens = 0
        if self.state.summary:
            summary_text = self.state.summary.model_dump_json()
            summary_tokens = count_tokens(summary_text)
        
        # Log current state
        logger.info(f"Session: turns={self.total_turns}, messages={len(self.raw_messages)}, "
                   f"raw_tokens={raw_tokens}, summary_tokens={summary_tokens}")
        
        if raw_tokens > config.TOKEN_THRESHOLD_RAW:
            logger.info(f"Trigger: raw_tokens ({raw_tokens}) > threshold ({config.TOKEN_THRESHOLD_RAW}) → Summarizing")
            self._perform_summarization()
            return True
        
        if self.state.summary and summary_tokens > config.SUMMARY_TOKEN_THRESHOLD:
            logger.info(f"Trigger: summary_tokens ({summary_tokens}) > threshold ({config.SUMMARY_TOKEN_THRESHOLD}) → Compressing")
            self._perform_compression()
            return True
        
        return False
    
    def _perform_summarization(self) -> None:
        """
        Summarize raw_messages into summary, keep KEEP_RECENT_N messages.
        """
        logger.info(f"Performing summarization at turn {self.state.total_turns}")
        
        # Summarize all messages (optionally merge with existing summary)
        new_summary = summarize_messages(
            messages=self.state.raw_messages,
            llm_client=self.llm_client,
            existing_summary=self.state.summary
        )
        
        # Keep recent messages
        self.state.raw_messages = self.state.raw_messages[-config.KEEP_RECENT_N:]
        
        # Update state
        self.state.summary = new_summary
        self.state.summarized_up_to_turn = self.state.total_turns
        self.state.last_updated = datetime.now()
        
        logger.info(f"Summarization complete: kept {len(self.state.raw_messages)} recent messages, "
                   f"summarized_up_to_turn={self.state.summarized_up_to_turn}")
    
    def _perform_compression(self) -> None:
        """
        Compress existing summary to reduce tokens.
        """
        if not self.state.summary:
            logger.warning("No summary to compress")
            return
            
        logger.info("Performing summary compression")
        
        compressed = compress_summary(
            old_summary=self.state.summary,
            new_messages=[],
            llm_client=self.llm_client
        )
        
        self.state.summary = compressed
        self.state.last_updated = datetime.now()
        
        logger.info("Compression complete")
    
    def get_recent_messages(self, count: Optional[int] = None) -> list:
        """
        Get recent messages for context.
        
        Args:
            count: Number of messages to get (defaults to RECENT_CONTEXT_SIZE)
            
        Returns:
            List of recent Message objects
        """
        if count is None:
            count = config.RECENT_CONTEXT_SIZE
        return self.state.raw_messages[-count:]
    
    def get_light_context(self) -> list:
        """Get light context for query rewriting (LIGHT_CONTEXT_SIZE messages)."""
        return self.state.raw_messages[-config.LIGHT_CONTEXT_SIZE:]
    
    def save(self) -> None:
        os.makedirs(config.SESSION_DATA_DIR, exist_ok=True)
        filepath = os.path.join(config.SESSION_DATA_DIR, f"{self.session_id}.json")
        
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(self.state.model_dump_json(indent=2))
            logger.debug(f"Saved session to {filepath}")
        except (IOError, OSError) as e:
            logger.error(f"Failed to save session {self.session_id}: {e}")
            raise
    
    def _load_session(self, session_id: str) -> SessionState:
        """Load session state from disk."""
        filepath = os.path.join(config.SESSION_DATA_DIR, f"{session_id}.json")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Session not found: {session_id}")
        
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        return SessionState.model_validate(data)
    
    @classmethod
    def load(cls, session_id: str, llm_client: Optional[BaseLLM] = None) -> "SessionManager":
        """
        Load an existing session from disk.
        
        Args:
            session_id: Session ID to load
            llm_client: Optional LLM client for operations
            
        Returns:
            SessionManager instance with loaded state
        """
        filepath = os.path.join(config.SESSION_DATA_DIR, f"{session_id}.json")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Session not found: {session_id}")
        
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        state = SessionState.model_validate(data)
        
        # Create new instance with loaded state
        instance = cls.__new__(cls)
        instance.state = state
        instance.llm_client = llm_client
        
        logger.info(f"Loaded session: {session_id} ({state.total_turns} turns)")
        return instance
    
    @classmethod
    def list_sessions(cls) -> list:
        """List all saved session IDs."""
        if not os.path.exists(config.SESSION_DATA_DIR):
            return []
        
        sessions = []
        for filename in os.listdir(config.SESSION_DATA_DIR):
            if filename.endswith(".json"):
                sessions.append(filename[:-5])  # Remove .json
        return sessions
    
    @classmethod
    def delete_session(cls, session_id: str) -> bool:
        """Delete a saved session."""
        filepath = os.path.join(config.SESSION_DATA_DIR, f"{session_id}.json")
        if os.path.exists(filepath):
            os.remove(filepath)
            logger.info(f"Deleted session: {session_id}")
            return True
        return False