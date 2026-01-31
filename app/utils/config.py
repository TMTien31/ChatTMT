import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()
class Config:
    def __init__(self):
        # LLM API CONFIGURATION
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
        self.OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4")
        self.OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))
        self.OPENAI_MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "2000"))

        # TOKEN THRESHOLDS
        #Trigger summarization when raw_messages exceeds this token count.
        self.TOKEN_THRESHOLD_RAW = int(os.getenv("TOKEN_THRESHOLD_RAW", "10000"))

        #Trigger compression when summary exceeds this token count.
        self.SUMMARY_TOKEN_THRESHOLD = int(os.getenv("SUMMARY_TOKEN_THRESHOLD", "2000"))
        
        # Number of recent messages to keep after summarization.
        # Note: 1 turn = 2 messages (user + assistant)
        # Example: 16 messages = last 8 turns
        self.KEEP_RECENT_N = int(os.getenv("KEEP_RECENT_N", "16"))

        # QUERY UNDERSTANDING CONFIGURATION
        # Number of recent messages to use for query rewriting (light context).
        # Note: 1 turn = 2 messages (user + assistant)
        # Default 8 messages = last 4 turns
        self.LIGHT_CONTEXT_SIZE = int(os.getenv("LIGHT_CONTEXT_SIZE", "8"))
        
        # Number of recent messages to include in augmentation.
        # Note: 1 turn = 2 messages (user + assistant)
        # Default 10 messages = last 5 turns
        self.RECENT_CONTEXT_SIZE = int(os.getenv("RECENT_CONTEXT_SIZE", "10"))
        
        #Maximum consecutive clarification attempts.
        self.MAX_CLARIFICATION_ROUNDS = int(os.getenv("MAX_CLARIFICATION_ROUNDS", "2"))
        
        # MODULE-SPECIFIC LLM SETTINGS
        # Temperature: lower = more deterministic, higher = more creative
        # Summarizer: low temp for consistent extraction
        self.SUMMARIZER_TEMPERATURE = float(os.getenv("SUMMARIZER_TEMPERATURE", "0.2"))
        self.SUMMARIZER_MAX_TOKENS = int(os.getenv("SUMMARIZER_MAX_TOKENS", "2000"))
        
        # Rewriter: low temp for accurate query understanding
        self.REWRITER_TEMPERATURE = float(os.getenv("REWRITER_TEMPERATURE", "0.2"))
        self.REWRITER_MAX_TOKENS = int(os.getenv("REWRITER_MAX_TOKENS", "1000"))
        
        # Clarifier: slightly higher for natural questions
        self.CLARIFIER_TEMPERATURE = float(os.getenv("CLARIFIER_TEMPERATURE", "0.3"))
        self.CLARIFIER_MAX_TOKENS = int(os.getenv("CLARIFIER_MAX_TOKENS", "500"))
        
        # Answer: higher temp for natural, creative responses
        self.ANSWER_TEMPERATURE = float(os.getenv("ANSWER_TEMPERATURE", "0.7"))
        self.ANSWER_MAX_TOKENS = int(os.getenv("ANSWER_MAX_TOKENS", "2000"))
        
        # SESSION CONFIGURATION
        #Directory to store session state files.
        self.SESSION_DATA_DIR = os.getenv("SESSION_DATA_DIR", "data/sessions")
        
        # LOGGING CONFIGURATION
        #Logging level (DEBUG, INFO, WARNING, ERROR).
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
        
        #Path to log file.
        self.LOG_FILE = os.getenv("LOG_FILE", "logs/app.log")
        
        #Whether to log to console in addition to file.
        self.LOG_TO_CONSOLE = os.getenv("LOG_TO_CONSOLE", "true").lower() == "true"
    
    def validate(self) -> None:
        """Validate configuration and raise errors if invalid."""
        if not self.OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY is required. "
                "Please set it in .env file or environment variables."
            )
        
        if self.TOKEN_THRESHOLD_RAW <= 0:
            raise ValueError("TOKEN_THRESHOLD_RAW must be positive")
        
        if self.SUMMARY_TOKEN_THRESHOLD <= 0:
            raise ValueError("SUMMARY_TOKEN_THRESHOLD must be positive")
        
        if self.KEEP_RECENT_N <= 0:
            raise ValueError("KEEP_RECENT_N must be positive")
        
        if self.MAX_CLARIFICATION_ROUNDS < 0:
            raise ValueError("MAX_CLARIFICATION_ROUNDS must be non-negative")

# Singleton instance
_config: Optional[Config] = None

def get_config() -> Config:
    """
    Get the global configuration instance (singleton).
    
    Returns:
        Config instance with loaded environment variables
    """
    global _config
    if _config is None:
        _config = Config()
    return _config

def reload_config() -> Config:
    """
    Reload configuration from environment variables.
    Useful for testing or runtime configuration changes.
    
    Returns:
        New Config instance
    """
    global _config
    load_dotenv(override=True)
    _config = Config()
    return _config