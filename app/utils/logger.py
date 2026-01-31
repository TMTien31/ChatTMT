import logging
import sys
from pathlib import Path
from typing import Optional
from app.utils.config import get_config

def setup_logger(
    name: str,
    level: Optional[str] = None,
    log_file: Optional[str] = None,
    log_to_console: Optional[bool] = None
) -> logging.Logger:
    """
    Set up a simple logger with file and/or console handlers.
    
    Args:
        name: Logger name (usually __name__ of the module)
        level: Log level (DEBUG, INFO, WARNING, ERROR). If None, uses config
        log_file: Path to log file. If None, uses config
        log_to_console: Whether to log to console. If None, uses config
    
    Returns:
        Configured logger instance
    """
    config = get_config()
    
    # Use config values if not provided
    if level is None:
        level = config.LOG_LEVEL
    if log_file is None:
        log_file = config.LOG_FILE
    if log_to_console is None:
        log_to_console = config.LOG_TO_CONSOLE
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Simple formatter for both file and console
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler
    if log_file:
        # Create log directory if needed
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger


# Global logger cache
_loggers: dict[str, logging.Logger] = {}

def get_logger(name: str) -> logging.Logger:
    """
    Get or create a logger for the given module.
    
    Args:
        name: Logger name (usually __name__ of the module)
    
    Returns:
        Logger instance
    """
    if name not in _loggers:
        _loggers[name] = setup_logger(name)
    return _loggers[name]

def clear_loggers() -> None:
    """Clear all cached loggers. Useful for testing."""
    global _loggers
    for logger in _loggers.values():
        logger.handlers.clear()
    _loggers.clear()