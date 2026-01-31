from abc import ABC, abstractmethod
from typing import List
from app.core.schemas import LLMMessage

class BaseLLM(ABC):
    """Base class for LLM clients."""
    
    @abstractmethod
    def chat(self, messages: List[LLMMessage], temperature: float = 0.7, max_tokens: int = 2000) -> str:
        """Send messages to LLM and return response."""
        pass