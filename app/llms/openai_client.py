from typing import List
import time
from openai import OpenAI, RateLimitError, APITimeoutError, APIConnectionError
from app.llms.base import BaseLLM
from app.core.schemas import LLMMessage
from app.utils.config import get_config
from app.utils.logger import get_logger

logger = get_logger(__name__)

class OpenAIClient(BaseLLM):
    def __init__(self):
        config = get_config()
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.model = config.OPENAI_MODEL
        logger.info(f"OpenAI initialized: {self.model}")
    
    def chat(self, messages: List[LLMMessage], temperature: float = None, max_tokens: int = None) -> str:
        """Call OpenAI API and return response."""
        config = get_config()
        
        # Use config defaults if not specified
        if temperature is None:
            temperature = config.OPENAI_TEMPERATURE
        if max_tokens is None:
            max_tokens = config.OPENAI_MAX_TOKENS
        
        # Convert LLMMessage objects to OpenAI format
        openai_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]
        
        logger.debug(f"Calling OpenAI API: {len(messages)} messages, temp={temperature}")
        
        max_retries = 3
        base_delay = 1
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=openai_messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                content = response.choices[0].message.content
                
                if response.usage:
                    logger.info(f"LLM call [prompt_tokens={response.usage.prompt_tokens}, "
                              f"completion_tokens={response.usage.completion_tokens}, "
                              f"total={response.usage.total_tokens}]")
                
                return content
                
            except (RateLimitError, APITimeoutError, APIConnectionError) as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(f"API error (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    logger.error(f"OpenAI API call failed after {max_retries} attempts: {e}")
                    raise
            except Exception as e:
                logger.error(f"OpenAI API call failed: {e}")
                raise