"""
Tests for conversation summarization using real OpenAI API.
Phase 2.1: Summarizer Tests
"""

import pytest
from app.core.schemas import Message, SessionSummary, UserProfile
from app.modules.summarizer import summarize_messages, compress_summary
from app.llms.openai_client import OpenAIClient
from app.utils.config import reload_config


@pytest.fixture(scope="function")
def llm_client():
    """Initialize OpenAI client for each test."""
    reload_config()
    return OpenAIClient()


class TestSummarizer:
    """Core summarizer tests with real OpenAI API."""
    
    def test_basic_summarization(self, llm_client):
        """Test basic message summarization."""
        messages = [
            Message(role="user", content="Hi, I'm John. I'm a Python developer interested in AI."),
            Message(role="assistant", content="Hello John! Great to meet you."),
            Message(role="user", content="Can you help me learn about neural networks?"),
            Message(role="assistant", content="Of course! Let's start with the basics.")
        ]
        
        summary = summarize_messages(messages, llm_client)
        
        # Verify structure
        assert isinstance(summary, SessionSummary)
        assert summary.user_profile is not None
        assert summary.topics is not None
        assert len(summary.topics) > 0
        
        # Verify content extraction
        all_text = str(summary).lower()
        assert "ai" in all_text or "neural" in all_text or "python" in all_text
    
    def test_compression(self, llm_client):
        """Test summary compression with new messages."""
        old_summary = SessionSummary(
            user_profile=UserProfile(prefs=["Python", "ML"], constraints=[], background="Developer"),
            current_goal="Learn deep learning",
            topics=["Neural networks", "Backpropagation"],
            key_facts=["Has 2 years Python experience"],
            decisions=["Start with PyTorch"],
            open_questions=["Which GPU to buy?"],
            todos=["Complete course chapter 3"]
        )
        
        new_messages = [
            Message(role="user", content="I've decided to use TensorFlow instead."),
            Message(role="assistant", content="Good choice! TensorFlow has great documentation.")
        ]
        
        compressed = compress_summary(old_summary, new_messages, llm_client)
        
        # Verify structure
        assert isinstance(compressed, SessionSummary)
        assert compressed.user_profile is not None
        
        # Should incorporate new information
        all_text = str(compressed).lower()
        assert "tensorflow" in all_text or "tf" in all_text
    
    def test_empty_messages(self, llm_client):
        """Test handling of edge cases."""
        # Empty messages should still return valid summary
        summary = summarize_messages([], llm_client)
        assert isinstance(summary, SessionSummary)
        
        # Single message
        single_msg = [Message(role="user", content="Hello")]
        summary2 = summarize_messages(single_msg, llm_client)
        assert isinstance(summary2, SessionSummary)
    
    def test_full_workflow(self, llm_client):
        """Test complete summarization workflow: summarize → compress."""
        # Initial conversation
        messages1 = [
            Message(role="user", content="I want to build a chatbot with memory."),
            Message(role="assistant", content="Great! We can use summarization for long-term memory."),
            Message(role="user", content="How does that work?"),
            Message(role="assistant", content="We summarize old messages to save tokens.")
        ]
        
        # First summarization
        summary1 = summarize_messages(messages1, llm_client)
        assert isinstance(summary1, SessionSummary)
        all_text = " ".join(summary1.topics).lower()
        assert "chatbot" in all_text or "memory" in all_text
        
        # Continue conversation
        messages2 = [
            Message(role="user", content="What about vector databases?"),
            Message(role="assistant", content="Good question! Vector DBs help with retrieval.")
        ]
        
        # Compression
        summary2 = compress_summary(summary1, messages2, llm_client)
        assert isinstance(summary2, SessionSummary)
        
        # Should contain both old and new topics
        all_topics = " ".join(summary2.topics).lower()
        assert "chatbot" in all_topics or "memory" in all_topics or "vector" in all_topics


# ============================================================================
# RUN ALL TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])