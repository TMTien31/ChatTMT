import pytest
from app.utils.tokenizer import (
    count_tokens,
    count_messages_tokens,
    count_summary_tokens,
)
from app.core.schemas import Message, UserProfile, SessionSummary


class TestCountTokens:
    
    def test_count_empty_string(self):
        assert count_tokens("") == 0
    
    def test_count_simple_text(self):
        text = "Hello, world!"
        count = count_tokens(text)
        assert count > 0
        assert count < 10
    
    def test_count_longer_text(self):
        text = "I want to build a REST API for an e-commerce platform using FastAPI and PostgreSQL."
        count = count_tokens(text)
        assert count > 10
        assert count < 30  # Should be around 20 tokens
    
    def test_count_special_characters(self):
        """Test counting with special characters."""
        text = "Code: `print('Hello')` and symbols: @#$%"
        count = count_tokens(text)
        assert count > 0
    
    def test_count_unicode(self):
        """Test counting with unicode characters."""
        text = "Xin chào! 你好! مرحبا!"
        count = count_tokens(text)
        assert count > 0


class TestCountMessagesTokens:
    """Test token counting for message lists."""
    
    def test_count_empty_messages(self):
        """Test counting empty message list."""
        assert count_messages_tokens([]) == 0
    
    def test_count_single_message(self):
        """Test counting single message."""
        msg = Message(role="user", content="Hello")
        count = count_messages_tokens([msg])
        
        # Should include: metadata (4) + role tokens + content tokens + priming (3)
        assert count > 7  # At minimum
    
    def test_count_multiple_messages(self):
        """Test counting multiple messages."""
        messages = [
            Message(role="user", content="What is FastAPI?"),
            Message(role="assistant", content="FastAPI is a modern Python web framework."),
            Message(role="user", content="How do I install it?"),
        ]
        count = count_messages_tokens(messages)
        
        # 3 messages * (4 overhead) + content tokens + priming (3)
        assert count > 15  # Should be around 30-40
    
    def test_count_conversation(self):
        """Test counting realistic conversation."""
        messages = [
            Message(role="user", content="I want to build a REST API"),
            Message(role="assistant", content="I recommend FastAPI or Django REST Framework"),
            Message(role="user", content="Tell me more about FastAPI"),
            Message(role="assistant", content="FastAPI is a modern, fast web framework for building APIs with Python 3.7+"),
        ]
        count = count_messages_tokens(messages)
        
        # Should be around 50-70 tokens
        assert count > 30
        assert count < 100
    
    def test_count_long_message(self):
        """Test counting very long message."""
        long_content = "word " * 1000  # 1000 words
        msg = Message(role="user", content=long_content)
        count = count_messages_tokens([msg])
        
        # Should be over 1000 tokens
        assert count > 1000


class TestCountSummaryTokens:
    """Test token counting for SessionSummary."""
    
    def test_count_empty_summary(self):
        """Test counting empty summary."""
        summary = SessionSummary()
        count = count_summary_tokens(summary)
        assert count == 0
    
    def test_count_summary_with_goal(self):
        """Test counting summary with only goal."""
        summary = SessionSummary(
            current_goal="Build a REST API"
        )
        count = count_summary_tokens(summary)
        assert count > 0
        assert count < 20
    
    def test_count_summary_with_topics(self):
        """Test counting summary with topics."""
        summary = SessionSummary(
            topics=["FastAPI", "PostgreSQL", "Docker", "Authentication"]
        )
        count = count_summary_tokens(summary)
        assert count > 5
        assert count < 30
    
    def test_count_summary_with_key_facts(self):
        """Test counting summary with key facts."""
        summary = SessionSummary(
            key_facts=[
                "Using FastAPI async features",
                "PostgreSQL database for 100k users",
                "Docker containerization required"
            ]
        )
        count = count_summary_tokens(summary)
        assert count > 10
        assert count < 50
    
    def test_count_full_summary(self):
        """Test counting comprehensive summary."""
        summary = SessionSummary(
            user_profile=UserProfile(
                prefs=["Detailed explanations", "Code examples"],
                constraints=["3 developers", "3-month timeline"],
                background="Software engineer with 5 years experience"
            ),
            current_goal="Build production-ready REST API for e-commerce",
            topics=["FastAPI", "PostgreSQL", "Docker", "AWS", "Authentication"],
            key_facts=[
                "Using FastAPI async for high performance",
                "PostgreSQL database with expected 100k users",
                "Docker containerization for deployment",
                "AWS hosting with auto-scaling"
            ],
            decisions=[
                "FastAPI over Django REST Framework",
                "PostgreSQL over MongoDB",
                "Docker over traditional deployment",
                "AWS over Azure"
            ],
            open_questions=[
                "Which authentication method to use?",
                "How to handle rate limiting?"
            ],
            todos=[
                "Set up Docker compose file",
                "Design database schema",
                "Implement authentication middleware"
            ]
        )
        count = count_summary_tokens(summary)
        
        # This should be substantial
        assert count > 100
        assert count < 500  # But not too large
    
    def test_count_summary_with_user_profile(self):
        """Test counting summary with detailed user profile."""
        summary = SessionSummary(
            user_profile=UserProfile(
                prefs=["Concise answers", "Practical examples"],
                constraints=["Limited budget", "Small team"],
                background="Junior developer learning web development"
            )
        )
        count = count_summary_tokens(summary)
        assert count > 10
        assert count < 50


class TestEdgeCases:
    """Test edge cases and special scenarios."""
    
    def test_none_handling_text(self):
        """Test that None text is handled gracefully."""
        # count_tokens expects string, but empty string should work
        assert count_tokens("") == 0
    
    def test_very_long_text(self):
        """Test token counting for very long text."""
        # Simulate a 5000-word document
        long_text = "word " * 5000
        count = count_tokens(long_text)
        
        # Should be around 5000+ tokens
        assert count > 4000
    
    def test_messages_with_empty_content(self):
        """Test messages with empty content."""
        messages = [
            Message(role="user", content=""),
            Message(role="assistant", content="")
        ]
        count = count_messages_tokens(messages)
        
        # Should still count overhead and priming
        assert count > 0
    
    def test_summary_partial_fields(self):
        """Test summary with only some fields populated."""
        summary = SessionSummary(
            current_goal="Test goal",
            topics=["Topic1"],
            # Other fields empty
        )
        count = count_summary_tokens(summary)
        assert count > 0
        assert count < 30


class TestIntegration:
    """Integration tests combining multiple components."""
    
    def test_realistic_session_tokens(self):
        """Test token counting for realistic session."""
        # Simulate conversation at turn 19 (before summarization)
        messages = []
        for i in range(19):
            messages.append(
                Message(
                    role="user", 
                    content=f"Question {i+1}: I want to know more about building a production-ready REST API. "
                            f"Specifically, how should I handle authentication, database connections, error handling, "
                            f"logging, testing, deployment, and monitoring? What are the best practices?"
                )
            )
            messages.append(
                Message(
                    role="assistant", 
                    content=f"Answer {i+1}: For building a production REST API, I recommend using FastAPI with PostgreSQL. "
                            f"For authentication, use OAuth2 with JWT tokens. Set up connection pooling for the database "
                            f"using SQLAlchemy with async support. Implement comprehensive error handling with custom exception "
                            f"handlers. Use structured logging with JSON format. Write unit tests with pytest and integration "
                            f"tests with TestClient. Deploy using Docker containers on AWS ECS or Kubernetes. Monitor with "
                            f"Prometheus and Grafana for metrics, and use CloudWatch for logs."
                )
            )
        
        # 38 messages total
        count = count_messages_tokens(messages)
        
        # Should be approaching 10k threshold (realistic long conversation)
        assert count > 2500  # ~2.9k tokens for 19 turns with detailed responses
        print(f"\n19 turns = {count} tokens ({count/1000:.1f}k)")
    
    def test_after_summarization_tokens(self):
        """Test tokens after summarization."""
        # After summarization: keep 10 recent messages
        recent_messages = [
            Message(role="user", content=f"Question {i}") 
            for i in range(10)
        ]
        messages_tokens = count_messages_tokens(recent_messages)
        
        # Plus a summary
        summary = SessionSummary(
            current_goal="Build REST API",
            topics=["FastAPI", "PostgreSQL", "Docker"],
            key_facts=["Using async", "100k users", "Containerized"],
            decisions=["FastAPI chosen", "PostgreSQL chosen"]
        )
        summary_tokens = count_summary_tokens(summary)
        
        total_tokens = messages_tokens + summary_tokens
        
        # Should be much less than original 10k
        assert total_tokens < 1000
        print(f"\nAfter summarization: {total_tokens} tokens")
        print(f"  - Messages: {messages_tokens} tokens")
        print(f"  - Summary: {summary_tokens} tokens")


# ============================================================================
# RUN ALL TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])
