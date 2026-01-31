import pytest
from app.core.schemas import Message, SessionSummary, UserProfile
from app.llms.openai_client import OpenAIClient
from app.modules.rewriter import rewrite_query


class TestRewriter:
    """Test query rewriting with pronoun/reference resolution."""
    
    @pytest.fixture
    def llm_client(self):
        """Create OpenAI client for real API testing."""
        return OpenAIClient()
    
    def test_pronoun_resolution(self, llm_client):
        """Test resolving pronouns (it/that) from recent context."""
        recent = [
            Message(role="user", content="Tell me about FastAPI"),
            Message(role="assistant", content="FastAPI is a modern Python web framework for building APIs."),
        ]
        
        result = rewrite_query("What are its main features?", recent, llm_client)
        
        # Should detect ambiguous pronoun "its"
        assert result.is_ambiguous == True
        assert result.original_query == "What are its main features?"
        
        # Should resolve to FastAPI
        assert result.rewritten_query is not None
        assert "fastapi" in result.rewritten_query.lower()
        
        # Should reference recent messages
        assert len(result.referenced_messages) > 0
    
    def test_clear_query_no_rewrite(self, llm_client):
        """Test that clear queries are not rewritten."""
        recent = [
            Message(role="user", content="What is Django?"),
            Message(role="assistant", content="Django is a Python web framework."),
        ]
        
        result = rewrite_query("What is Flask?", recent, llm_client)
        
        # Should recognize as clear query
        assert result.is_ambiguous == False
        assert result.rewritten_query is None
        assert len(result.referenced_messages) == 0
    
    def test_context_usage_flags(self, llm_client):
        """Test that context usage flags are set correctly."""
        recent = []
        
        # Query asking to continue suggests needing goal/todos
        result = rewrite_query("Continue where we left off", recent, llm_client)
        
        # Should request context (at least current_goal or todos)
        context = result.context_usage
        assert (context.use_current_goal or context.use_todos or context.use_topics)
    
    def test_with_summary_context(self, llm_client):
        """Test rewriting with session summary available."""
        recent = [
            Message(role="user", content="I'm learning web development"),
            Message(role="assistant", content="Great! Let's start with the basics."),
        ]
        
        summary = SessionSummary(
            user_profile=UserProfile(prefs=["detailed explanations"], constraints=[], background="beginner"),
            current_goal="Learn web development",
            topics=["web basics"],
            key_facts=[],
            decisions=[],
            open_questions=[],
            todos=[]
        )
        
        result = rewrite_query("Tell me more about it", recent, llm_client, summary)
        
        # Should resolve "it" to web development
        assert result.is_ambiguous == True
        assert result.rewritten_query is not None
        assert "web" in result.rewritten_query.lower() or "development" in result.rewritten_query.lower()
