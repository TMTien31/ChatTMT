"""
Tests for Answer Generation Module
"""

import pytest
from app.core.schemas import AugmentedContext, Message
from app.modules.answer import generate_answer, generate_contextual_response
from app.llms.openai_client import OpenAIClient


class TestAnswerGeneration:
    """Test answer generation functionality."""
    
    @pytest.fixture
    def llm_client(self):
        """Create OpenAI client for testing."""
        return OpenAIClient()
    
    def test_answer_with_general_knowledge(self, llm_client):
        """Test answering general knowledge questions."""
        query = "What is Python?"
        
        # Minimal context for general knowledge question
        augmented = AugmentedContext(
            recent_messages=[],
            memory_fields_used=[],
            memory_context="",
            final_augmented_context="RECENT MESSAGES:\n(none)\n\nMEMORY:\n(none)"
        )
        
        answer = generate_answer(query, augmented, llm_client)
        
        # Should provide a proper answer about Python
        assert len(answer) > 50
        assert "python" in answer.lower()
        
    def test_answer_with_conversation_context(self, llm_client):
        """Test answering using conversation context."""
        query = "What did I say I wanted to build?"
        
        # Context mentions web app
        augmented = AugmentedContext(
            recent_messages=[
                Message(role="user", content="I want to build a web app"),
                Message(role="assistant", content="Great! What features?"),
            ],
            memory_fields_used=["topics"],
            memory_context="TOPICS DISCUSSED: Web development",
            final_augmented_context="RECENT MESSAGES:\nuser: I want to build a web app\nassistant: Great! What features?\n\nMEMORY:\nTOPICS: Web development"
        )
        
        answer = generate_answer(query, augmented, llm_client)
        
        # Should reference the web app from context
        assert len(answer) > 20
        assert "web app" in answer.lower() or "application" in answer.lower()
        
    def test_answer_with_memory_preferences(self, llm_client):
        """Test answering using user preferences from memory."""
        query = "Suggest a programming language for me"
        
        # Context has user preference for Python
        augmented = AugmentedContext(
            recent_messages=[],
            memory_fields_used=["prefs"],
            memory_context="USER PREFERENCES: Prefers Python, likes clean code",
            final_augmented_context="RECENT MESSAGES:\n(none)\n\nMEMORY:\nPREFS: Prefers Python, likes clean code"
        )
        
        answer = generate_answer(query, augmented, llm_client)
        
        # Should mention or recommend Python based on preferences
        assert len(answer) > 30
        assert "python" in answer.lower()
        
    def test_answer_with_ongoing_tasks(self, llm_client):
        """Test answering with awareness of ongoing tasks."""
        query = "What should I do next?"
        
        # Context has open tasks
        augmented = AugmentedContext(
            recent_messages=[
                Message(role="user", content="I need to setup database and write tests"),
                Message(role="assistant", content="Let's start with the database"),
            ],
            memory_fields_used=["todos", "topics"],
            memory_context="TODOS: [✗] Setup database [✗] Write tests\nTOPICS: Database setup, testing",
            final_augmented_context="RECENT MESSAGES:\nuser: I need to setup database and write tests\nassistant: Let's start with the database\n\nMEMORY:\nTODOS: [✗] Setup database [✗] Write tests\nTOPICS: Database setup, testing"
        )
        
        answer = generate_answer(query, augmented, llm_client)
        
        # Should reference the tasks
        assert len(answer) > 30
        assert "database" in answer.lower() or "test" in answer.lower()
        
    def test_contextual_response_with_metadata(self, llm_client):
        """Test generate_contextual_response returns answer with metadata."""
        query = "Hello"
        
        augmented = AugmentedContext(
            recent_messages=[
                Message(role="user", content="Previous message"),
                Message(role="assistant", content="Response"),
            ],
            memory_fields_used=["prefs", "topics"],
            memory_context="PREFS: Python\nTOPICS: Web dev",
            final_augmented_context="RECENT MESSAGES:\nuser: Previous message\n\nMEMORY:\nPREFS: Python"
        )
        
        result = generate_contextual_response(query, augmented, llm_client, include_metadata=True)
        
        # Should have answer and metadata
        assert "answer" in result
        assert "metadata" in result
        assert len(result["answer"]) > 10
        assert result["metadata"]["memory_fields_used"] == ["prefs", "topics"]
        assert result["metadata"]["recent_message_count"] == 2
        assert result["metadata"]["has_memory_context"] == True
        
    def test_answer_with_empty_context(self, llm_client):
        """Test answering with completely empty context."""
        query = "What's 2 + 2?"
        
        # No context at all
        augmented = AugmentedContext(
            recent_messages=[],
            memory_fields_used=[],
            memory_context="",
            final_augmented_context="RECENT MESSAGES:\n(none)\n\nMEMORY:\n(none)"
        )
        
        answer = generate_answer(query, augmented, llm_client)
        
        # Should still answer basic math
        assert len(answer) > 5
        assert "4" in answer
