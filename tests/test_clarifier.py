import pytest
from app.core.schemas import Message, AugmentedContext, ContextUsage
from app.modules.clarifier import check_clarification_needed
from app.llms.openai_client import OpenAIClient


class TestClarifier:
    @pytest.fixture
    def llm_client(self):
        """OpenAI client for testing."""
        return OpenAIClient()
    
    def test_general_knowledge_no_clarification(self, llm_client):
        """Test that general knowledge questions don't need clarification."""
        query = "What is FastAPI?"
        
        # Context with recent chat about Python
        augmented = AugmentedContext(
            recent_messages=[
                Message(role="user", content="I want to learn Python web frameworks"),
                Message(role="assistant", content="Great! Let's explore some options."),
            ],
            memory_fields_used=[],
            memory_context="",
            final_augmented_context="RECENT MESSAGES:\nuser: I want to learn Python web frameworks\nassistant: Great! Let's explore some options."
        )
        
        result = check_clarification_needed(query, augmented, llm_client)
        
        # General knowledge question - should NOT need clarification
        assert result.needs_clarification == False
        assert len(result.clarifying_questions) == 0
    
    def test_vague_query_needs_clarification(self, llm_client):
        """Test that vague queries without context need clarification."""
        query = "Set up the database"
        
        # Context doesn't mention any database
        augmented = AugmentedContext(
            recent_messages=[
                Message(role="user", content="I'm building a web app"),
                Message(role="assistant", content="That's great! What features do you need?"),
            ],
            memory_fields_used=["topics"],
            memory_context="TOPICS DISCUSSED: Web development, Python",
            final_augmented_context="RECENT MESSAGES:\nuser: I'm building a web app\nassistant: That's great!\n\nMEMORY:\nTOPICS: Web development, Python"
        )
        
        result = check_clarification_needed(query, augmented, llm_client)
        
        # Vague query - should NEED clarification
        assert result.needs_clarification == True
        assert len(result.clarifying_questions) >= 1
        # Questions should be about database choice
        all_questions = " ".join(result.clarifying_questions).lower()
        assert any(word in all_questions for word in ["database", "postgresql", "mysql", "sqlite"])
    
    def test_context_provides_answer_no_clarification(self, llm_client):
        """Test that clear queries with sufficient context don't need clarification."""
        query = "What are the main features of it?"
        
        # Context clearly discusses FastAPI
        augmented = AugmentedContext(
            recent_messages=[
                Message(role="user", content="Tell me about FastAPI"),
                Message(role="assistant", content="FastAPI is a modern Python web framework with automatic API docs, async support, and type hints."),
            ],
            memory_fields_used=["topics"],
            memory_context="TOPICS DISCUSSED: FastAPI, Python frameworks",
            final_augmented_context="RECENT MESSAGES:\nuser: Tell me about FastAPI\nassistant: FastAPI is a modern Python framework with automatic API docs\n\nMEMORY:\nTOPICS: FastAPI, Python frameworks"
        )
        
        result = check_clarification_needed(query, augmented, llm_client)
        
        # Context provides clear reference - should NOT need clarification
        assert result.needs_clarification == False
        assert len(result.clarifying_questions) == 0
    
    def test_continuation_with_context_no_clarification(self, llm_client):
        """Test that continuation queries with clear context don't need clarification."""
        query = "Continue where we left off"
        
        # Context has clear goal and todos
        augmented = AugmentedContext(
            recent_messages=[
                Message(role="user", content="I want to learn Python basics"),
                Message(role="assistant", content="Great! We covered variables. Let's do functions next."),
            ],
            memory_fields_used=["current_goal", "topics", "todos"],
            memory_context="CURRENT GOAL: Learn Python basics\n\nTOPICS DISCUSSED: Variables\n\nTODOS:\n- Learn functions\n- Practice loops",
            final_augmented_context="RECENT MESSAGES:\nuser: Learn Python\nassistant: We covered variables\n\nMEMORY:\nCURRENT GOAL: Learn Python basics\nTOPICS: Variables\nTODOS: Learn functions, Practice loops"
        )
        
        result = check_clarification_needed(query, augmented, llm_client)
        
        # Context shows next step clearly - should NOT need clarification
        assert result.needs_clarification == False
    
    def test_ambiguous_reference_needs_clarification(self, llm_client):
        """Test that ambiguous references need clarification."""
        query = "Fix the bug"
        
        # Context doesn't mention any specific bug
        augmented = AugmentedContext(
            recent_messages=[
                Message(role="user", content="I'm working on my project"),
                Message(role="assistant", content="Great! How can I help?"),
            ],
            memory_fields_used=[],
            memory_context="",
            final_augmented_context="RECENT MESSAGES:\nuser: I'm working on my project\nassistant: How can I help?"
        )
        
        result = check_clarification_needed(query, augmented, llm_client)
        
        # Ambiguous reference - should NEED clarification
        assert result.needs_clarification == True
        assert len(result.clarifying_questions) >= 1
        # Questions should ask about the bug
        all_questions = " ".join(result.clarifying_questions).lower()
        assert any(word in all_questions for word in ["bug", "error", "issue", "problem", "what"])
    
    def test_question_limit(self, llm_client):
        """Test that clarifying questions are limited to max 3."""
        query = "Help me with my project"
        
        # Very vague context
        augmented = AugmentedContext(
            recent_messages=[],
            memory_fields_used=[],
            memory_context="",
            final_augmented_context="RECENT MESSAGES: (empty)"
        )
        
        result = check_clarification_needed(query, augmented, llm_client)
        
        # If clarification needed, should have at most 3 questions
        if result.needs_clarification:
            assert len(result.clarifying_questions) <= 3
            assert len(result.clarifying_questions) >= 1
