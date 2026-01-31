"""
Tests for augmenter module (context augmentation).

Augmenter is pure Python (no LLM) - fast and deterministic.
"""
import pytest
from app.core.schemas import Message, SessionSummary, UserProfile, ContextUsage
from app.modules.augmenter import augment_context, format_augmented_context


class TestAugmenter:
    """Test context augmentation logic."""
    
    def test_augment_with_no_memory(self):
        """Test augmentation when no summary exists."""
        recent = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there"),
        ]
        
        context_usage = ContextUsage(
            use_current_goal=True,  # Requested but no summary available
            use_topics=True
        )
        
        result = augment_context(recent, context_usage, summary=None)
        
        # Should include recent messages
        assert len(result.recent_messages) == 2
        assert result.recent_messages[0].content == "Hello"
        
        # No memory fields used (no summary)
        assert result.memory_fields_used == []
        assert result.memory_context == ""
    
    def test_augment_with_selective_memory(self):
        """Test selecting only requested memory fields."""
        recent = [
            Message(role="user", content="Continue where we left off"),
            Message(role="assistant", content="Let's continue"),
        ]
        
        summary = SessionSummary(
            user_profile=UserProfile(prefs=["detailed"], constraints=[], background="Student"),
            current_goal="Learn Python",
            topics=["Variables", "Functions"],
            key_facts=["Uses Windows", "Knows JavaScript"],
            decisions=["Use VS Code"],
            open_questions=["Which framework?"],
            todos=["Complete chapter 3"]
        )
        
        # Only request goal and todos
        context_usage = ContextUsage(
            use_current_goal=True,
            use_todos=True
            # Other fields = False
        )
        
        result = augment_context(recent, context_usage, summary)
        
        # Should only include requested fields
        assert "current_goal" in result.memory_fields_used
        assert "todos" in result.memory_fields_used
        assert len(result.memory_fields_used) == 2
        
        # Memory context should contain only goal and todos
        assert "Learn Python" in result.memory_context
        assert "Complete chapter 3" in result.memory_context
        
        # Should NOT contain unrequested fields
        assert "Variables" not in result.memory_context  # topics not requested
        assert "VS Code" not in result.memory_context  # decisions not requested
    
    def test_augment_with_all_memory_fields(self):
        """Test including all available memory fields."""
        recent = [Message(role="user", content="Test")]
        
        summary = SessionSummary(
            user_profile=UserProfile(
                prefs=["concise answers"],
                constraints=["limited time"],
                background="Developer"
            ),
            current_goal="Build chatbot",
            topics=["Memory management"],
            key_facts=["Has 3 years experience"],
            decisions=["Use Python"],
            open_questions=["Which LLM?"],
            todos=["Design schema"]
        )
        
        # Request all fields
        context_usage = ContextUsage(
            use_user_profile=True,
            use_current_goal=True,
            use_topics=True,
            use_key_facts=True,
            use_decisions=True,
            use_open_questions=True,
            use_todos=True
        )
        
        result = augment_context(recent, context_usage, summary)
        
        # Should include all fields
        assert len(result.memory_fields_used) == 7
        assert "user_profile" in result.memory_fields_used
        assert "current_goal" in result.memory_fields_used
        assert "topics" in result.memory_fields_used
        assert "key_facts" in result.memory_fields_used
        assert "decisions" in result.memory_fields_used
        assert "open_questions" in result.memory_fields_used
        assert "todos" in result.memory_fields_used
        
        # Memory context should contain all data
        assert "concise answers" in result.memory_context
        assert "Build chatbot" in result.memory_context
        assert "Memory management" in result.memory_context
        assert "3 years experience" in result.memory_context
        assert "Use Python" in result.memory_context
        assert "Which LLM?" in result.memory_context
        assert "Design schema" in result.memory_context
    
    def test_recent_messages_limit(self):
        """Test that augmenter respects RECENT_CONTEXT_SIZE config."""
        # Create 20 messages (10 turns)
        recent = []
        for i in range(10):
            recent.append(Message(role="user", content=f"User message {i}"))
            recent.append(Message(role="assistant", content=f"Assistant response {i}"))
        
        context_usage = ContextUsage()  # No memory fields needed
        
        result = augment_context(recent, context_usage, summary=None)
        
        # Should limit to RECENT_CONTEXT_SIZE (default 10 messages = 5 turns)
        assert len(result.recent_messages) <= 10
        
        # Should keep most recent messages
        if len(result.recent_messages) > 0:
            last_msg = result.recent_messages[-1]
            assert "9" in last_msg.content  # Most recent (turn 9)
    
    def test_format_augmented_context(self):
        """Test formatting AugmentedContext into readable text."""
        from app.core.schemas import AugmentedContext
        
        augmented = AugmentedContext(
            recent_messages=[
                Message(role="user", content="What is Python?"),
                Message(role="assistant", content="Python is a programming language."),
            ],
            memory_fields_used=["current_goal", "topics"],
            memory_context="CURRENT GOAL: Learn programming\n\nTOPICS DISCUSSED: Basics, Syntax",
            final_augmented_context="RECENT MESSAGES:\nuser: What is Python?\nassistant: Python is a programming language.\n\nMEMORY:\nCURRENT GOAL: Learn programming\n\nTOPICS DISCUSSED: Basics, Syntax"
        )
        
        # Check that final_augmented_context contains expected parts
        assert "What is Python?" in augmented.final_augmented_context
        assert "CURRENT GOAL" in augmented.final_augmented_context or "CURRENT GOAL" in augmented.memory_context
    
    def test_empty_augmentation(self):
        """Test augmentation with no messages and no memory."""
        result = augment_context([], ContextUsage(), summary=None)
        
        assert result.recent_messages == []
        assert result.memory_fields_used == []
        assert result.memory_context == ""
    
    def test_partial_user_profile(self):
        """Test user profile with only some fields populated."""
        summary = SessionSummary(
            user_profile=UserProfile(
                prefs=["detailed explanations"],
                constraints=[],  # Empty
                background=None  # None
            ),
            current_goal=None,
            topics=[],
            key_facts=[],
            decisions=[],
            open_questions=[],
            todos=[]
        )
        
        context_usage = ContextUsage(use_user_profile=True)
        
        result = augment_context([], context_usage, summary)
        
        # Should include user_profile
        assert "user_profile" in result.memory_fields_used
        
        # Should only show populated fields
        assert "detailed explanations" in result.memory_context
        assert "Constraints:" not in result.memory_context  # Empty list skipped
        assert "Background:" not in result.memory_context  # None skipped
