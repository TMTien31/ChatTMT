"""
Tests for Pydantic schemas to validate structure, defaults, and serialization.
"""

import pytest
from datetime import datetime
from pydantic import ValidationError

from app.core.schemas import (
    Message,
    UserProfile,
    SessionSummary,
    SummarizationResult,
    SessionState,
    ContextUsage,
    RewriteResult,
    AugmentedContext,
    ClarificationResult,
    QueryUnderstandingResult,
    Answer,
    PromptPayload,
)

# 1. MESSAGE SCHEMA TESTS
class TestMessage:
    """Test Message model validation and serialization."""
    
    def test_message_valid_user_role(self):
        """Test creating message with user role."""
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.timestamp is None
    
    def test_message_valid_assistant_role(self):
        """Test creating message with assistant role."""
        msg = Message(role="assistant", content="Hi there")
        assert msg.role == "assistant"
        assert msg.content == "Hi there"
    
    def test_message_with_timestamp(self):
        """Test message with explicit timestamp."""
        now = datetime.now()
        msg = Message(role="user", content="Hello", timestamp=now)
        assert msg.timestamp == now
    
    def test_message_invalid_role(self):
        """Test that invalid role raises ValidationError."""
        with pytest.raises(ValidationError):
            Message(role="system", content="Hello")
    
    def test_message_serialization(self):
        """Test message can be serialized to dict and back."""
        msg = Message(role="user", content="Test message")
        data = msg.model_dump()
        
        assert data["role"] == "user"
        assert data["content"] == "Test message"
        
        # Reconstruct from dict
        msg2 = Message(**data)
        assert msg2.role == msg.role
        assert msg2.content == msg.content

# 2. USER PROFILE & SESSION SUMMARY TESTS
class TestUserProfile:
    """Test UserProfile model."""
    
    def test_user_profile_defaults(self):
        """Test UserProfile with default values."""
        profile = UserProfile()
        assert profile.prefs == []
        assert profile.constraints == []
        assert profile.background is None
    
    def test_user_profile_with_data(self):
        """Test UserProfile with data."""
        profile = UserProfile(
            prefs=["detailed", "code examples"],
            constraints=["3 developers", "2 months"],
            background="software engineer"
        )
        assert len(profile.prefs) == 2
        assert len(profile.constraints) == 2
        assert profile.background == "software engineer"


class TestSessionSummary:
    """Test SessionSummary model."""
    
    def test_session_summary_defaults(self):
        """Test SessionSummary with all defaults."""
        summary = SessionSummary()
        assert isinstance(summary.user_profile, UserProfile)
        assert summary.current_goal is None
        assert summary.topics == []
        assert summary.key_facts == []
        assert summary.decisions == []
        assert summary.open_questions == []
        assert summary.todos == []
    
    def test_session_summary_with_data(self):
        """Test SessionSummary with full data."""
        profile = UserProfile(background="engineer")
        summary = SessionSummary(
            user_profile=profile,
            current_goal="Build REST API",
            topics=["FastAPI", "PostgreSQL"],
            key_facts=["Using async", "100k users"],
            decisions=["FastAPI over Django"],
            open_questions=["Authentication approach?"],
            todos=["Docker setup"]
        )
        assert summary.user_profile.background == "engineer"
        assert summary.current_goal == "Build REST API"
        assert len(summary.topics) == 2
        assert len(summary.key_facts) == 2
    
    def test_session_summary_serialization(self):
        """Test SessionSummary can be serialized."""
        summary = SessionSummary(
            current_goal="Test goal",
            topics=["Topic1"]
        )
        data = summary.model_dump()
        
        assert data["current_goal"] == "Test goal"
        assert data["topics"] == ["Topic1"]
        
        # Reconstruct
        summary2 = SessionSummary(**data)
        assert summary2.current_goal == summary.current_goal


class TestSummarizationResult:
    """Test SummarizationResult model."""
    
    def test_summarization_result_for_summarization(self):
        """Test SummarizationResult for SUMMARIZATION operation."""
        summary = SessionSummary(current_goal="Test")
        result = SummarizationResult(
            session_summary=summary,
            summarized_up_to_turn=10,
            token_count_before=10000,
            token_count_after=800,
            was_compressed=False
        )
        assert result.summarized_up_to_turn == 10
        assert result.token_count_before == 10000
        assert result.token_count_after == 800
        assert result.was_compressed is False
    
    def test_summarization_result_for_compression(self):
        """Test SummarizationResult for COMPRESSION operation."""
        summary = SessionSummary(topics=["T1", "T2"])
        result = SummarizationResult(
            session_summary=summary,
            summarized_up_to_turn=19,
            token_count_before=2300,
            token_count_after=1400,
            was_compressed=True
        )
        assert result.was_compressed is True
        assert result.summarized_up_to_turn == 19


# 3. SESSION STATE TESTS
class TestSessionState:
    """Test SessionState model."""
    
    def test_session_state_requires_session_id(self):
        """Test that session_id is required."""
        with pytest.raises(ValidationError):
            SessionState()
    
    def test_session_state_with_minimal_data(self):
        """Test SessionState with only session_id."""
        state = SessionState(session_id="test-123")
        assert state.session_id == "test-123"
        assert isinstance(state.created_at, datetime)
        assert isinstance(state.last_updated, datetime)
        assert state.raw_messages == []
        assert state.summary is None
        assert state.summarized_up_to_turn is None
        assert state.total_turns == 0
        assert state.clarification_count == 0
    
    def test_session_state_with_messages(self):
        """Test SessionState with messages."""
        msg1 = Message(role="user", content="Hello")
        msg2 = Message(role="assistant", content="Hi")
        
        state = SessionState(
            session_id="test-456",
            raw_messages=[msg1, msg2],
            total_turns=1
        )
        assert len(state.raw_messages) == 2
        assert state.total_turns == 1
    
    def test_session_state_with_summary(self):
        """Test SessionState with summary."""
        summary = SessionSummary(current_goal="Test goal")
        state = SessionState(
            session_id="test-789",
            summary=summary,
            summarized_up_to_turn=10,
            total_turns=15
        )
        assert state.summary is not None
        assert state.summary.current_goal == "Test goal"
        assert state.summarized_up_to_turn == 10
        assert state.total_turns == 15
    
    def test_session_state_serialization(self):
        """Test SessionState can be fully serialized."""
        msg = Message(role="user", content="Test")
        summary = SessionSummary(topics=["Topic1"])
        
        state = SessionState(
            session_id="test-serialize",
            raw_messages=[msg],
            summary=summary,
            summarized_up_to_turn=5,
            total_turns=10,
            clarification_count=1
        )
        
        # Serialize to dict
        data = state.model_dump()
        assert data["session_id"] == "test-serialize"
        assert data["total_turns"] == 10
        assert data["clarification_count"] == 1
        
        # Reconstruct
        state2 = SessionState(**data)
        assert state2.session_id == state.session_id
        assert state2.total_turns == state.total_turns
        assert len(state2.raw_messages) == 1


# 4. CONTEXT USAGE TESTS
class TestContextUsage:
    """Test ContextUsage model."""
    
    def test_context_usage_defaults(self):
        """Test all flags default to False."""
        usage = ContextUsage()
        assert usage.use_user_profile is False
        assert usage.use_current_goal is False
        assert usage.use_topics is False
        assert usage.use_key_facts is False
        assert usage.use_decisions is False
        assert usage.use_open_questions is False
        assert usage.use_todos is False
    
    def test_context_usage_with_flags(self):
        """Test setting specific flags."""
        usage = ContextUsage(
            use_user_profile=True,
            use_key_facts=True,
            use_decisions=True
        )
        assert usage.use_user_profile is True
        assert usage.use_key_facts is True
        assert usage.use_decisions is True
        assert usage.use_topics is False  # Others still False


# 5. PIPELINE SCHEMAS TESTS
class TestRewriteResult:
    """Test RewriteResult model."""
    
    def test_rewrite_result_not_ambiguous(self):
        """Test RewriteResult for clear query."""
        result = RewriteResult(
            original_query="Build a REST API",
            is_ambiguous=False
        )
        assert result.is_ambiguous is False
        assert result.rewritten_query is None
        assert result.referenced_messages == []
        assert result.context_usage.use_key_facts is False
    
    def test_rewrite_result_ambiguous_with_rewrite(self):
        """Test RewriteResult for ambiguous query."""
        msg = Message(role="user", content="About FastAPI")
        usage = ContextUsage(use_key_facts=True)
        
        result = RewriteResult(
            original_query="What about it?",
            is_ambiguous=True,
            rewritten_query="What about FastAPI's async support?",
            referenced_messages=[msg],
            context_usage=usage
        )
        assert result.is_ambiguous is True
        assert result.rewritten_query is not None
        assert len(result.referenced_messages) == 1
        assert result.context_usage.use_key_facts is True


class TestAugmentedContext:
    """Test AugmentedContext model."""
    
    def test_augmented_context_minimal(self):
        """Test AugmentedContext with no memory."""
        context = AugmentedContext(
            recent_messages=[],
            memory_fields_used=[],
            memory_context="",
            final_augmented_context="Just recent messages"
        )
        assert context.recent_messages == []
        assert context.memory_fields_used == []
        assert context.memory_context == ""
    
    def test_augmented_context_with_memory(self):
        """Test AugmentedContext with memory fields."""
        msg = Message(role="user", content="Test")
        context = AugmentedContext(
            recent_messages=[msg],
            memory_fields_used=["user_profile", "key_facts"],
            memory_context="User: engineer. Facts: Using FastAPI",
            final_augmented_context="[Full context here]"
        )
        assert len(context.recent_messages) == 1
        assert len(context.memory_fields_used) == 2
        assert "engineer" in context.memory_context


class TestClarificationResult:
    """Test ClarificationResult model."""
    
    def test_clarification_not_needed(self):
        """Test ClarificationResult when no clarification needed."""
        result = ClarificationResult(
            needs_clarification=False
        )
        assert result.needs_clarification is False
        assert result.clarifying_questions == []
    
    def test_clarification_needed(self):
        """Test ClarificationResult when clarification needed."""
        result = ClarificationResult(
            needs_clarification=True,
            clarifying_questions=[
                "What do you want to set up?",
                "1) FastAPI? 2) PostgreSQL? 3) Docker?"
            ]
        )
        assert result.needs_clarification is True
        assert len(result.clarifying_questions) == 2


class TestQueryUnderstandingResult:
    """Test QueryUnderstandingResult composition."""
    
    def test_query_understanding_result_composition(self):
        """Test full pipeline result composition."""
        rewrite = RewriteResult(
            original_query="Test query",
            is_ambiguous=False
        )
        augment = AugmentedContext(
            final_augmented_context="Context"
        )
        clarify = ClarificationResult(
            needs_clarification=False
        )
        
        result = QueryUnderstandingResult(
            rewrite=rewrite,
            augment=augment,
            clarify=clarify
        )
        
        assert isinstance(result.rewrite, RewriteResult)
        assert isinstance(result.augment, AugmentedContext)
        assert isinstance(result.clarify, ClarificationResult)
        assert result.rewrite.original_query == "Test query"


# 6. ANSWER & PROMPT PAYLOAD TEST
class TestAnswer:
    """Test Answer model."""
    
    def test_answer_creation(self):
        """Test Answer creation."""
        answer = Answer(answer="This is the answer")
        assert answer.answer == "This is the answer"


class TestPromptPayload:
    """Test PromptPayload model."""
    
    def test_prompt_payload_creation(self):
        """Test PromptPayload for debugging."""
        payload = PromptPayload(
            system_prompt="You are a helpful assistant",
            augmented_context="Context here",
            user_query="What is X?"
        )
        assert "helpful assistant" in payload.system_prompt
        assert payload.user_query == "What is X?"


# 7. EDGE CASES & INTEGRATION TESTS
class TestEdgeCases:
    """Test edge cases and integration scenarios."""
    
    def test_empty_lists_serialize_correctly(self):
        """Test that empty lists serialize/deserialize properly."""
        summary = SessionSummary()
        data = summary.model_dump()
        
        assert data["topics"] == []
        assert data["key_facts"] == []
        
        # Reconstruct should preserve empty lists
        summary2 = SessionSummary(**data)
        assert summary2.topics == []
    
    def test_optional_fields_none_serialize(self):
        """Test Optional fields with None values."""
        summary = SessionSummary()
        data = summary.model_dump()
        
        assert data["current_goal"] is None
        
        # Reconstruct
        summary2 = SessionSummary(**data)
        assert summary2.current_goal is None
    
    def test_nested_model_serialization(self):
        """Test nested model (UserProfile in SessionSummary) serializes."""
        profile = UserProfile(background="engineer")
        summary = SessionSummary(user_profile=profile)
        
        data = summary.model_dump()
        assert data["user_profile"]["background"] == "engineer"
        
        # Reconstruct
        summary2 = SessionSummary(**data)
        assert summary2.user_profile.background == "engineer"
    
    def test_datetime_fields_serialize(self):
        """Test datetime fields can be serialized."""
        state = SessionState(session_id="test")
        
        # Use mode='json' to get ISO format strings
        data = state.model_dump(mode='json')
        
        # created_at and last_updated should be ISO format strings
        assert isinstance(data["created_at"], str)
        assert isinstance(data["last_updated"], str)
    
    def test_full_session_state_round_trip(self):
        """Test complete SessionState round-trip serialization."""
        # Create complex state
        msg1 = Message(role="user", content="Query 1")
        msg2 = Message(role="assistant", content="Answer 1")
        summary = SessionSummary(
            user_profile=UserProfile(background="engineer"),
            current_goal="Build API",
            topics=["FastAPI"],
            key_facts=["Using async"]
        )
        
        state = SessionState(
            session_id="round-trip-test",
            raw_messages=[msg1, msg2],
            summary=summary,
            summarized_up_to_turn=1,
            total_turns=2,
            clarification_count=0
        )
        
        # Serialize
        data = state.model_dump()
        
        # Reconstruct
        state2 = SessionState(**data)
        
        # Verify
        assert state2.session_id == state.session_id
        assert len(state2.raw_messages) == 2
        assert state2.summary.current_goal == "Build API"
        assert state2.total_turns == 2
        assert state2.clarification_count == 0


# RUN ALL TESTS
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
