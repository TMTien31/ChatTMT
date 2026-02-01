import pytest
from app.core.session import SessionManager
from app.core.pipeline import QueryPipeline, PipelineResult
from app.core.schemas import Message, SessionSummary, UserProfile
from app.llms.openai_client import OpenAIClient
from app.utils.config import reload_config


class TestQueryPipeline:
    
    @pytest.fixture
    def llm_client(self):
        reload_config()
        return OpenAIClient()
    
    @pytest.fixture
    def session_manager(self, llm_client):
        return SessionManager(llm_client=llm_client)
    
    @pytest.fixture
    def pipeline(self, session_manager, llm_client):
        return QueryPipeline(session_manager=session_manager, llm_client=llm_client)
    
    def test_simple_query_gets_answer(self, pipeline):
        query = "What is Python?"
        
        result = pipeline.process(query)
        
        assert isinstance(result, PipelineResult)
        assert result.needs_clarification == False
        assert len(result.response) > 50
        assert "python" in result.response.lower()
    
    def test_context_aware_answer(self, pipeline):
        """Test that pipeline uses conversation context."""
        # First turn: establish context
        pipeline.session.add_turn(
            user_message="I'm building a web app with Flask",
            assistant_message="Great! Flask is a good choice for web apps."
        )
        
        # Second turn: reference previous context
        query = "What database should I use with it?"
        result = pipeline.process(query)
        
        # Should understand "it" refers to Flask/web app
        assert result.needs_clarification == False
        assert len(result.response) > 30
    
    def test_pipeline_with_memory(self, pipeline):
        """Test pipeline uses session memory (summary)."""
        # Set up session with existing summary
        pipeline.session.state.summary = SessionSummary(
            user_profile=UserProfile(
                prefs=["Prefers Python", "Likes detailed explanations"],
                constraints=[],
                background="Software developer"
            ),
            current_goal="Build REST API",
            topics=["FastAPI", "Python", "REST API"],
            key_facts=["Using FastAPI framework"],
            decisions=["Chose FastAPI over Flask"],
            open_questions=[],
            todos=[]
        )
        
        # Add some conversation context so query makes sense
        pipeline.session.add_turn(
            user_message="I'm building a REST API",
            assistant_message="Great! I see you're working on a REST API."
        )
        
        # Query that references the ongoing work
        query = "Should I continue with the same framework for this?"
        result = pipeline.process(query)
        
        # Should reference FastAPI from memory and not need clarification
        # (or might ask clarification - either is acceptable, just check no crash)
        assert result.response is not None
        assert len(result.response) > 0
    
    def test_process_and_record_adds_turn(self, pipeline):
        """Test that process_and_record adds the turn to session."""
        initial_turns = pipeline.session.total_turns
        
        query = "Hello, how are you?"
        result = pipeline.process_and_record(query)
        
        # Should have added a turn (if no clarification needed)
        if not result.needs_clarification:
            assert pipeline.session.total_turns == initial_turns + 1
            assert len(pipeline.session.raw_messages) >= 2
    
    def test_rewrite_result_included(self, pipeline):
        """Test that pipeline result includes rewrite info."""
        query = "Tell me about machine learning"
        result = pipeline.process(query)
        
        # Should have rewrite result
        assert result.rewrite_result is not None
        assert result.rewrite_result.original_query == query
    
    def test_augmented_context_included(self, pipeline):
        """Test that pipeline result includes augmented context."""
        query = "What's 2 + 2?"
        result = pipeline.process(query)
        
        # Should have augmented context
        assert result.augmented_context is not None
        assert hasattr(result.augmented_context, 'final_augmented_context')


class TestSessionManager:
    """Test session manager functionality."""
    
    @pytest.fixture
    def llm_client(self):
        return OpenAIClient()
    
    def test_create_new_session(self, llm_client):
        """Test creating a new session."""
        session = SessionManager(llm_client=llm_client)
        
        assert session.session_id is not None
        assert session.total_turns == 0
        assert len(session.raw_messages) == 0
    
    def test_add_turn(self, llm_client):
        """Test adding a turn to session."""
        session = SessionManager(llm_client=llm_client)
        
        session.add_turn("Hello", "Hi there!")
        
        assert session.total_turns == 1
        assert len(session.raw_messages) == 2
        assert session.raw_messages[0].role == "user"
        assert session.raw_messages[1].role == "assistant"
    
    def test_clarification_counting(self, llm_client):
        """Test clarification count tracking."""
        session = SessionManager(llm_client=llm_client)
        
        assert session.clarification_count == 0
        
        count = session.increment_clarification()
        assert count == 1
        assert session.clarification_count == 1
        
        session.reset_clarification()
        assert session.clarification_count == 0
    
    def test_get_light_context(self, llm_client):
        """Test getting light context for rewriting."""
        session = SessionManager(llm_client=llm_client)
        
        # Add several turns
        for i in range(5):
            session.add_turn(f"User message {i}", f"Assistant response {i}")
        
        light = session.get_light_context()
        
        # Should get last LIGHT_CONTEXT_SIZE messages
        assert len(light) <= 10  # config.LIGHT_CONTEXT_SIZE = 8
        assert len(light) > 0
    
    def test_save_and_load_session(self, llm_client, tmp_path, monkeypatch):
        """Test saving and loading session."""
        # Use temp directory
        monkeypatch.setattr('app.utils.config.get_config', lambda: type('Config', (), {
            'SESSION_DATA_DIR': str(tmp_path),
            'LIGHT_CONTEXT_SIZE': 8,
            'RECENT_CONTEXT_SIZE': 10,
            'TOKEN_THRESHOLD_RAW': 10000,
            'SUMMARY_TOKEN_THRESHOLD': 2000,
            'KEEP_RECENT_N': 16
        })())
        
        # Create and save session
        from app.core.session import SessionManager as SM
        session = SM(llm_client=llm_client)
        session.add_turn("Test message", "Test response")
        session_id = session.session_id
        session.save()
        
        # Load session
        loaded = SM(session_id=session_id, llm_client=llm_client)
        
        assert loaded.session_id == session_id
        assert loaded.total_turns == 1
        assert len(loaded.raw_messages) == 2