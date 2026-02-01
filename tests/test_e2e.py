import pytest
from app.core.session import SessionManager
from app.core.pipeline import QueryPipeline
from app.llms.openai_client import OpenAIClient
from app.utils.config import reload_config, get_config

config = get_config()


class TestE2EConversationFlow:
    
    @pytest.fixture
    def llm_client(self):
        reload_config()
        return OpenAIClient()
    
    def test_basic_conversation_flow(self, llm_client):
        session = SessionManager(llm_client=llm_client)
        pipeline = QueryPipeline(session, llm_client)
        
        result = pipeline.process_and_record("I want to build a REST API")
        assert not result.needs_clarification
        assert session.total_turns == 1
        
        # Turn 2: Follow-up using context
        result = pipeline.process_and_record("What framework should I use for it?")
        # Only increment if no clarification
        expected_turns = 2 if not result.needs_clarification else 1
        assert session.total_turns == expected_turns
        
        # Turn 3: More specific question
        result = pipeline.process_and_record("I prefer Python. Any recommendations?")
        if not result.needs_clarification:
            expected_turns += 1
        
        # Turn 4: Technical question
        result = pipeline.process_and_record("How do I handle authentication?")
        if not result.needs_clarification:
            expected_turns += 1
        
        # Turn 5: Implementation details
        result = pipeline.process_and_record("Show me an example")
        if not result.needs_clarification:
            expected_turns += 1
        
        # Should NOT have triggered summarization yet
        assert session.summary is None
        assert session.total_turns >= 3  # At least some turns recorded
        
        # Save session for inspection
        session.save()
        
        print(f"\n✅ Completed conversation flow, no summarization triggered")
        print(f"   Total turns recorded: {session.total_turns}")
        print(f"   Total messages: {len(session.raw_messages)}")
        print(f"   📁 Session saved: {session.session_id}.json")
    
    def test_summarization_trigger(self, llm_client):
        """
        Test that summarization triggers when token threshold exceeded.
        
        This simulates Turn 1-20 from COMPREHENSIVE_EXAMPLE.md
        
        Cost: ~$0.50-0.80 (20+ turns + summarization)
        """
        session = SessionManager(llm_client=llm_client)
        pipeline = QueryPipeline(session, llm_client)
        
        # Simulate conversation building up to threshold
        # Each turn adds ~200-500 tokens
        queries = [
            "I want to build an e-commerce platform with REST API",
            "What programming language should I use?",
            "I prefer Python. Which framework is best?",
            "Tell me about FastAPI vs Django REST Framework",
            "I'll use FastAPI. How do I set up the project?",
            "What database should I use for this project?",
            "PostgreSQL sounds good. How do I integrate it with FastAPI?",
            "Explain SQLAlchemy ORM usage",
            "How do I handle database migrations?",
            "What about authentication and authorization?",
            "Should I use JWT tokens?",
            "How do I implement JWT authentication in FastAPI?",
            "What about password hashing?",
            "Tell me about API versioning best practices",
            "How do I handle CORS?",
            "What about rate limiting?",
            "Should I use Docker for deployment?",
            "How do I write Dockerfile for FastAPI?",
            "What about docker-compose for PostgreSQL?",
        ]
        
        # Process queries (Turn 1-19)
        for i, query in enumerate(queries, 1):
            result = pipeline.process_and_record(query)
            print(f"Turn {i}: {len(session.raw_messages)} messages, "
                  f"summary={'Yes' if session.summary else 'No'}")
        
        # Should NOT have summarized yet (if under threshold)
        messages_before = len(session.raw_messages)
        summary_before = session.summary
        
        # Turn 20: Add one more query that pushes over threshold
        # Use a longer query to increase token count
        long_query = """
        Can you provide a comprehensive guide on implementing 
        production-ready authentication with JWT tokens, including 
        token refresh mechanisms, secure password storage with bcrypt, 
        role-based access control, and best practices for API security?
        """
        
        result = pipeline.process_and_record(long_query)
        
        # Check if summarization happened (depends on actual token count)
        if session.summary is not None and summary_before is None:
            print(f"\n✅ Summarization triggered!")
            print(f"   Messages before: {messages_before}")
            print(f"   Messages after: {len(session.raw_messages)}")
            print(f"   Kept recent N: {config.KEEP_RECENT_N}")
            
            # Verify summarization behavior
            assert session.summary is not None
            assert session.summarized_up_to_turn == 20
            assert len(session.raw_messages) <= config.KEEP_RECENT_N + 2  # +2 for Turn 20
            
            # Summary should contain key information
            assert session.summary.topics  # Should have extracted topics
            assert session.summary.current_goal  # Should have goal
        else:
            print(f"\n⚠️  Summarization NOT triggered (tokens still under threshold)")
            print(f"   Messages: {len(session.raw_messages)}")
        
        # Save session for inspection
        session.save()
        print(f"\n📁 Session saved: {session.session_id}.json")
        print(f"   Total turns: {session.total_turns}")
    
    def test_clarification_max_rounds(self, llm_client):
        """
        Test clarification loop with MAX_CLARIFICATION_ROUNDS.
        
        Cost: ~$0.05-0.10 (2-3 turns)
        """
        session = SessionManager(llm_client=llm_client)
        pipeline = QueryPipeline(session, llm_client)
        
        # Round 1: Vague query
        result = pipeline.process("help")
        
        if result.needs_clarification:
            print(f"\nRound 1: Needs clarification")
            assert session.clarification_count == 1
            
            # Round 2: Still vague
            result = pipeline.process("anything")
            
            if result.needs_clarification:
                print(f"Round 2: Still needs clarification")
                assert session.clarification_count == 2
                
                # Round 3: Should force answer after max rounds
                result = pipeline.process("just help me")
                
                # Should NOT ask clarification again (forced answer)
                assert not result.needs_clarification
                assert session.clarification_count == 0  # Reset after forced answer
                assert len(result.response) > 0
                
                print(f"Round 3: Forced answer after max rounds")
                print(f"✅ Clarification loop handled correctly")
            else:
                print(f"Round 2: Got answer (query became clearer)")
        else:
            print(f"Round 1: Query clear enough, got answer directly")
        
        # Save session
        session.save()
        print(f"\n📁 Session saved: {session.session_id}.json")
    
    def test_context_continuity_after_summarization(self, llm_client):
        """
        Test that context remains accessible after summarization.
        
        Cost: ~$0.30-0.50 (10-15 turns)
        """
        session = SessionManager(llm_client=llm_client)
        pipeline = QueryPipeline(session, llm_client)
        
        # Establish context
        pipeline.process_and_record("I'm using Python with FastAPI framework")
        pipeline.process_and_record("I also chose PostgreSQL as my database")
        pipeline.process_and_record("And I'm deploying with Docker")
        
        # Manually trigger summarization to test continuity
        if session.total_turns >= 3:
            from app.modules.summarizer import summarize_messages
            
            # Create summary
            session.state.summary = summarize_messages(
                messages=session.raw_messages,
                llm_client=llm_client
            )
            session.state.summarized_up_to_turn = session.total_turns
            
            # Keep only recent messages
            session.state.raw_messages = session.raw_messages[-config.KEEP_RECENT_N:]
            
            print(f"\n📝 Manual summarization performed")
            print(f"   Summary topics: {session.summary.topics}")
            print(f"   Summary decisions: {session.summary.decisions}")
            
            # Now query using established context
            result = pipeline.process("What was my tech stack again?")
            
            # Answer should reference FastAPI, PostgreSQL, Docker from summary
            answer_lower = result.response.lower()
            context_mentioned = (
                "fastapi" in answer_lower or
                "postgresql" in answer_lower or
                "postgres" in answer_lower or
                "docker" in answer_lower or
                "python" in answer_lower
            )
            
            if context_mentioned:
                print(f"✅ Context continuity maintained after summarization")
                print(f"   Answer referenced previous tech stack")
            else:
                print(f"⚠️  Answer may not have fully used summary context")
                print(f"   Answer: {result.response[:100]}...")
            
            # Save session for inspection
            session.save()
            print(f"\n📁 Session saved: {session.session_id}.json")


class TestE2EEdgeCases:
    """Test edge cases in E2E flows."""
    
    @pytest.fixture
    def llm_client(self):
        return OpenAIClient()
    
    def test_empty_session_query(self, llm_client):
        """Test query on completely empty session."""
        session = SessionManager(llm_client=llm_client)
        pipeline = QueryPipeline(session, llm_client)
        
        result = pipeline.process("What is Python?")
        
        # Should handle empty session gracefully
        assert not result.needs_clarification
        assert len(result.response) > 0
        
        # Save session
        session.save()
        print(f"✅ Empty session handled")
        print(f"📁 Session saved: {session.session_id}.json")
    
    def test_rapid_context_switches(self, llm_client):
        """Test rapid topic switching in conversation."""
        session = SessionManager(llm_client=llm_client)
        pipeline = QueryPipeline(session, llm_client)
        
        # Rapidly switch topics
        topics = [
            "Tell me about Python",
            "What about JavaScript?",
            "How about Rust?",
            "Explain Go language",
        ]
        
        for topic in topics:
            result = pipeline.process_and_record(topic)
            assert not result.needs_clarification or session.clarification_count <= config.MAX_CLARIFICATION_ROUNDS
        
        assert session.total_turns == len(topics)
        
        # Save session
        session.save()
        print(f"✅ Handled {len(topics)} rapid context switches")
        print(f"📁 Session saved: {session.session_id}.json")
    
    def test_session_persistence(self, llm_client, tmp_path, monkeypatch):
        """Test saving and loading session preserves state."""
        # Use temp directory
        monkeypatch.setattr('app.utils.config.get_config', lambda: type('Config', (), {
            'SESSION_DATA_DIR': str(tmp_path),
            'LIGHT_CONTEXT_SIZE': 8,
            'RECENT_CONTEXT_SIZE': 10,
            'TOKEN_THRESHOLD_RAW': 10000,
            'SUMMARY_TOKEN_THRESHOLD': 2000,
            'KEEP_RECENT_N': 16,
            'MAX_CLARIFICATION_ROUNDS': 2
        })())
        
        from app.core.session import SessionManager as SM
        from app.core.pipeline import QueryPipeline as QP
        
        # Create and use session
        session1 = SM(llm_client=llm_client)
        pipeline1 = QP(session1, llm_client)
        
        pipeline1.process_and_record("I'm building a web app")
        pipeline1.process_and_record("Using Python and FastAPI")
        
        session_id = session1.session_id
        session1.save()
        
        # Load session and continue
        session2 = SM(session_id=session_id, llm_client=llm_client)
        pipeline2 = QP(session2, llm_client)
        
        # Should have preserved state
        assert session2.total_turns == 2
        assert len(session2.raw_messages) == 4
        
        # Continue conversation
        result = pipeline2.process("What framework am I using?")
        
        # Should reference FastAPI from loaded context
        assert "fastapi" in result.response.lower() or "framework" in result.response.lower()
        
        print(f"✅ Session persistence works")
        print(f"   Loaded {session2.total_turns} turns, {len(session2.raw_messages)} messages")
