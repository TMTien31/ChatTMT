"""
E2E Test - Generate 3 conversation logs demonstrating:
1. Session memory being triggered (summarization after 10k+ tokens)
2. Ambiguous user queries

Topics:
- Conversation 1: Web Development (English)
- Conversation 2: Economics & Finance (English)  
- Conversation 3: Vietnamese History & Culture (Vietnamese)

Run: pytest tests/test_e2e.py -v -s
Output: data/sessions/*.json
"""
import pytest
from app.core.session import SessionManager
from app.core.pipeline import QueryPipeline
from app.llms.openai_client import OpenAIClient
from app.utils.config import get_config

config = get_config()


class TestConversationLogs:
    """Generate 3 conversation logs for assignment submission."""
    
    @pytest.fixture
    def llm_client(self):
        # Force reload config from .env to avoid mock keys from other tests
        from app.utils.config import reload_config
        reload_config()
        return OpenAIClient()
    
    def test_conversation_1_web_development(self, llm_client):
        """
        Topic: Web Development with FastAPI
        Language: English
        Features: Summarization trigger, ambiguous queries, context references
        
        Expected output: ~20+ turns, session memory triggered
        """
        session = SessionManager(llm_client=llm_client)
        pipeline = QueryPipeline(session, llm_client)
        
        queries = [
            # Phase 1: Project setup (turns 1-5)
            "I want to build an e-commerce platform with REST API",
            "What programming language should I use for it?",
            "I prefer Python. Which framework is best for building APIs?",
            "Tell me more about FastAPI vs Django REST Framework",
            "I'll go with FastAPI. How do I set up the initial project structure?",
            
            # Phase 2: Database (turns 6-10)
            "What database should I use for this project?",
            "PostgreSQL sounds good. How do I integrate it with FastAPI?",
            "Can you explain SQLAlchemy ORM usage in more detail?",
            "How do I handle database migrations with it?",  # Ambiguous: "it" refers to SQLAlchemy
            "What about that migration tool you mentioned?",  # Ambiguous: refers to previous answer
            
            # Phase 3: Authentication (turns 11-15)
            "Now I need to add authentication. What are my options?",
            "Should I use JWT tokens?",
            "How do I implement JWT authentication in FastAPI?",
            "What about password hashing?",  # Ambiguous: context-dependent
            "Can you show me an example of that?",  # Ambiguous: refers to password hashing
            
            # Phase 4: API Design (turns 16-20)
            "Tell me about API versioning best practices",
            "How do I handle CORS in my application?",  # Ambiguous: "my application" needs context
            "What about rate limiting?",
            "How do I implement the first one?",  # Ambiguous: refers to a rate limiting method
            "Should I use Docker for deployment?",
            
            # Phase 5: Advanced topics (turns 21-25) - Trigger summarization
            "What about docker-compose for PostgreSQL?",
            "How do I set up CI/CD for this project?",
            "Can you explain the deployment strategy you mentioned earlier?",  # Ambiguous: context reference
            "What about monitoring and logging?",
            """
            Can you provide a comprehensive guide on implementing 
            production-ready authentication with JWT tokens, including 
            token refresh mechanisms, secure password storage with bcrypt, 
            role-based access control, and best practices for API security?
            """,
        ]
        
        print(f"\n{'='*60}")
        print("CONVERSATION 1: Web Development (English)")
        print(f"{'='*60}")
        
        for i, query in enumerate(queries, 1):
            result = pipeline.process_and_record(query)
            status = "✓" if not result.needs_clarification else "? (clarification)"
            print(f"Turn {i:2d}: {status} | Messages: {len(session.raw_messages):2d} | Summary: {'Yes' if session.summary else 'No'}")
        
        # Verify requirements
        print(f"\n{'='*40}")
        print("VERIFICATION:")
        print(f"  Total turns: {session.total_turns}")
        print(f"  Total messages: {len(session.raw_messages)}")
        print(f"  Summary triggered: {session.summary is not None}")
        if session.summary:
            print(f"  Summarized up to turn: {session.summarized_up_to_turn}")
            print(f"  Topics: {session.summary.topics[:3]}...")
        
        # Save session
        session.save()
        print(f"\n📁 Saved: data/sessions/{session.session_id}.json")
        
        # Assertions
        assert session.total_turns >= 20, "Should have at least 20 turns"
        assert session.summary is not None, "Summarization should have triggered"
    
    def test_conversation_2_economics_finance(self, llm_client):
        """
        Topic: Economics & Personal Finance
        Language: English
        Features: Summarization trigger, ambiguous queries, topic switches
        
        Expected output: ~20+ turns, session memory triggered
        """
        session = SessionManager(llm_client=llm_client)
        pipeline = QueryPipeline(session, llm_client)
        
        queries = [
            # Phase 1: Basic concepts (turns 1-5)
            "I want to learn about investing. Where should I start?",
            "What's the difference between stocks and bonds?",
            "Which one is better for beginners?",  # Ambiguous: refers to stocks vs bonds
            "Tell me more about that option",  # Ambiguous: needs context
            "How do I actually buy them?",  # Ambiguous: stocks or bonds?
            
            # Phase 2: Stock market (turns 6-10)
            "Explain how the stock market works",
            "What are the major stock exchanges?",
            "How do I analyze stocks before buying?",
            "What about that ratio you mentioned?",  # Ambiguous: P/E ratio or other?
            "Can you give me an example using a real company?",
            
            # Phase 3: Investment strategies (turns 11-15)
            "What investment strategies should I consider?",
            "Explain dollar-cost averaging in detail",
            "How does it compare to the other strategy?",  # Ambiguous: which other strategy?
            "What about diversification?",
            "How should I balance it?",  # Ambiguous: portfolio balance
            
            # Phase 4: Risk & returns (turns 16-20)
            "How do I calculate investment returns?",
            "What about compound interest?",
            "Can you show me the formula for that?",  # Ambiguous: compound interest formula
            "What risks should I be aware of?",
            "How do I mitigate those?",  # Ambiguous: which risks specifically
            
            # Phase 5: Advanced topics (turns 21-25) - Trigger summarization
            "Tell me about index funds vs mutual funds",
            "What are ETFs and how do they differ?",
            "Which one would you recommend for my situation?",  # Ambiguous: needs context
            "What about tax implications of investing?",
            """
            Can you provide a comprehensive investment plan for a beginner 
            with $10,000 to invest, considering risk tolerance, diversification,
            tax efficiency, and long-term wealth building strategies including
            retirement accounts, index funds, and emergency fund allocation?
            """,
        ]
        
        print(f"\n{'='*60}")
        print("CONVERSATION 2: Economics & Finance (English)")
        print(f"{'='*60}")
        
        for i, query in enumerate(queries, 1):
            result = pipeline.process_and_record(query)
            status = "✓" if not result.needs_clarification else "? (clarification)"
            print(f"Turn {i:2d}: {status} | Messages: {len(session.raw_messages):2d} | Summary: {'Yes' if session.summary else 'No'}")
        
        # Verify requirements
        print(f"\n{'='*40}")
        print("VERIFICATION:")
        print(f"  Total turns: {session.total_turns}")
        print(f"  Total messages: {len(session.raw_messages)}")
        print(f"  Summary triggered: {session.summary is not None}")
        if session.summary:
            print(f"  Summarized up to turn: {session.summarized_up_to_turn}")
            print(f"  Topics: {session.summary.topics[:3]}...")
        
        # Save session
        session.save()
        print(f"\n📁 Saved: data/sessions/{session.session_id}.json")
        
        # Assertions
        assert session.total_turns >= 20, "Should have at least 20 turns"
        assert session.summary is not None, "Summarization should have triggered"
    
    def test_conversation_3_vietnamese_history_culture(self, llm_client):
        """
        Topic: Vietnamese History & Culture
        Language: Vietnamese
        Features: Summarization trigger, ambiguous queries, cultural context
        
        Expected output: ~20+ turns, session memory triggered
        """
        session = SessionManager(llm_client=llm_client)
        pipeline = QueryPipeline(session, llm_client)
        
        queries = [
            # Giai đoạn 1: Lịch sử Việt Nam (turns 1-5)
            "Tôi muốn tìm hiểu về lịch sử Việt Nam. Nên bắt đầu từ đâu?",
            "Kể cho tôi về thời kỳ Hùng Vương và nguồn gốc dân tộc Việt",
            "Thời kỳ đó kéo dài bao lâu?",  # Ambiguous: thời kỳ nào?
            "Còn về truyền thuyết Con Rồng Cháu Tiên thì sao?",
            "Nó có ý nghĩa gì với người Việt?",  # Ambiguous: "nó" là gì?
            
            # Giai đoạn 2: Các triều đại (turns 6-10)
            "Hãy kể về các triều đại phong kiến Việt Nam",
            "Triều đại nào hùng mạnh nhất?",
            "Tại sao bạn cho rằng như vậy?",  # Ambiguous: cần ngữ cảnh
            "Còn về triều Nguyễn thì sao?",
            "Vị vua nào nổi tiếng nhất trong triều đó?",  # Ambiguous: triều nào?
            
            # Giai đoạn 3: Văn hóa (turns 11-15)
            "Giờ tôi muốn tìm hiểu về văn hóa Việt Nam",
            "Những lễ hội truyền thống quan trọng nhất là gì?",
            "Kể chi tiết về cái đầu tiên",  # Ambiguous: lễ hội nào?
            "Ý nghĩa của nó là gì?",  # Ambiguous: lễ hội hay phong tục?
            "Còn về Tết Nguyên Đán thì sao?",
            
            # Giai đoạn 4: Nghệ thuật (turns 16-20)
            "Nghệ thuật truyền thống Việt Nam có gì đặc biệt?",
            "Hát chèo và cải lương khác nhau như thế nào?",
            "Cái nào phổ biến hơn ở miền nào?",  # Ambiguous: chèo hay cải lương?
            "Còn về tranh Đông Hồ thì sao?",
            "Nó được làm như thế nào?",  # Ambiguous: "nó" là gì?
            
            # Giai đoạn 5: Tổng hợp (turns 21-25) - Trigger summarization
            "Ẩm thực Việt Nam có những đặc điểm gì nổi bật?",
            "Món ăn nào đại diện cho mỗi miền?",
            "Phở có nguồn gốc từ đâu và phát triển như thế nào?",
            "So sánh phở Bắc và phở Nam",
            """
            Hãy tổng hợp cho tôi một bài viết chi tiết về bản sắc văn hóa 
            Việt Nam, bao gồm lịch sử hình thành, các giá trị truyền thống,
            lễ hội, nghệ thuật, ẩm thực và cách người Việt gìn giữ văn hóa
            trong thời đại toàn cầu hóa ngày nay?
            """,
        ]
        
        print(f"\n{'='*60}")
        print("CONVERSATION 3: Lịch sử & Văn hóa Việt Nam (Tiếng Việt)")
        print(f"{'='*60}")
        
        for i, query in enumerate(queries, 1):
            result = pipeline.process_and_record(query)
            status = "✓" if not result.needs_clarification else "? (clarification)"
            print(f"Turn {i:2d}: {status} | Messages: {len(session.raw_messages):2d} | Summary: {'Yes' if session.summary else 'No'}")
        
        # Verify requirements
        print(f"\n{'='*40}")
        print("VERIFICATION:")
        print(f"  Total turns: {session.total_turns}")
        print(f"  Total messages: {len(session.raw_messages)}")
        print(f"  Summary triggered: {session.summary is not None}")
        if session.summary:
            print(f"  Summarized up to turn: {session.summarized_up_to_turn}")
            print(f"  Topics: {session.summary.topics[:3]}...")
        
        # Save session
        session.save()
        print(f"\n📁 Saved: data/sessions/{session.session_id}.json")
        
        # Assertions
        assert session.total_turns >= 20, "Should have at least 20 turns"
        assert session.summary is not None, "Summarization should have triggered"


class TestAmbiguousQueries:
    """Additional test to explicitly demonstrate ambiguous query handling."""
    
    @pytest.fixture
    def llm_client(self):
        # Force reload config from .env to avoid mock keys from other tests
        from app.utils.config import reload_config
        reload_config()
        return OpenAIClient()
    
    def test_ambiguous_query_handling(self, llm_client):
        """
        Demonstrate how the system handles ambiguous queries.
        Some may trigger clarification, others resolved via context.
        """
        session = SessionManager(llm_client=llm_client)
        pipeline = QueryPipeline(session, llm_client)
        
        print(f"\n{'='*60}")
        print("AMBIGUOUS QUERY HANDLING TEST")
        print(f"{'='*60}")
        
        # Establish context
        result = pipeline.process_and_record("I'm learning Python programming")
        print(f"Context: I'm learning Python programming")
        
        result = pipeline.process_and_record("I want to build a web app with Flask")
        print(f"Context: I want to build a web app with Flask")
        
        # Ambiguous queries
        ambiguous_queries = [
            ("Tell me more about it", "Pronoun reference"),
            ("What are the alternatives?", "Implicit context"),
            ("How do I install that?", "Pronoun reference"),
            ("Is it better than the other one?", "Comparison without clear subjects"),
            ("Show me an example", "Missing specificity"),
        ]
        
        for query, ambiguity_type in ambiguous_queries:
            result = pipeline.process_and_record(query)
            status = "CLARIFICATION" if result.needs_clarification else "RESOLVED"
            print(f"\n  Query: '{query}'")
            print(f"  Type: {ambiguity_type}")
            print(f"  Result: {status}")
            if result.needs_clarification:
                print(f"  Clarification: {result.response[:100]}...")
        
        session.save()
        print(f"\n📁 Saved: data/sessions/{session.session_id}.json")
