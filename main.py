"""
ChatTMT - Chat Assistant with Session Memory

Usage: python main.py

Commands: /exit, /summary, /save, /clear
"""

import sys
from pathlib import Path

from app.core.session import SessionManager
from app.core.pipeline import QueryPipeline
from app.llms.openai_client import OpenAIClient
from app.utils.logger import get_logger
from app.utils.config import get_config

logger = get_logger(__name__)
config = get_config()


def print_banner():
    print("\n" + "=" * 70)
    print("  ChatTMT - Conversational Assistant with Session Memory")
    print("=" * 70)
    print("\nCommands: /exit, /summary, /save, /clear")
    print("Type your message and press Enter to chat.\n")


def print_summary(session: SessionManager):
    if not session.summary:
        print("\n[No summary yet - conversation too short]\n")
        return
    
    print("\n" + "-" * 70)
    print("CONVERSATION SUMMARY")
    print("-" * 70)
    
    summary = session.summary
    
    if summary.topics:
        print(f"\nTopics: {', '.join(summary.topics)}")
    
    if summary.key_facts:
        print(f"\nKey Facts:")
        for fact in summary.key_facts[:5]:
            print(f"   - {fact}")
    
    if summary.decisions:
        print(f"\nDecisions:")
        for decision in summary.decisions[:5]:
            print(f"   - {decision}")
    
    if summary.current_goal:
        print(f"\nCurrent Goal: {summary.current_goal}")
    
    print("\n" + "-" * 70 + "\n")


def run_interactive_chat():
    print_banner()
    
    try:
        llm = OpenAIClient()
        session = SessionManager(llm_client=llm)
        pipeline = QueryPipeline(session_manager=session, llm_client=llm)
        
        print("Session initialized")
        print(f"Session ID: {session.session_id}\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['/exit', '/quit']:
                    print("\nSaving session...")
                    session.save()
                    print(f"Session saved: {session.session_id}.json")
                    print("\nGoodbye!\n")
                    break
                
                elif user_input.lower() == '/summary':
                    print_summary(session)
                    continue
                
                elif user_input.lower() == '/save':
                    session.save()
                    session_path = Path(config.SESSIONS_DIR) / f"{session.session_id}.json"
                    print(f"\nSession saved: {session_path}\n")
                    continue
                
                elif user_input.lower() == '/clear':
                    print("\nStarting new conversation...")
                    session = SessionManager(llm_client=llm)
                    pipeline = QueryPipeline(session_manager=session, llm_client=llm)
                    print(f"New session ID: {session.session_id}\n")
                    continue
                
                print("Bot: ", end="", flush=True)
                result = pipeline.process_and_record(user_input)
                print(result.response)
                print()
                
            except KeyboardInterrupt:
                print("\n\nInterrupted. Saving session...")
                session.save()
                print(f"Session saved: {session.session_id}.json")
                print("\nGoodbye!\n")
                break
            
            except Exception as e:
                logger.error(f"Error during chat: {e}", exc_info=True)
                print(f"\nError: {e}\n")
                print("You can continue chatting or type /exit to quit.\n")
    
    except Exception as e:
        logger.error(f"Failed to initialize: {e}", exc_info=True)
        print(f"\nFailed to initialize: {e}")
        print("\nPlease check:")
        print("  1. OPENAI_API_KEY is set in .env file")
        print("  2. You have internet connection")
        print("  3. Check logs/app.log for details\n")
        sys.exit(1)


if __name__ == "__main__":
    run_interactive_chat()
