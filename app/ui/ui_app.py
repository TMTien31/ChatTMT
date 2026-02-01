import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
from datetime import datetime
import json
from typing import List, Optional

from app.core.session import SessionManager
from app.core.pipeline import QueryPipeline
from app.llms.openai_client import OpenAIClient
from app.utils.config import get_config

config = get_config()


def list_saved_sessions(force_refresh: bool = False) -> List[tuple[str, dict]]:
    sessions_dir = Path(config.SESSION_DATA_DIR)
    if not sessions_dir.exists():
        return []
    
    sessions = []
    for file in sessions_dir.glob("*.json"):
        try:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                metadata = {
                    'session_id': data.get('session_id'),
                    'created_at': data.get('created_at'),
                    'last_activity': data.get('last_activity'),
                    'total_turns': data.get('total_turns', 0),
                    'file_path': str(file),
                    'modified_time': file.stat().st_mtime
                }
                sessions.append((file.stem, metadata))
        except Exception:
            continue
    
    # Sort by modified time, newest first
    sessions.sort(key=lambda x: x[1]['modified_time'], reverse=True)
    return sessions


def format_timestamp(iso_string: Optional[str]) -> str:
    if not iso_string:
        return "Unknown"
    
    try:
        dt = datetime.fromisoformat(iso_string)
        now = datetime.now()
        diff = now - dt
        
        if diff.days == 0:
            hours = diff.seconds // 3600
            minutes = (diff.seconds % 3600) // 60
            if hours > 0:
                return f"{hours}h {minutes}m ago"
            elif minutes > 0:
                return f"{minutes}m ago"
            else:
                return "Just now"
        elif diff.days == 1:
            return "Yesterday"
        elif diff.days < 7:
            return f"{diff.days} days ago"
        else:
            return dt.strftime("%b %d, %Y")
    except Exception:
        return "Unknown"


def format_session_name(session_id: str, metadata: dict) -> str:
    turns = metadata.get('total_turns', 0)
    time_str = format_timestamp(metadata.get('last_activity'))
    short_id = session_id[:8]
    return f"{short_id} • {turns} turns • {time_str}"


def initialize_session_state():
    if 'llm_client' not in st.session_state:
        try:
            st.session_state.llm_client = OpenAIClient()
            from app.core.schemas import LLMMessage
            st.session_state.llm_client.chat([LLMMessage(role="user", content="Hi")], max_tokens=5)
        except Exception as e:
            st.error(f"Failed to initialize OpenAI client: {str(e)}")
            st.error("Please check your OPENAI_API_KEY in .env file")
            st.stop()
    
    if 'session_manager' not in st.session_state:
        st.session_state.session_manager = SessionManager(
            llm_client=st.session_state.llm_client
        )
    
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = QueryPipeline(
            session_manager=st.session_state.session_manager,
            llm_client=st.session_state.llm_client
        )
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
        for msg in st.session_state.session_manager.raw_messages:
            st.session_state.messages.append({
                'role': msg.role,
                'content': msg.content
            })


def load_session(session_id: str):
    try:
        session = SessionManager.load(session_id, llm_client=st.session_state.llm_client)
        
        st.session_state.session_manager = session
        st.session_state.pipeline = QueryPipeline(
            session_manager=session,
            llm_client=st.session_state.llm_client
        )
        
        st.session_state.messages = []
        for msg in session.raw_messages:
            st.session_state.messages.append({
                'role': msg.role,
                'content': msg.content
            })
        
        st.success(f"Loaded session: {session_id[:8]}")
        st.rerun()
    except Exception as e:
        st.error(f"Failed to load session: {e}")


def create_new_session():
    st.session_state.session_manager = SessionManager(
        llm_client=st.session_state.llm_client
    )
    st.session_state.pipeline = QueryPipeline(
        session_manager=st.session_state.session_manager,
        llm_client=st.session_state.llm_client
    )
    st.session_state.messages = []
    
    st.success("Created new session")
    st.rerun()


def main():
    st.set_page_config(
        page_title="ChatTMT",
        page_icon="💬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    initialize_session_state()
    
    with st.sidebar:
        st.title("ChatTMT")
        st.markdown("---")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button("New Chat", use_container_width=True, type="primary"):
                create_new_session()
        with col2:
            if st.button("🔄", help="Refresh session list"):
                st.rerun()
        
        st.markdown("---")
        
        st.subheader("Current Session")
        session = st.session_state.session_manager
        st.text(f"ID: {session.session_id[:8]}...")
        st.text(f"Turns: {session.total_turns}")
        st.text(f"Created: {format_timestamp(session.state.created_at.isoformat())}")
        
        if session.summary and (session.summary.topics or session.summary.key_facts):
            with st.expander("Summary", expanded=False):
                if session.summary.topics:
                    st.caption("Topics:")
                    for topic in session.summary.topics[:3]:
                        st.markdown(f"- {topic}")
                if session.summary.key_facts:
                    st.caption("Key Facts:")
                    for fact in session.summary.key_facts[:3]:
                        st.markdown(f"- {fact}")
        
        if st.button("Save Session", use_container_width=True):
            session.save()
            st.success("Session saved!")
        
        st.markdown("---")
        
        with st.expander("Debug Panel", expanded=False):
            st.subheader("Session State")
            st.json({
                "session_id": session.session_id,
                "total_turns": session.total_turns,
                "clarification_count": session.clarification_count,
                "summarized_up_to_turn": session.summarized_up_to_turn,
                "raw_messages_count": len(session.raw_messages),
            })
            
            if session.summary:
                st.subheader("Summary Schema")
                summary_dict = {
                    "topics": session.summary.topics,
                    "key_facts": session.summary.key_facts,
                    "decisions": session.summary.decisions,
                    "current_goal": session.summary.current_goal,
                    "open_questions": session.summary.open_questions,
                    "todos": session.summary.todos,
                    "user_profile": {
                        "background": session.summary.user_profile.background,
                        "preferences": session.summary.user_profile.preferences,
                        "expertise": session.summary.user_profile.expertise,
                    } if session.summary.user_profile else None
                }
                st.json(summary_dict)
            else:
                st.info("No summary yet (conversation too short)")
        
        st.markdown("---")
        
        st.subheader("Previous Sessions")
        saved_sessions = list_saved_sessions()
        
        if saved_sessions:
            current_id = session.session_id
            options = []
            session_map = {}
            
            for sess_id, metadata in saved_sessions:
                if sess_id == current_id:
                    continue
                
                display_name = format_session_name(sess_id, metadata)
                options.append(display_name)
                session_map[display_name] = sess_id
            
            if options:
                selected = st.selectbox(
                    "Load session:",
                    options=["Select..."] + options,
                    key="session_selector"
                )
                
                if selected != "Select...":
                    if st.button("Load", use_container_width=True):
                        load_session(session_map[selected])
            else:
                st.info("Only current session exists")
        else:
            st.info("No saved sessions yet")
        
        st.markdown("---")
        st.caption("Built with OpenAI & Streamlit")
    
    st.title("Chat with ChatTMT")
    
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    if prompt := st.chat_input("Type your message..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = st.session_state.pipeline.process_and_record(prompt)
                    response = result.response
                    
                    st.markdown(response)
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response
                    })
                    
                    st.session_state.session_manager.save()
                    
                except json.JSONDecodeError as e:
                    error_msg = f"JSON parsing error: {str(e)}. Please try again."
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })
                except (IOError, OSError) as e:
                    error_msg = f"File system error: {str(e)}. Check session directory permissions."
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })
                except Exception as e:
                    error_msg = f"Unexpected error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })


if __name__ == "__main__":
    main()