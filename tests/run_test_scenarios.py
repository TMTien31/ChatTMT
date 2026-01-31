#!/usr/bin/env python3
import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.core.session import SessionManager
from app.core.pipeline import QueryPipeline
from app.llms.openai_client import OpenAIClient
from app.utils.logger import get_logger

logger = get_logger(__name__)


def load_scenario(filepath: str):
    messages = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if isinstance(data, dict) and data.get('role') == 'user':
                    messages.append(data)
            except json.JSONDecodeError:
                continue
    return messages


def run_scenario(scenario_file: str, verbose: bool = True):
    print(f"\n{'='*80}")
    print(f"Running Scenario: {Path(scenario_file).name}")
    print(f"{'='*80}\n")
    
    messages = load_scenario(scenario_file)
    print(f"Loaded {len(messages)} user messages\n")
    
    llm = OpenAIClient()
    session = SessionManager(llm_client=llm)
    pipeline = QueryPipeline(session_manager=session, llm_client=llm)
    
    for i, msg in enumerate(messages, 1):
        user_query = msg['content']
        turn_num = msg.get('turn', i)
        
        if verbose:
            print(f"\n--- Turn {turn_num} ---")
            print(f"User: {user_query}")
        
        try:
            result = pipeline.process_and_record(user_query)
            
            if verbose:
                print(f"Bot: {result.response}")
                print(f"\nPipeline Details:")
                print(f"  - Rewritten: {'Yes' if result.rewrite_result.rewritten_query else 'No'}")
                if result.rewrite_result.rewritten_query:
                    print(f"  - Rewritten query: {result.rewrite_result.rewritten_query}")
                print(f"  - Ambiguous: {result.rewrite_result.is_ambiguous}")
                print(f"  - Context used: {result.augmented_context.memory_fields_used}")
                if result.clarification_result:
                    print(f"  - Clarification needed: {result.clarification_result.needs_clarification}")
                
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    
    session.save()
    print(f"\n{'='*80}")
    print(f"Scenario completed")
    print(f"Session ID: {session.session_id}")
    print(f"Total turns: {session.total_turns}")
    print(f"Session saved to: data/sessions/{session.session_id}.json")
    
    if session.summary:
        print(f"\nSession Summary:")
        if session.summary.topics:
            print(f"  Topics: {', '.join(session.summary.topics[:5])}")
        if session.summary.key_facts:
            print(f"  Key Facts: {len(session.summary.key_facts)} extracted")
        if session.summary.decisions:
            print(f"  Decisions: {len(session.summary.decisions)} recorded")
        if session.summary.current_goal:
            print(f"  Current Goal: {session.summary.current_goal}")
    
    print(f"{'='*80}\n")
    return session


def run_all_scenarios(test_dir: str = "data/test_conversations"):
    test_path = Path(test_dir)
    scenario_files = sorted(test_path.glob("scenario_*.jsonl"))
    
    if not scenario_files:
        print(f"No scenario files found in {test_dir}")
        return
    
    print(f"\nFound {len(scenario_files)} scenarios\n")
    
    for scenario_file in scenario_files:
        run_scenario(str(scenario_file), verbose=True)
        print("\nPress Enter to continue (or Ctrl+C to stop)...")
        try:
            input()
        except KeyboardInterrupt:
            print("\nStopped by user")
            break


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run test conversation scenarios")
    parser.add_argument(
        "scenario",
        nargs="?",
        help="Specific scenario file to run (e.g., 'scenario_1_summarization.jsonl'). If not provided, runs all scenarios."
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity"
    )
    
    args = parser.parse_args()
    
    if args.scenario:
        # Run specific scenario
        if not args.scenario.startswith("data/"):
            scenario_path = f"data/test_conversations/{args.scenario}"
        else:
            scenario_path = args.scenario
        
        if not Path(scenario_path).exists():
            print(f"Error: Scenario file not found: {scenario_path}")
            sys.exit(1)
        
        run_scenario(scenario_path, verbose=not args.quiet)
    else:
        # Run all scenarios
        run_all_scenarios()
