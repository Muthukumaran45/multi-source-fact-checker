from typing import TypedDict

class AgentState(TypedDict):
    question: str
    selected_sources: list[str]
    reasoning: str
    source_results: dict
    analysis: dict
    final_answer: str
    simulate_wiki_failure: bool
    simulate_conflict: bool

