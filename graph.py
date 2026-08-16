from state import AgentState
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END

from pydantic import BaseModel, Field

from rag import retrieve_documents
from wikipedia_source import search_wikipedia


load_dotenv()


# STRUCTURED OUTPUTS

class PlannerDecision(BaseModel):
    selected_sources: list[str] = Field(
        description="Sources to query. Must contain only local_rag or wikipedia."
    )
    reasoning: str = Field(
        description="Why these sources were selected."
    )


class EvidenceAnalysis(BaseModel):
    verdict: str = Field(
        description="One of: agreement, conflict, insufficient"
    )
    confidence: str = Field(
        description="One of: high, medium, low"
    )
    answer: str = Field(
        description="The answer supported by the available evidence."
    )
    explanation: str = Field(
        description="Explain how the evidence supports the verdict."
    )


# LLM

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

planner_llm = llm.with_structured_output(
    PlannerDecision
)

analyzer_llm = llm.with_structured_output(
    EvidenceAnalysis
)


# PLANNER

def planner(state: AgentState):

    question = state["question"]

    prompt = f"""
You are a fact-checking planner.

Question:
{question}

Available sources:

1. local_rag
   - Contains our internal knowledge base.

2. wikipedia
   - External public knowledge source.

Choose the sources that are useful for answering the question.

Use both sources when independent verification would be useful.

Only use these source names:
- local_rag
- wikipedia
"""

    decision = planner_llm.invoke(prompt)

    print("\n===== PLANNER =====")
    print("Selected sources:", decision.selected_sources)
    print("Reason:", decision.reasoning)

    return {
        "selected_sources": decision.selected_sources,
        "reasoning": decision.reasoning
    }


# EXECUTE SOURCES
# IN PARALLEL

def execute_sources(state: AgentState):

    question = state["question"]
    selected_sources = state["selected_sources"]
    simulate_wiki_failure = state.get("simulate_wiki_failure", False)
    simulate_conflict = state.get("simulate_conflict", False)

    results = {}

    tasks = {}

    print("\n===== EXECUTING SOURCES =====")
    print("Running selected sources in parallel...")

    with ThreadPoolExecutor(max_workers=2) as executor:

        if "local_rag" in selected_sources:

            tasks["local_rag"] = executor.submit(
                retrieve_documents,
                question,
                simulate_conflict,
            )

        if "wikipedia" in selected_sources:

            tasks["wikipedia"] = executor.submit(
                search_wikipedia,
                question,
                simulate_wiki_failure,
            )

        for source, future in tasks.items():

            try:

                results[source] = future.result()

            except Exception as e:

                results[source] = {
                    "source": source,
                    "status": "error",
                    "content": None,
                    "error": str(e)
                }


    print("\n===== SOURCE RESULTS =====")

    for source, result in results.items():

        print(f"\n{source}:")
        print(result)


    return {
        "source_results": results
    }


# EVIDENCE ANALYZER

def analyze_evidence(state: AgentState):

    question = state["question"]
    source_results = state["source_results"]


    # Detect failed sources

    failed_sources = [
        source
        for source, result in source_results.items()
        if result.get("status") != "success"
    ]


    # Detect successful sources

    successful_sources = [
        source
        for source, result in source_results.items()
        if result.get("status") == "success"
    ]


    # Degradation information

    degradation_note = ""

    if failed_sources:

        degradation_note = f"""
IMPORTANT DEGRADATION CONDITION:

The following sources failed:

{failed_sources}

The following sources succeeded:

{successful_sources}

Do NOT claim that the sources agree.

Do NOT give high confidence.

If the successful source contains evidence for an answer,
you may provide that answer, but explicitly state that
independent verification was unavailable.

Use:

verdict = insufficient

confidence = medium or low
"""


    # Analyzer prompt

    prompt = f"""
You are a strict fact-checking evidence analyzer.

Question:
{question}

Evidence:
{source_results}

{degradation_note}

Rules:

1. Use ONLY the evidence provided.

2. Never use your own world knowledge.

3. Never treat an unavailable source as supporting evidence.

4. If a source failed, do not claim that source agrees.

5. If independent sources support the same conclusion:
   verdict = agreement

6. If sources contradict each other:
   verdict = conflict
   confidence = low

   Do NOT choose one source as correct.

   The answer must clearly state that the result
   cannot be determined with confidence.

7. If independent verification is unavailable
   or evidence is insufficient:
   verdict = insufficient

8. Never invent missing information.

9. Do not claim that a source confirms something
   unless that information explicitly appears
   in its content.

10. Confidence must reflect the actual available evidence.

11. If sources conflict, explain exactly what
    each source claims.

12. If ALL sources failed (no successful sources at all):
    verdict = insufficient
    confidence = low
    answer must explicitly state that no source could be reached
    and no answer can be given, rather than guessing.
"""


    # Run analyzer

    result = analyzer_llm.invoke(prompt)


    print("\n===== EVIDENCE ANALYSIS =====")

    print("Verdict:", result.verdict)

    print("Confidence:", result.confidence)

    print("Answer:", result.answer)

    print("Explanation:", result.explanation)


    return {
        "analysis": result.model_dump()
    }


# PRESENT ANSWER

def present_answer(state: AgentState):

    analysis = state["analysis"]
    source_results = state["source_results"]

    source_lines = []
    for source, result in source_results.items():
        status = result.get("status")
        source_lines.append(f"  - {source}: {status}")

    summary = (
        f"Verdict: {analysis['verdict']}\n"
        f"Confidence: {analysis['confidence']}\n"
        f"Answer: {analysis['answer']}\n"
        f"Why: {analysis['explanation']}\n"
        f"Sources consulted:\n" + "\n".join(source_lines)
    )

    print("\n===== FINAL ANSWER =====")
    print(summary)

    return {
        "final_answer": summary
    }


# BUILD GRAPH
graph_builder = StateGraph(AgentState)

graph_builder.add_node("planner", planner)
graph_builder.add_node("execute_sources", execute_sources)
graph_builder.add_node("analyze_evidence", analyze_evidence)
graph_builder.add_node("present_answer", present_answer)


# GRAPH EDGES
graph_builder.add_edge(START, "planner")
graph_builder.add_edge("planner", "execute_sources")
graph_builder.add_edge("execute_sources", "analyze_evidence")
graph_builder.add_edge("analyze_evidence", "present_answer")
graph_builder.add_edge("present_answer", END)


# COMPILE
graph = graph_builder.compile()


def run(
    question: str,
    simulate_wiki_failure: bool = False,
    simulate_conflict: bool = False,
):
    """Convenience entry point used by both the CLI test below and server.py."""

    return graph.invoke({
        "question": question,
        "selected_sources": [],
        "reasoning": "",
        "source_results": {},
        "analysis": {},
        "final_answer": "",
        "simulate_wiki_failure": simulate_wiki_failure,
        "simulate_conflict": simulate_conflict,
    })

