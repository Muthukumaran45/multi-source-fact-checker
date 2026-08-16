
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from graph import run

app = FastAPI(title="Multi-Source Fact Checker")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


SUGGESTED_QUESTIONS = [
    "Who is the CEO of Apple?",
    "Who founded Google?",
    "Where is Microsoft headquartered?",
    "Who is the CEO of Microsoft?",
    "What is Google's parent company called?",
]


class AskRequest(BaseModel):
    question: str
    simulate_wiki_failure: bool = False
    simulate_conflict: bool = False


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/suggested-questions")
def suggested_questions():
    return {"questions": SUGGESTED_QUESTIONS}


@app.post("/ask")
def ask(req: AskRequest):
    result = run(
        question=req.question,
        simulate_wiki_failure=req.simulate_wiki_failure,
        simulate_conflict=req.simulate_conflict,
    )

    return {
        "question": result["question"],
        "selected_sources": result["selected_sources"],
        "planner_reasoning": result["reasoning"],
        "source_results": result["source_results"],
        "analysis": result["analysis"],
        "final_answer": result["final_answer"],
    }