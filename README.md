# Multi-Source Fact Checker

A multi-source fact-checking agent built for the LEC AI Engineering Intern build assessment.

The system answers factual questions by consulting multiple independent sources, comparing the returned evidence, and gracefully handling source failures, timeouts, and conflicting information.

## Overview

The agent uses:

- **LangGraph** for agent orchestration
- **OpenAI GPT-4o-mini** for planning and evidence analysis
- **Local RAG** using OpenAI embeddings + FAISS
- **Wikipedia API** as an independent external source
- **FastAPI** for the backend API
- **React** for the frontend

The main design goal is **graceful degradation**.

The agent should not:

- silently guess
- pretend an unavailable source provided evidence
- crash when one source fails
- choose one source arbitrarily when sources conflict

Instead, it reports the available evidence and adjusts its confidence accordingly.

---

## Architecture

```text
                    User Question
                         |
                         v
                    +---------+
                    | Planner |
                    +---------+
                         |
              Select relevant sources
                         |
                         v
              +---------------------+
              | Execute in Parallel |
              +---------------------+
                  /             \
                 /               \
                v                 v
        +-------------+    +-------------+
        |  Local RAG  |    |  Wikipedia  |
        |    FAISS    |    |     API     |
        +-------------+    +-------------+
                \                 /
                 \               /
                  v             v
              +-------------------+
              | Evidence Analyzer |
              +-------------------+
                       |
                       v
              Agreement / Conflict
                 / Insufficient
                       |
                       v
                Final Answer