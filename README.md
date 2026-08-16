# Multi-Source Fact Checker

A multi-source fact-checking agent built for the LEC AI Engineering Intern build assessment.

## How to Run

### 1. Install backend dependencies

```bash
pip install -r requirements.txt
```

### 2. Create a `.env` file

Create a `.env` file in the backend root directory:

```env
OPENAI_API_KEY=your_openai_api_key
```

### 3. Start the backend

```bash
uvicorn main:app --reload
```

The backend will run at:

```text
http://localhost:8000
```

### 4. Run the frontend

Open a new terminal:

```bash
cd frontend
npm install
npm run dev
```

Then open the URL shown in the terminal.

## Architecture

```text
User Question
      |
      v
   Planner
      |
      v
+-------------+
| Local RAG   |
| Wikipedia   |
+-------------+
      |
      v
Evidence Analyzer
      |
      v
Final Answer
```

## What I Would Do Next

With more time, I would:

* Add more independent data sources.
* Improve source reliability and ranking.
* Add automated evaluation and test cases for conflicting sources.
* Add better logging and monitoring.
* Add caching and rate limiting for production use.
* Improve the RAG retrieval and evaluation pipeline.
