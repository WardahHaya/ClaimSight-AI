# ClaimSight AI

ClaimSight AI is a locally runnable full-stack app for auto-insurance claims triage. It accepts a text question and an optional vehicle-damage photo, analyzes the image with a multimodal LLM, rewrites the query with conversation context, decides whether retrieval is needed, runs a real Chroma-backed RAG flow over public insurance documents, and stores multi-turn chat history in SQLite by `session_id`.

## What Is In This Repo

- `backend/`
  FastAPI API, LangGraph orchestration, Chroma vector store, SQLite conversation memory, ingestion CLI, and graph test harness.
- `data/`
  A fetch manifest of real public insurance and claims sources plus a fetch script that downloads them into `data/raw/`.
- `frontend/`
  React + Vite + TypeScript chat UI with image upload, inline image previews, execution-trace badges, and multi-session local state.

## Flowchart To Code Mapping

The implemented graph mirrors the requested pipeline node for node:

1. `User Query (text + optional image)`
   API entry in `backend/app.py` `chat()`.
2. `Load Conversation History`
   `backend/app.py` calls `backend.database.get_history()`.
3. `Image Analysis`
   `backend/graph.py` `analyze_image()`.
4. `Query Rewriter`
   `backend/graph.py` `rewrite_query()`.
5. `Orchestrator`
   `backend/graph.py` `orchestrate()`.
6. `Retrieve Documents`
   `backend/graph.py` `retrieve_docs()`.
7. `Re-rank Documents (RRF)`
   `backend/graph.py` `rerank_docs()`.
8. `Document Relevance Evaluator`
   `backend/graph.py` `evaluate_relevance()`.
9. `Retry Query Rewrite`
   `backend/graph.py` `retry_rewrite_query()`.
10. `Main LLM Call - Grounded`
    `backend/graph.py` `generate_grounded_response()`.
11. `Main LLM Call - Direct`
    `backend/graph.py` `generate_direct_response()`.
12. `Safe Response`
    `backend/graph.py` `generate_safe_response()`.
13. `Save Query and Response`
    `backend/app.py` calls `backend.database.save_message()`.

The retry loop is a real LangGraph cycle:

- `evaluate -> retry_rewrite -> retrieve`

The hard retry cap is `2`.

## Real Knowledge Base

The corpus is built from real public insurance material listed in `data/sources.json`. The current manifest includes public sources from:

- NAIC consumer auto-insurance pages
- California Department of Insurance guides
- California DMV total-loss guidance
- Washington Office of the Insurance Commissioner claims guidance
- South Carolina DOI consumer pages
- State Farm public policy booklet
- Allstate public policy and coverage pages
- Travelers public coverage page

Running ingestion with the current manifest produced `17` downloaded source files and `480` indexed chunks locally during verification.

## Requirements

- Python `3.13` or newer
- Node.js `18+`
- npm
- One multimodal LLM API key

Recommended providers:

- `groq`
- `openai`
- `anthropic`

Notes:

- Groq now works for local image uploads in this app via base64 `data:image/...` URLs.
- For Groq specifically, base64 image uploads must be `4MB` or smaller.

## Setup

### 1. Create and activate a virtual environment

From the repo root:

```bash
python -m venv venv
```

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
source venv/bin/activate
```

### 2. Install backend dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Copy `backend/.env.example` to `backend/.env` and set a real API key.

Example using Groq:

```env
LLM_PROVIDER=groq
GROQ_API_KEY=your_real_key_here
GROQ_TEXT_MODEL=llama-3.3-70b-versatile
GROQ_VISION_MODEL=meta-llama/llama-4-scout-17b-16e-instruct
```

Example using OpenAI:

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=your_real_key_here
OPENAI_TEXT_MODEL=gpt-4o-mini
OPENAI_VISION_MODEL=gpt-4o
```

Example using Anthropic:

```env
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=your_real_key_here
ANTHROPIC_TEXT_MODEL=claude-3-5-sonnet-latest
ANTHROPIC_VISION_MODEL=claude-3-5-sonnet-latest
```

### 4. Fetch the real source corpus

```bash
python data/fetch_real_docs.py
```

This downloads the public source files into `data/raw/`.

### 5. Build the local Chroma vector store

From the repo root:

```bash
python backend/ingest.py --fetch
```

If you want to force a fresh re-download and rebuild:

```bash
python backend/ingest.py --fetch --force-fetch
```

### 6. Start the integrated localhost app

Single-command integrated mode:

```bash
python run_local.py
```

This will:

- build the frontend
- serve the built React app from FastAPI
- run the whole app on `http://127.0.0.1:8000`

If the frontend is already built:

```bash
python run_local.py --skip-build
```

### 7. Start only the backend

```bash
uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

Useful endpoints:

- `GET http://127.0.0.1:8000/health`
- `POST http://127.0.0.1:8000/chat`
- `GET http://127.0.0.1:8000/history/{session_id}`
- `POST http://127.0.0.1:8000/reset/{session_id}`

### 8. Frontend dev mode only

In a second terminal:

```bash
cd frontend
npm install
npm run dev -- --host 127.0.0.1 --port 5173
```

Open:

```text
http://127.0.0.1:5173
```

In Vite dev mode, API requests are proxied to the backend automatically.

## CLI Graph Harness

Before using the API, you can exercise the LangGraph pipeline directly:

```bash
python backend/test_graph.py "My bumper was damaged after hitting a guardrail. Is that collision coverage?"
```

With an image:

```bash
python backend/test_graph.py "What should I do next?" --image path/to/damage.jpg
```

## Example Session

This is a representative end-to-end flow the app now supports.

1. User uploads a front-bumper damage photo and asks:
   `My bumper got hit in a parking lot. Is this likely collision coverage and what do I do next?`

   Expected path:
   `Load Conversation History -> Vision Analysis -> Query Rewrite -> Route Decision: RAG -> Retrieve Documents -> Re-rank Documents -> Evaluate Relevance -> Generate Grounded Response`

2. Same session, follow-up text only:
   `Do I need a police report and what deductible would usually apply?`

   Expected behavior:
   The query rewriter uses prior conversation plus the saved image-analysis context from the earlier photo turn.

3. Out-of-scope or unsupported question:
   `Can you tell me whether my marine trailer claim follows these same auto physical damage rules?`

   Expected path:
   retrieval plus up to two retries, then `Generate Safe Response` if the corpus is still insufficient.

## Frontend Features

- Multi-session chat threads stored in browser local storage
- Inline image preview before send
- Inline uploaded-image thumbnail in the chat thread
- Pipeline trace panel per assistant response
- Status badges for direct, grounded, retry, and safe-response paths

## Backend State Shape

The LangGraph state carries the requested core fields:

- `original_query`
- `rewritten_query`
- `image_analysis`
- `conversation_history`
- `retrieved_docs`
- `rerank_scores`
- `relevance_verdict`
- `retry_count`
- `final_response`

Additional internal state is also tracked for routing and intermediate retrieval results.

## Persistence

- SQLite chat memory:
  `backend/chat_history.db`
- Chroma vector store:
  `backend/chroma_db/`
- Downloaded public corpus:
  `data/raw/`

## Verification Performed

The following checks were run during repair:

- Backend dependency installation on Python `3.13`
- `python backend/ingest.py --fetch --force-fetch`
- Backend import smoke test for `backend.app`, `backend.graph`, and `backend.ingest`
- Frontend production build with `npm run build`
- Localhost startup smoke test for:
  `http://127.0.0.1:8000/health`
  `http://127.0.0.1:5173`
- Integrated serving path where FastAPI returns the built frontend from `frontend/dist`

## Troubleshooting

- If `/chat` returns an API-key error, set a real provider key in `backend/.env`.
- If you change providers, restart the backend.
- If you use Groq and image upload fails, compress the image below `4MB`.
- If Vite behaves oddly because the repo path contains `#`, move the project to a directory without `#` in its name. The dev server still started successfully during verification, but Vite warns about that path character.
