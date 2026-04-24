# GM_RAG — Internal Company Knowledge Agent

A **fully local** Retrieval-Augmented Generation assistant that lets your employees chat with your
company's code + docs from a browser on the internal network. No data leaves your machines.

- **LLM**: [Ollama](https://ollama.com) (default `llama3.1:8b`)
- **Embeddings**: Ollama `nomic-embed-text`
- **Vector DB**: ChromaDB (persistent, on disk)
- **Agent**: LangGraph ReAct-style with a `search_repo` tool
- **API + UI**: FastAPI + a minimal HTML chat page

---

## 1. Prerequisites

1. **Python 3.11+**
2. **Ollama** installed and running:
   ```bash
   # macOS
   brew install ollama
   ollama serve          # keep running in a terminal
   ```
3. Pull the models:
   ```bash
   ollama pull llama3.1:8b
   ollama pull nomic-embed-text
   ```
   > For code-heavy repos, `qwen2.5-coder:7b` is a strong alternative. Update `config.yaml`.

## 2. Install

```bash
cd /Users/varunteja/Desktop/GM_RAG
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 3. Point it at your repositories

Edit [config.yaml](config.yaml) — specifically the `sources:` list. Put absolute paths to each repo
or docs folder you want indexed. You can add as many as you like.

```yaml
sources:
  - /Users/you/work/backend
  - /Users/you/work/frontend
  - /Users/you/work/company-docs
```

## 4. Build the index

```bash
make ingest
# or:  python -m src.ingest
```

This walks each source, respects `.gitignore`, skips binaries / large files / ignored directories,
chunks with language-aware splitters, embeds with Ollama, and stores everything in
`storage/chroma/`. Re-run any time you want to refresh — chunk IDs are content-hashed so it upserts
cleanly.

## 5. Run the server

```bash
make serve
# or:  python -m src.server
```

Open **http://localhost:8000** on your machine.

Other employees on the same LAN can open **http://YOUR-LAN-IP:8000** (the server binds to
`0.0.0.0` by default; change in `config.yaml` if you want localhost-only).

## 6. API

- `GET  /health`  → model + chunk count
- `POST /chat`    → `{ "message": "...", "session_id": "abc" }`
- `POST /search`  → `{ "query": "...", "k": 6 }` (raw retrieval, no LLM)
- `POST /reset?session_id=abc` → clear a conversation

---

## Project layout

```
GM_RAG/
├── config.yaml              # all tunables
├── requirements.txt
├── Makefile
├── src/
│   ├── config.py            # config loader
│   ├── loaders.py           # text / pdf / docx readers
│   ├── chunking.py          # language-aware splitters
│   ├── vector_store.py      # Chroma + Ollama embeddings
│   ├── ingest.py            # walk → chunk → embed → upsert
│   ├── retriever.py         # similarity search
│   ├── agent.py             # LangGraph agent + search_repo tool
│   └── server.py            # FastAPI app
└── static/
    └── index.html           # chat UI
```

---

## Deployment notes (for delivering to the company)

1. **Host**: a single Linux box with ~16 GB RAM is enough for `llama3.1:8b` on CPU, faster with GPU.
2. Run `ollama serve` as a systemd service.
3. Run the FastAPI app behind `uvicorn` (or `gunicorn -k uvicorn.workers.UvicornWorker`) as another
   systemd service.
4. Put **nginx** in front for TLS + basic auth if you want browser-level access control.
5. Schedule a nightly `cron` job to re-run `python -m src.ingest` so new commits get indexed.
6. For multi-repo orgs, add each repo path to `config.yaml:sources`.

## Upgrade path (after MVP is accepted)

- Add a **reranker** (e.g. `bge-reranker-base` via a small local service) before passing chunks to
  the LLM.
- Add **BM25 hybrid search** (Chroma + rank_bm25) for exact-symbol queries.
- Add more tools to the agent: `read_file(path)`, `list_files(dir)`, `run_tests`, `sql_query`.
- Swap the in-memory session store for Redis.
- Add SSO (Okta / Google Workspace) via an auth proxy.
