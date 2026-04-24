"""FastAPI server exposing the agent + a tiny static chat UI.

Run:  python -m src.server
Then open http://localhost:8000
Other employees on the LAN can open http://<your-ip>:8000
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel

from .agent import build_agent
from .config import load_config
from .retriever import search
from .vector_store import get_or_create_collection

ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = ROOT / "static"

cfg = load_config()
app = FastAPI(title="GM_RAG - Internal Knowledge Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Build agent graph once at startup.
_agent = build_agent(cfg)

# Very simple in-memory session store (fine for internal tool; swap for Redis later).
_sessions: Dict[str, List] = {}


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"


class ChatResponse(BaseModel):
    answer: str
    session_id: str


class SearchRequest(BaseModel):
    query: str
    k: int = 6


@app.get("/health")
def health():
    col = get_or_create_collection(cfg)
    return {
        "status": "ok",
        "collection": cfg.collection,
        "indexed_chunks": col.count(),
        "chat_model": cfg.ollama.chat_model,
        "embed_model": cfg.ollama.embed_model,
    }


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="empty message")

    history = _sessions.setdefault(req.session_id, [])
    history.append(HumanMessage(content=req.message))

    try:
        result = _agent.invoke({"messages": list(history)})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"agent error: {e}")

    final = result["messages"][-1]
    answer_text = final.content if hasattr(final, "content") else str(final)
    history.append(AIMessage(content=answer_text))

    # Cap history so it doesn't grow forever.
    if len(history) > 20:
        _sessions[req.session_id] = history[-20:]

    return ChatResponse(answer=answer_text, session_id=req.session_id)


@app.post("/search")
def raw_search(req: SearchRequest):
    chunks = search(req.query, k=req.k, cfg=cfg)
    return [
        {"path": c.path, "score": round(c.score, 4), "chunk_index": c.chunk_index, "text": c.text}
        for c in chunks
    ]


@app.post("/reset")
def reset(session_id: str = "default"):
    _sessions.pop(session_id, None)
    return {"ok": True}


# Static chat UI
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/")
    def index():
        return FileResponse(str(STATIC_DIR / "index.html"))


def main():
    import uvicorn
    uvicorn.run(
        "src.server:app",
        host=cfg.server.host,
        port=cfg.server.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
