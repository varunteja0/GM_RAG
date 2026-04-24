"""CLI chat client for quick testing without the web UI.

Usage:  python -m scripts.cli_chat
"""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.agent import SYSTEM_PROMPT, build_agent
from src.config import load_config


def main() -> None:
    cfg = load_config()
    graph = build_agent(cfg)
    history = [SystemMessage(content=SYSTEM_PROMPT)]

    print(f"GM_RAG CLI · model={cfg.ollama.chat_model}  (type 'exit' to quit)\n")
    while True:
        try:
            q = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            break

        history.append(HumanMessage(content=q))
        result = graph.invoke({"messages": list(history)})
        final = result["messages"][-1]
        answer = final.content if hasattr(final, "content") else str(final)
        history.append(AIMessage(content=answer))
        print(f"\nagent> {answer}\n")


if __name__ == "__main__":
    main()
