"""LangGraph ReAct-style agent with a code/document search tool, powered by Ollama."""
from __future__ import annotations

from typing import Annotated, List, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from .config import Config, load_config
from .retriever import build_context, search


SYSTEM_PROMPT = """You are the internal company knowledge assistant for this engineering organisation.
You answer questions about the company's code, documentation, and configuration that live in the
indexed repository.

RULES:
1. Prefer calling the `search_repo` tool before answering any question about code, files, features,
   configuration, processes, or architecture. Do not guess.
2. If the retrieved context does not contain the answer, say so plainly. Never invent file paths,
   function names, or behavior.
3. Always cite sources as `path/to/file` from the retrieved chunks when you use them.
4. Keep answers concise and technical. Show short code snippets when relevant.
5. Ask a clarifying question only if the request is genuinely ambiguous.
"""


def _build_tools(cfg: Config):
    @tool
    def search_repo(query: str, k: int = 6) -> str:
        """Search the company's indexed repository for code and documentation relevant to the query.
        Returns the top matching chunks with their file paths. Use this whenever the user asks
        about internal code, features, configuration, or documentation."""
        chunks = search(query, k=k, cfg=cfg)
        if not chunks:
            return "NO_RESULTS: the vector index is empty or no chunks matched."
        return build_context(chunks, cfg.retrieval.max_context_chars)

    return [search_repo]


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


def build_agent(cfg: Config | None = None):
    cfg = cfg or load_config()
    tools = _build_tools(cfg)
    tools_by_name = {t.name: t for t in tools}

    llm = ChatOllama(
        model=cfg.ollama.chat_model,
        base_url=cfg.ollama.base_url,
        temperature=0.1,
    ).bind_tools(tools)

    def call_model(state: AgentState):
        msgs = state["messages"]
        if not msgs or not isinstance(msgs[0], SystemMessage):
            msgs = [SystemMessage(content=SYSTEM_PROMPT)] + list(msgs)
        response = llm.invoke(msgs)
        return {"messages": [response]}

    def call_tools(state: AgentState):
        last = state["messages"][-1]
        outputs: List[BaseMessage] = []
        for call in getattr(last, "tool_calls", []) or []:
            name = call["name"]
            args = call.get("args", {}) or {}
            tool_impl = tools_by_name.get(name)
            if tool_impl is None:
                result = f"ERROR: unknown tool '{name}'"
            else:
                try:
                    result = tool_impl.invoke(args)
                except Exception as e:  # tool failures must not crash the graph
                    result = f"ERROR running {name}: {e}"
            outputs.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
        return {"messages": outputs}

    def should_continue(state: AgentState) -> str:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
            return "tools"
        return END

    g = StateGraph(AgentState)
    g.add_node("model", call_model)
    g.add_node("tools", call_tools)
    g.set_entry_point("model")
    g.add_conditional_edges("model", should_continue, {"tools": "tools", END: END})
    g.add_edge("tools", "model")
    return g.compile()


def answer(question: str, history: List[BaseMessage] | None = None, cfg: Config | None = None) -> str:
    cfg = cfg or load_config()
    graph = build_agent(cfg)
    msgs: List[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT)]
    if history:
        msgs.extend(history)
    msgs.append(HumanMessage(content=question))
    result = graph.invoke({"messages": msgs})
    final = result["messages"][-1]
    return final.content if hasattr(final, "content") else str(final)
