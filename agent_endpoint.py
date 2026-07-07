import os
import httpx
from typing import TypedDict, Optional
from runpod_flash import Endpoint

# LLM endpoint the agent calls. Defaults to the local Flash dev route;
# override with LLM_ENDPOINT_URL to point at a deployed RunPod endpoint.
LLM_ENDPOINT_URL = os.getenv(
    "LLM_ENDPOINT_URL", "http://localhost:8888/llm_endpoint/runsync"
)
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY", "")

# ---- LangGraph state ----
class AgentState(TypedDict):
    question: str
    needs_search: bool
    search_result: Optional[str]
    answer: Optional[str]

# ---- Node 1: Classify ----
async def classify_node(state: AgentState) -> AgentState:
    """Decide if the question needs fresh web data."""
    keywords = ["today", "latest", "current", "now", "recently", "2025", "2026"]
    needs = any(k in state["question"].lower() for k in keywords)
    return {**state, "needs_search": needs}

# ---- Node 2: Tool / Web fetch ----
async def search_node(state: AgentState) -> AgentState:
    if not state["needs_search"]:
        return {**state, "search_result": None}
    
    async with httpx.AsyncClient() as client:
        # Using DuckDuckGo instant answer API as a simple fetch
        resp = await client.get(
            "https://api.duckduckgo.com/",
            params={"q": state["question"], "format": "json", "no_html": 1},
            timeout=10
        )
        data = resp.json()
        snippet = data.get("AbstractText") or data.get("Answer") or "No result found."
        return {**state, "search_result": snippet}

# ---- Node 3: LLM call ----
async def llm_node(state: AgentState) -> AgentState:
    context = ""
    if state.get("search_result"):
        context = f"\n\nAdditional context from web: {state['search_result']}"
    
    prompt = f"""Answer this question clearly and concisely.
Question: {state['question']}{context}
Answer:"""
    
    # Call the LLM endpoint (local Flash dev route by default; RunPod GPU when deployed)
    headers = {}
    if RUNPOD_API_KEY:
        headers["Authorization"] = f"Bearer {RUNPOD_API_KEY}"

    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            LLM_ENDPOINT_URL,
            json={"input": {"prompt": prompt}},
            headers=headers,
        )
        resp.raise_for_status()
        result = resp.json()
        # RunPod runsync returns {"output": ...}; be tolerant of a raw string too
        answer = result.get("output", result) if isinstance(result, dict) else result

    return {**state, "answer": answer}

# ---- Build graph ----
def build_graph():
    from langgraph.graph import StateGraph, END

    graph = StateGraph(AgentState)
    graph.add_node("classify", classify_node)
    graph.add_node("search", search_node)
    graph.add_node("answer", llm_node)

    graph.set_entry_point("classify")
    graph.add_edge("classify", "search")
    graph.add_edge("search", "answer")
    graph.add_edge("answer", END)

    return graph.compile()

# ---- Flash endpoint (CPU worker) ----
@Endpoint(
    name="agent",
    dependencies=["langgraph", "langchain", "httpx"]
    # No gpu= means it runs on CPU — much cheaper
)
async def agent(question: str) -> dict:
    graph = build_graph()
    result = await graph.ainvoke({
        "question": question,
        "needs_search": False,
        "search_result": None,
        "answer": None
    })
    return {
        "question": result["question"],
        "used_search": result["needs_search"],
        "search_result": result.get("search_result"),
        "answer": result["answer"]
    }
