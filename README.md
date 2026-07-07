# LangGraph Agent on RunPod Serverless GPU

A small but production-shaped demo: a **LangGraph agent** that answers questions by calling **Llama 3.1 8B** running on an **RTX 4090**. The system uses two [RunPod Flash](https://docs.runpod.io/flash/overview) endpoints — one on CPU for orchestration, one on GPU for inference.

This mirrors how real agent systems are built: cheap CPU workers coordinate logic and tools; expensive GPU workers handle model inference only when needed.

## Architecture

```
┌─────────────────┐     POST /agent      ┌──────────────────────┐
│  Your terminal  │ ──────────────────►  │  agent (CPU worker)  │
│  or REST client │                        │  LangGraph — 3 nodes │
└─────────────────┘                        └──────────┬───────────┘
                                                      │
                              classify → search? → call LLM endpoint
                                                      │
                                                      ▼
                                           ┌──────────────────────┐
                                           │ llm-inference (4090) │
                                           │  Llama 3.1 8B (vLLM) │
                                           └──────────────────────┘
```

| Endpoint | Worker | Role |
|----------|--------|------|
| `agent` | CPU | LangGraph orchestration — classify, optional web search, call LLM |
| `llm-inference` | RTX 4090 | vLLM inference with `meta-llama/Llama-3.1-8B-Instruct` |

Workers scale from zero on demand. You pay only for the seconds they are actually running.

## Project files

| File | Description |
|------|-------------|
| `agent_endpoint.py` | CPU agent — LangGraph with three nodes |
| `llm_endpoint.py` | GPU LLM endpoint — vLLM + Llama 3.1 8B |
| `api.js` | Optional Node.js client example for calling the deployed agent |

## How it works

### 1. LLM endpoint (`llm_endpoint.py`)

A single Python function decorated with `@Endpoint`. Flash provisions an **RTX 4090**, installs `vllm` and `torch`, and runs inference on the GPU.

```python
@Endpoint(
    name="llm-inference",
    gpu=GpuType.NVIDIA_GEFORCE_RTX_4090,
    dependencies=["vllm", "torch"],
    env={"HF_TOKEN": HF_TOKEN}
)
async def llm(prompt: str) -> str:
    ...
```

- Loads `meta-llama/Llama-3.1-8B-Instruct` via vLLM
- Accepts a `prompt` string and returns generated text (up to 512 tokens)
- Requires a Hugging Face token (`HF_TOKEN`) because Llama is a gated model

### 2. Agent endpoint (`agent_endpoint.py`)

No GPU — this runs on a **CPU worker**, which is much cheaper. The agent only orchestrates; the heavy lifting happens in the LLM endpoint.

Inside the agent, a small **LangGraph** with three nodes:

1. **Classify** — checks whether the question needs fresh web data (keywords like "today", "latest", "current", etc.)
2. **Search** — if needed, fetches a snippet from the DuckDuckGo instant answer API
3. **Answer** — builds a prompt (question + optional search context) and calls the `llm-inference` endpoint over HTTP

```python
@Endpoint(
    name="agent",
    dependencies=["langgraph", "langchain", "httpx"]
)
async def agent(question: str) -> dict:
    graph = build_graph()
    result = await graph.ainvoke({...})
    return {...}
```

The agent returns a structured response:

```json
{
  "question": "...",
  "used_search": false,
  "search_result": null,
  "answer": "..."
}
```

## Prerequisites

- A [RunPod](https://www.runpod.io/) account with billing enabled
- [RunPod Flash CLI](https://docs.runpod.io/flash/cli/overview) installed
- A [Hugging Face](https://huggingface.co/) account with access to `meta-llama/Llama-3.1-8B-Instruct`
- Python 3.10+ (3.11 or 3.12 recommended)

## Setup

### 1. Install Flash

```bash
pip install runpod-flash
```

### 2. Log in to RunPod

```bash
flash login
```

### 3. Configure environment variables

Create a `.env` file in the project root (or export these in your shell):

```bash
# Required for Llama 3.1 (gated model on Hugging Face)
HF_TOKEN=hf_your_hugging_face_token

# Required for calling deployed endpoints
RUNPOD_API_KEY=your_runpod_api_key
```

Request access to [Llama 3.1 8B Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) on Hugging Face before deploying.

### 4. Update the agent's API key

In `agent_endpoint.py`, replace the placeholder in the LLM call:

```python
headers={"Authorization": f"Bearer YOUR_RUNPOD_API_KEY"}
```

Use your real RunPod API key, or wire it through an environment variable the same way `HF_TOKEN` is passed to the LLM endpoint.

After deployment, Flash prints the actual endpoint URLs. Update the LLM URL in `agent_endpoint.py` if it differs from the default:

```python
"https://api.runpod.ai/v2/llm-inference/runsync"
```

## Run locally (development)

Test both endpoints on your machine before deploying:

```bash
flash run
```

Flash starts local workers for each endpoint. The agent comes up quickly; the LLM endpoint takes longer on first run while model weights are downloaded.

### Test with curl

```bash
curl -X POST http://localhost:8000/agent \
  -H "Content-Type: application/json" \
  -d '{"question": "What is serverless GPU and why does it matter?"}'
```

Watch the terminal trace: the agent classifies the question, skips search (no freshness keywords), calls the LLM endpoint, and returns the answer.

Try a question that triggers search:

```bash
curl -X POST http://localhost:8000/agent \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the latest news about serverless GPUs in 2026?"}'
```

## Deploy to RunPod

When you are ready for cloud execution:

```bash
flash deploy
```

Flash builds both endpoints, uploads the artifact, and provisions Serverless workers on RunPod.

- The **agent (CPU)** endpoint becomes ready almost immediately.
- The **LLM (4090)** endpoint has a longer **cold start** on first run while it pulls model weights and loads vLLM — typically **30–90 seconds** for the first request, much faster once the worker is warm.

After deployment, Flash prints the endpoint URLs. Call the agent with your RunPod API key:

```bash
curl -X POST https://api.runpod.ai/v2/<your-agent-endpoint-id>/runsync \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": {"question": "What is serverless GPU and why does it matter?"}}'
```

### Optional: Node.js client

`api.js` shows how to call a deployed agent endpoint from JavaScript. Update the URL and API key, then run:

```bash
node api.js
```

## What to watch on the RunPod console

After sending a request, open the [RunPod console](https://www.runpod.io/console/serverless):

1. **Two endpoints** — `agent` (CPU) and `llm-inference` (GPU)
2. **Workers spin up on demand** when a request arrives
3. **Cost meter** ticks only while workers are active
4. When traffic stops, workers scale back to zero

That is the serverless GPU promise: GPU inference without always-on infrastructure, with code as the source of truth.

## Cold start expectations

| Endpoint | First request | Warm requests |
|----------|---------------|---------------|
| `agent` (CPU) | ~seconds | Fast |
| `llm-inference` (4090) | ~30–90s (model download + load) | Fast |

Subsequent calls reuse warm workers until idle timeout.

## Learn more

- [RunPod Flash quickstart](https://docs.runpod.io/flash/quickstart)
- [Deploy Flash apps](https://docs.runpod.io/flash/apps/deploy-apps)
- [LangGraph documentation](https://langchain-ai.github.io/langgraph/)
- [vLLM documentation](https://docs.vllm.ai/)

---

**Subscribe to [ByteMonk](https://www.youtube.com/@ByteMonk) for more such content.**
# runpod
