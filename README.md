# ai-performance-engineering-ai-model-to-ai-product-hw3

Bitext customer-support agent built with LangGraph.

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file with your Nebius API key:

```
NEBIUS_API_KEY=your-key-here
```

## Run with LangGraph CLI

Start the local dev server (LangGraph Studio UI):

```bash
langgraph dev
```

The graph is registered in `langgraph.json` as `bitext_agent`, exported from `app/ai_agent/agent.py:graph`.

Use a `thread_id` in configurable fields for multi-turn memory (the server provides checkpointing).

### Debug (`debugpy` on port 5678)

**Terminal:**

```bash
langgraph dev --debug-port 5678 --wait-for-client
```

The server pauses until a debugger attaches. In Cursor/VS Code, run **Run and Debug → Attach: LangGraph dev (5678)** (or use **LangGraph: dev + debug attach** to start the server and attach in one step via `.vscode/launch.json`).

## Run scripted demo (optional)

```bash
python main.py
```

This uses SQLite checkpointing via `start_agent()` instead of the CLI server.
