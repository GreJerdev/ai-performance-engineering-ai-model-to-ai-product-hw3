# ai-performance-engineering-ai-model-to-ai-product-hw3

Bitext customer-support agent built with LangGraph.

## Setup

From the project root:

```bash
python -m venv .venv
```

**Windows (PowerShell):**

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**macOS / Linux:**

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file with your Nebius API key:

```
NEBIUS_API_KEY=your-key-here
```

The first run may take a while while data and embedding models are loaded.

## Run with Streamlit UI (recommended)

Chat UI in the browser, backed by `LangGraphAgent` and SQLite checkpointing (`checkpoints.sqlite`).

**Terminal:**

```bash
streamlit run streamlit_app.py
```

Open the URL shown in the terminal (usually http://localhost:8501).

**Cursor / VS Code:**

1. Select the `.venv` interpreter (**Python: Select Interpreter**).
2. **Run and Debug** (`Ctrl+Shift+D`).
3. Choose **Streamlit: streamlit_app** and press **F5**.

The sidebar lets you edit the **Thread ID** (for multi-turn memory) and **Clear chat** (UI history only).

> Streamlit runs the agent directly. You do not need `langgraph dev` running at the same time.

## Run with LangGraph CLI

Start the local dev server (LangGraph Studio UI):

```bash
langgraph dev
```

The graph is registered in `langgraph.json` as `bitext_agent`. For the CLI, export a compiled graph from `app/ai_agent/agent.py` (uncomment or add):

```python
graph = build_graph(get_llm())
```

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

This uses SQLite checkpointing via `start_agent()` instead of the CLI server or Streamlit.
