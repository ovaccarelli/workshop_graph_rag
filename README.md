# Agents Graph RAG Workshop

A hands-on workshop for building AI agents with **Pydantic AI**, vector RAG, knowledge graphs, tool calling, and agents.

Use case: **Alpine Service Agreement**.

A company booked a quirky Swiss Night event package: cheese tasting, yodel performance, and an optional alp-horn introduction. Participants answer practical contract questions: who must do what, by when, and what happens if something goes wrong.

---

## Setup

### 1. Install uv

Install `uv`, the Python project and dependency manager used by this entry
point.

macOS and Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Windows PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Then restart your terminal and check that `uv` is available:

```bash
uv --version
```

### 2. Install the dependencies

From the project root, run:

```bash
uv sync
```

This command creates or updates the local `.venv` environment and installs all
dependencies needed for the entry script from `pyproject.toml` and `uv.lock`.

### 3. Install Ollama and pull the model

Preferred setup: install Ollama locally, then download the entry-point model.

1. Install Ollama from <https://ollama.com/download>.
2. Start Ollama.
3. Pull the model:

```bash
ollama pull qwen3.5:2b
```

The entry script uses this model by default:

```text
qwen3.5:2b
```

### 4. Check That Everything Works

To verify that your environment is ready for the entry point, run:

```bash
uv run python exercises/part_01/01_entry_point.py
```

Expected result: the script asks the local model a simple question and prints
the answer.

Note: the first run can take a few seconds because Ollama has to load the
`qwen3.5:2b` model.

Note: If the output is less than 5 tokens per second, pull a smaller model:

```bash
ollama pull qwen3.5:0.8b
```

Then run the script with that model:

```bash
OLLAMA_MODEL=qwen3.5:0.8b uv run python exercises/part_01/01_entry_point.py
```

### 5. Run any script with `uv`

   ```bash
   uv run python exercises/part_01/01_entry_point.py
   ```

> **Models:** Default is `qwen3.5:2b`.

---

## Project Structure

| Directory | Purpose |
|---|---|
| `data/my_documents/` | Alpine Event PDFs and image documents |
| `data/vectorstores/` | ChromaDB vector store |
| `data/graph_solutions/` | JSON and HTML examples of graphs |
| `exercises/` | Your workspace — fill in the blanks here |
| `presentation/` | pdf presentation of the workshop |
| `solutions/` | Exercises solutions |
| `src/graph_rag_workshop/` | Shared utilities for document tools, RAG, graph construction, and Pydantic AI model setup |

---

## Workshop Exercises

### Part 01 — Intro: Creating Agents and Document Tools

| Script | What it does |
|---|---|
| [`01_entry_point.py`](exercises/part_01/01_entry_point.py) | Minimal agent and smoke test (CLI and web app on port 8000) |
| [`02_document_tools.py`](exercises/part_01/02_document_tools.py) | Lists workshop documents and extracts text from PDF/image files (CLI) |

```bash
uv run python exercises/part_01/01_entry_point.py
uv run python exercises/part_01/02_document_tools.py
```

---

### Part 02 — Offline Data Preprocessing & Online Semantic Retrieval

| Script | What it does |
|---|---|
| [`01_offline_data_preprocessing.py`](exercises/part_02/01_offline_data_preprocessing.py) | Indexes one PDF into ChromaDB (CLI) |
| [`02_online_semantic_retrieval.py`](exercises/part_02/02_online_semantic_retrieval.py) | Retrieves relevant chunks and answers from document context (CLI) |

```bash
uv run python exercises/part_02/01_offline_data_preprocessing.py
uv run python exercises/part_02/02_online_semantic_retrieval.py
```

---

### Part 03 — Knowledge Graph Construction & Graph-RAG

| Script | What it does |
|---|---|
| [`01_graph_construction.py`](exercises/part_03/01_graph_construction.py) | Builds a knowledge graph from the Alpine Event agreement and annex (CLI) |
| [`02_graph_rag.py`](exercises/part_03/02_graph_rag.py) | Graph-RAG — combines graph facts with vector chunks before answering (CLI) |

```bash
uv run python exercises/part_03/01_graph_construction.py
uv run python exercises/part_03/02_graph_rag.py
```

---

### Part 04 — Tool Calling and Agentic Workflows

This part turns retrieval and graph reasoning into tools an agent can choose to call.

| Script | What it does |
|---|---|
| [`01_simple_tool_call.py`](exercises/part_04/01_simple_tool_call.py) | Agent with a simple date/time tool (web app on port 8000) |
| [`02_graph_rag_tool.py`](exercises/part_04/02_graph_rag_tool.py) | Agent with Graph-RAG exposed as a tool (web app on port 8000) |
| [`03_more_tools.py`](exercises/part_04/03_more_tools.py) | Agent with document tools, graph facts, Graph-RAG, web search, and time (web app on port 8000) |

```bash
uv run python exercises/part_04/01_simple_tool_call.py
uv run python exercises/part_04/02_graph_rag_tool.py
uv run python exercises/part_04/03_more_tools.py
```