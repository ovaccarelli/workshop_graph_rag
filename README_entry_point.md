# Agents Graph RAG Workshop

Entry point for the workshop **Design Thinking Machines**.

The first exercise runs a local Pydantic AI agent with an Ollama model. The
recommended setup is to install Ollama directly on your machine, then use `uv`
to install the Python dependencies and run the script.

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

## Check That Everything Works

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
