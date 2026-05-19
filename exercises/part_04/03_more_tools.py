"""Agentic workflow with multiple tools.

The agent can choose between:
- listing available source documents,
- searching with normal RAG,
- searching with Graph-RAG,
- searching the web.
"""

import uvicorn
import chromadb
from loguru import logger
from pydantic_ai import Agent, ModelSettings
from pydantic_ai.capabilities import WebSearch

from datetime import datetime
from pathlib import Path

from graph_rag_workshop.utils.part_03_graph_construction_utils import (
    extend_graph_from_retrieved_chunks,
    load_graph,
    retrieve_graph_context,
)
from graph_rag_workshop.utils.part_01_document_tools import (
    list_my_available_documents,
    extract_text_from_pdf_file,
    extract_text_from_image_file,
)
from graph_rag_workshop.utils.part_02_rag_utils import retrieve_context
from graph_rag_workshop.utils.pydantic_utils import get_ollama_model

# Define file paths and constants
HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent.parent / "data"
MY_DOCUMENTS = DATA_DIR / "my_documents"
VECTORSTORES_DIR = DATA_DIR / "vectorstores"
VECTORSTORE_NAME = "chromadb_from_all_pdfs_documents"
GRAPH_JSON_PATH = DATA_DIR / "knowledge_graph.json"


agent = Agent(
    model=get_ollama_model(),
    instructions=(
        "You are a helpful document assistant. Use the tools to answer the user question. Mention source filenames you used."
    ),
    capabilities=[
        WebSearch(builtin=False),
    ],
    # EXERCISE - Document tools:
    # Use the Part 1 document helper functions so the agent can list and read local files.
    tools=...,
    model_settings=ModelSettings(
        thinking="minimal",
    ),
)


collection = chromadb.PersistentClient(path=str(VECTORSTORES_DIR)).get_collection(
    name=VECTORSTORE_NAME
)

# EXERCISE - Current date tool:
# Add the same @agent.tool_plain date tool from the simple tool-calling exercise.
...

# EXERCISE - Graph-RAG tool:
# Add the graph-context retrieval tool, as in the previous exercise.
...


@agent.tool_plain
def search_rag_context_from_alpine_evening_agreement(question: str) -> str:
    """Search RAG context from the ALPINE EVENING documents."""
    return retrieve_context(collection, question)


if __name__ == "__main__":
    app = agent.to_web()
    logger.info("Starting More Tools Agent on http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)
