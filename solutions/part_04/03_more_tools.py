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
    search_graph_rag_context,
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
GRAPH_JSON_PATH = DATA_DIR / "graph_solutions" / "knowledge_graph.json"
VECTORSTORE_DATABASE = chromadb.PersistentClient(
    path=str(VECTORSTORES_DIR)
).get_collection(name=VECTORSTORE_NAME)

agent = Agent(
    model=get_ollama_model(),
    instructions=(
        "You are a helpful assistant. Use the tools to answer the user question. Mention source filenames you used."
    ),
    capabilities=[
        WebSearch(builtin=False),
    ],
    # SOLUTION - Document tools:
    # Register the Part 1 helpers as tools the agent can choose from.
    tools=[
        list_my_available_documents,
        extract_text_from_pdf_file,
        extract_text_from_image_file,
    ],
    model_settings=ModelSettings(
        thinking="minimal",
    ),
)


# SOLUTION - Current date tool:
# This mirrors the simple tool-calling exercise and gives the agent current time access.
@agent.tool_plain
def get_current_date() -> str:
    """Get the current date and time.
    Returns:
        A string representation of the current date and time.
    """
    logger.info("Agent is using the tool to get the current date and time.")
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# SOLUTION - Graph-RAG tool:
# This tool returns both graph facts and retrieved chunks for contract questions.
@agent.tool_plain
def search_graph_rag_context_from_alpine_evening_agreement(question: str) -> str:
    """Search graph facts and vector chunks from the ALPINE EVENING documents. Better retrieval, but more resource consuming"""
    return search_graph_rag_context(
        vector_database=VECTORSTORE_DATABASE,
        graph_json_path=GRAPH_JSON_PATH,
        question=question,
        max_results=5,
    )


@agent.tool_plain
def search_rag_context_from_alpine_evening_agreement(question: str) -> str:
    """Search RAG context from the ALPINE EVENING documents."""
    return retrieve_context(VECTORSTORE_DATABASE, question)


if __name__ == "__main__":
    app = agent.to_web()
    logger.info("Starting More Tools Agent on http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)
