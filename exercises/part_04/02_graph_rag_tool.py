"""Agentic Graph-RAG exercise.

The agent receives Graph-RAG as a callable tool. It decides when to call the
tool, instead of receiving all context in every prompt.
"""

from pathlib import Path

import uvicorn
from loguru import logger
import chromadb
from pydantic_ai import Agent, ModelSettings

from graph_rag_workshop.utils.part_03_graph_construction_utils import (
    search_graph_rag_context,
)
from graph_rag_workshop.utils.pydantic_utils import get_ollama_model

# Define file paths and constants
HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent.parent / "data"
VECTORSTORES_DIR = DATA_DIR / "vectorstores"
VECTORSTORE_NAME = "chromadb_from_all_pdfs_documents"
GRAPH_JSON_PATH = DATA_DIR / "knowledge_graph.json"
VECTORSTORE_DATABASE = chromadb.PersistentClient(
    path=str(VECTORSTORES_DIR)
).get_collection(name=VECTORSTORE_NAME)


agent = Agent(
    model=get_ollama_model(),
    instructions=(
        "You are a careful contract assistant for the ALPINE EVENING "
        "agreement. Use the Graph-RAG tool for questions about obligations, "
        "deadlines, conditions, remedies, changes, or source documents. Cite "
        "source PDF names from the tool output."
    ),
    model_settings=ModelSettings(thinking="minimal", output_retries=3),
)

# EXERCISE: Add search_graph_rag_context as a tool
...
def search_graph_rag_context_from_alpine_evening_agreement(question: str) -> str:
    """Search graph facts and vector chunks from the ALPINE EVENING documents."""
    return search_graph_rag_context(
        vector_database=VECTORSTORE_DATABASE,
        graph_json_path=GRAPH_JSON_PATH,
        question=question,
        max_results=5,
    )


if __name__ == "__main__":
    app = agent.to_web()
    logger.info("Starting Graph-RAG Tool Agent on http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)
