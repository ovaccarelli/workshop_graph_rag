"""Schema Graph-RAG exercise.

Flow:
1. Build a tiny graph from the main agreement using the user-provided schema.
2. Use RAG to retrieve the chunks most relevant to the user's question.
3. Extract the same schema from those chunks and merge it into the graph.
4. Answer using the expanded graph plus the retrieved chunks.
"""

from pathlib import Path
import re
import time

import chromadb
from pydantic_ai import Agent, ModelSettings

from graph_rag_workshop.utils.pydantic_utils import get_ollama_model
from graph_rag_workshop.utils.part_02_rag_utils import retrieve_context
from graph_rag_workshop.utils.part_03_graph_construction_utils import (
    facts_to_graph,
    graph_edges_to_text,
    load_graph,
    visualize_graph,
    extract_facts,
)

# Define file paths and constants
HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent.parent / "data"
VECTORSTORES_DIR = DATA_DIR / "vectorstores"
VECTORSTORE_NAME = "chromadb_from_all_pdfs_documents"

GRAPH_JSON_PATH = DATA_DIR / "knowledge_graph.json"
GRAPH_HTML_PATH = DATA_DIR / "expanded_graph_rag.html"
USER_QUERY = "Which venue proposals listed in the email are compliant with respect to the service agreement?"

#################################################################
# STEP 1 - Load graph built in previous exercise
#################################################################

baseline_graph = load_graph(GRAPH_JSON_PATH)

print(
    f"✅ Loaded graph with {len(baseline_graph.nodes)} nodes and {len(baseline_graph.edges)} edges"
)

#################################################################
# STEP 2 - Retrieve relevant chunks from vector database build in part_02
#################################################################

vector_database = chromadb.PersistentClient(path=str(VECTORSTORES_DIR)).get_collection(
    name=VECTORSTORE_NAME
)

relevant_context = retrieve_context(vector_database, USER_QUERY)

print(f"✅ Retrieved chunks for question: {USER_QUERY}")

#################################################################
# STEP 3 - Extend the graph from the retrieved chunks
#################################################################

# Construct a graph from the retrieved chunks
start = time.time()
retrieved_facts = extract_facts(relevant_context)
retrieved_graph = facts_to_graph(retrieved_facts, "retrieved_chunks")
end = time.time()

print(f"✅ Constructed retrieved graph in {end - start:.1f} seconds")

# SOLUTION - Merge graphs:
# Add facts extracted from retrieved chunks to the graph built in the previous exercise.
baseline_graph.merge(retrieved_graph)
graph = baseline_graph
end = time.time()

print(f"✅ Expanded graph: {len(graph.nodes)} nodes, {len(graph.edges)} edges")

#################################################################
# STEP 4 - Retrieve graph facts required by the question
#################################################################

# Retrieve graph edges that overlap with the user question
query_terms = set(re.findall(r"[a-z0-9]+", USER_QUERY.lower()))

# SOLUTION - Limit graph facts:
# Keep the top-scoring graph edges so the answer prompt stays focused.
max_results = 12

scored = []
for edge in graph.edges:
    source = graph.nodes[edge.source]
    target = graph.nodes[edge.target]
    text = f"{source.label} {edge.relation} {target.label} {edge.evidence}".lower()
    score = len(query_terms & set(re.findall(r"[a-z0-9]+", text)))
    scored.append((score, edge))

scored.sort(key=lambda item: item[0], reverse=True)
selected = [edge for score, edge in scored[:max_results] if score > 0]
selected = selected or graph.edges[:max_results]

# Convert selected graph edges to text for the agent
graph_context = graph_edges_to_text(graph, selected)

print(f"✅ Graph context: {graph_context}")

#################################################################
# STEP 5 - Visualize expanded graph
#################################################################

output_path = visualize_graph(
    graph,
    GRAPH_HTML_PATH,
    open_browser=True,
)

print(f"✅ Expanded graph visualization written to: {output_path}")

#################################################################
# STEP 6 - Ask the model
#################################################################

agent = Agent(
    model=get_ollama_model(),
    instructions=(
        "Answer only from the schema graph facts and retrieved chunks. "
        "Focus on the user's question and the schema fields: parties, "
        "obligations, and dates. If the answer is missing from the context, "
        "say what is missing."
    ),
    model_settings=ModelSettings(thinking="minimal"),
)

# SOLUTION - Build the final prompt:
# Provide the model with the question, graph facts, and retrieved chunks.
prompt = f"""
Question:
{USER_QUERY}

Schema graph facts:
{graph_context}

Retrieved context:
{relevant_context}
"""

result = agent.run_sync(prompt)

print("\n✅ FINAL ANSWER")
print(result.output)
