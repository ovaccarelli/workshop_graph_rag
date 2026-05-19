"""Online Semantic Retrieval Example.

This script demonstrates a simple Retrieval-Augmented Generation (RAG)
workflow using ChromaDB as the vector database and a language model
for answering user questions based on retrieved document chunks.

The script performs the following steps:
1. Creates a vector database collection from PDF documents in a specified directory.
2. Retrieves relevant chunks from the vector database for a user question.
3. Uses a language model agent to answer the user question based on the retrieved context.
"""

import time
from pathlib import Path

from pydantic_ai import Agent, ModelSettings

from graph_rag_workshop.utils.pydantic_utils import get_ollama_model
from graph_rag_workshop.utils.part_02_rag_utils import (
    create_vector_database_from_directory,
)


# Define file paths and constants
HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent.parent / "data"
MY_DOCUMENTS = DATA_DIR / "my_documents"
VECTORSTORES_DIR = DATA_DIR / "vectorstores"
VECTORSTORE_NAME = "chromadb_from_all_pdfs_documents"

#################################################################
# STEP 1 - Create vector database from PDF documents in a directory (see part_02/01_offline_data_preprocessing.py for details)
#################################################################

vector_database = create_vector_database_from_directory(
    documents_dir=MY_DOCUMENTS,
    vectorstores_dir=VECTORSTORES_DIR,
    vectorstore_name=VECTORSTORE_NAME,
)

print(f"✅ Created vector database collection: {VECTORSTORE_NAME}")
print("Collection count:", vector_database.count())

#################################################################
# STEP 2 - Retrieve relevant chunks for a user question
#################################################################

# SOLUTION - Retrieval parameter:
# Retrieve a small set of likely relevant chunks for the user question.
NB_RETRIEVED_CHUNKS = 5
USER_QUERY = "Which venue proposals listed in the email are compliant with respect to the service agreement?"

results = vector_database.query(
    query_texts=[USER_QUERY],
    n_results=NB_RETRIEVED_CHUNKS,
    include=["documents"],
)

retrieved_chunks = results.get("documents", [[]])[0]
print(f"\n✅ Retrieved {len(retrieved_chunks)} chunks")

for i, chunk in enumerate(retrieved_chunks, 1):
    print(f"\nCHUNK {i} — Preview:")
    print(chunk[:500])

#################################################################
# STEP 3 - Answer the user question using the retrieved context
#################################################################

# SOLUTION - System prompt:
# The prompt constrains the model to use retrieved context instead of guessing.
SYSTEM_PROMPT = (
    "You are a helpful assistant. Use the retrieved document context to "
    "answer the user's question. If the answer is not in the context, "
    "say you don't know. Cite sources using the PDF name."
)

agent = Agent(
    model=get_ollama_model(),
    model_settings=ModelSettings(thinking="minimal"),
    instructions=(SYSTEM_PROMPT),
)

# SOLUTION - Build relevant context:
# Join the retrieved chunks with a visible separator before sending them to the model.
relevant_context = "\n\n------------\n\n".join(retrieved_chunks)

start = time.time()
result = agent.run_sync(
    f"Question: {USER_QUERY}\n\nRetrieved context:\n{relevant_context}"
)
end = time.time()

print(f"\n✅ Answer generated in {end - start:.2f} seconds")
print(f"User Question: {USER_QUERY}")
print("\nANSWER:")
print(result.output)
