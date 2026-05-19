"""Offline data preprocessing for a RAG pipeline.

This script stores PDF documents in a vector database for later retrieval.
"""

from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from pdfminer.high_level import extract_text


# Define file paths and constants
HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent.parent / "data"
MY_DOCUMENTS = DATA_DIR / "my_documents"
VECTORSTORES_DIR = DATA_DIR / "vectorstores"
COLLECTION_NAME = "part_02_service_agreement"

#################################################################
# STEP 1 - Extract PDF text (see part_01/02_document_tools.py)
#################################################################

pdf_path = MY_DOCUMENTS / "Service_Agreement.pdf"
pdf_text = extract_text(str(pdf_path))

print(f"✅ Extracted text from {pdf_path.name}: {len(pdf_text)} characters")

#################################################################
# STEP 2 - Split extracted text into chunks
#################################################################

# SOLUTION - Chunking parameters:
# These values create readable chunks while preserving context across chunk boundaries.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

chunks = []
metadatas = []

step = CHUNK_SIZE - CHUNK_OVERLAP

for position in range(0, len(pdf_text), step):
    chunk = pdf_text[position : position + CHUNK_SIZE].strip()

    if chunk:
        chunks.append(chunk)
        metadatas.append({"source": pdf_path.name})

print(f"✅ Split into {len(chunks)} chunks")
print("\nPreview of first chunk:")
print(chunks[0][:200])

#################################################################
# STEP 3 - Convert text to embeddings
#################################################################

embedding_function = embedding_functions.DefaultEmbeddingFunction()

# SOLUTION - Create embeddings:
# The embedding function maps each chunk to a numeric vector for semantic search.
vectors = embedding_function(chunks)

print(f"✅ Converted {len(vectors)} chunks to {len(vectors)} embedding vectors")
print(f"Vectors length: {len(vectors[0])}. (Fixed by the embedding model)")
print(f"First 3 values of first vector: {vectors[0][:3]}")

#################################################################
# STEP 4 - Store chunks in ChromaDB vector database
#################################################################

client = chromadb.PersistentClient(path=str(VECTORSTORES_DIR))

vector_database = client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_function,
)

# SOLUTION - Upsert chunks:
# Store the prepared chunks with metadata and stable chunk IDs.
vector_database.upsert(
    documents=chunks,
    metadatas=metadatas,
    ids=[f"chunk_{i}" for i in range(len(chunks))],
)

print(
    f"✅ Stored chunks in ChromaDB collection '{COLLECTION_NAME}', with {vector_database.count()} chunks"
)
