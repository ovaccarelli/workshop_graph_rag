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

# EXERCISE - Parameters for chunking:
# Choose a chunk size and overlap for splitting the extracted PDF text.
# The overlap keeps a little context shared between neighboring chunks.
CHUNK_SIZE = ...
CHUNK_OVERLAP = ...

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

# EXERCISE - Create embeddings:
# Convert every text chunk into an embedding vector using the embedding function above.
vectors = ...

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

# EXERCISE - Upsert chunks into ChromaDB:
# Store the text chunks in the ChromaDB collection.
# The generated IDs below let ChromaDB insert new chunks or update existing ones.
vector_database.upsert(
    documents=...,
    metadatas=metadatas,
    ids=[f"chunk_{i}" for i in range(len(chunks))],
)

print(
    f"✅ Stored chunks in ChromaDB collection '{COLLECTION_NAME}', with {vector_database.count()} chunks"
)
