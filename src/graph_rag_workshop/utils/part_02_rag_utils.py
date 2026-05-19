"""Utility functions for a simple Retrieval-Augmented Generation (RAG) workflow."""

from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from pdfminer.high_level import extract_text
from loguru import logger

from graph_rag_workshop.settings import (
    DEFAULT_VECTORSTORE_NAME,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_NB_RETRIEVED_CHUNKS,
)


def split_text_into_chunks(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """Split extracted text into overlapping chunks.

    Args:
        text: The extracted text to split into chunks.
        chunk_size: The size of each text chunk.
        chunk_overlap: The number of overlapping characters between chunks.

    Returns:
        A list of text chunks.
    """

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks = []
    step = chunk_size - chunk_overlap

    for position in range(0, len(text), step):
        chunk = text[position : position + chunk_size].strip()

        if chunk:
            chunks.append(chunk)

    return chunks


def create_vector_database(
    chunks: list[str],
    vectorstores_dir: Path,
    vectorstore_name: str = DEFAULT_VECTORSTORE_NAME,
    reindex: bool = True,
) -> chromadb.Collection:
    """Create a ChromaDB collection and store the chunks.

    Args:
        chunks: A list of text chunks to store in the vector database.
        vectorstores_dir: The directory where ChromaDB will store its data.
        vectorstore_name: The name of the ChromaDB collection to create or load.
        reindex: If True, any existing collection with the same name will be deleted and recreated.
            If False, the existing collection will be used if it exists.

    Returns:
        A ChromaDB collection containing the stored chunks.
    """

    client = chromadb.PersistentClient(path=str(vectorstores_dir))

    if reindex:
        try:
            client.delete_collection(name=vectorstore_name)
        except Exception:
            pass

    embedding_function = embedding_functions.DefaultEmbeddingFunction()

    vector_database = client.get_or_create_collection(
        name=vectorstore_name,
        embedding_function=embedding_function,
    )

    if not reindex and vector_database.count() > 0:
        return vector_database

    if not chunks:
        return vector_database

    try:
        vector_database.upsert(
            documents=chunks,
            ids=[f"chunk_{i}" for i in range(len(chunks))],
        )
    except Exception as exc:
        logger.error(
            f"Failed to upsert chunks into ChromaDB collection "
            f"'{vectorstore_name}': {exc}"
        )
        raise

    return vector_database


def create_vector_database_from_directory(
    documents_dir: Path,
    vectorstores_dir: Path,
    vectorstore_name: str = DEFAULT_VECTORSTORE_NAME,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    reindex: bool = True,
) -> chromadb.Collection:
    """Build a ChromaDB vector database from a directory of PDF files.

    Args:
        documents_dir: The directory containing the PDF documents to process.
        vectorstores_dir: The directory where ChromaDB will store its data.
        vectorstore_name: The name of the ChromaDB collection to create or load.
        chunk_size: The size of each text chunk.
        chunk_overlap: The number of overlapping characters between chunks.
        reindex: If True, any existing collection with the same name will be deleted and recreated.
            If False, the existing collection will be used if it exists.

    Returns:
        A ChromaDB collection containing the stored chunks from all PDF documents.
    """

    if not documents_dir.exists():
        logger.error(f"Documents directory does not exist: {documents_dir}")

        return create_vector_database(
            chunks=[],
            vectorstores_dir=vectorstores_dir,
            vectorstore_name=vectorstore_name,
            reindex=reindex,
        )

    pdf_paths = [
        path
        for path in documents_dir.rglob("*")
        if path.is_file() and path.suffix.lower() == ".pdf"
    ]

    if not pdf_paths:
        logger.error(f"No PDF files found in directory: {documents_dir}")

    chunks: list[str] = []

    for pdf_path in pdf_paths:
        try:
            text = extract_text(pdf_path)
        except Exception as exc:
            logger.error(f"Failed to extract text from {pdf_path}: {exc}")
            continue

        if not text.strip():
            logger.error(f"No extractable text found in PDF: {pdf_path}")
            continue

        pdf_chunks = split_text_into_chunks(
            text=text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        source_name = pdf_path.stem.replace("_", " ")
        chunks.extend(
            f"Source document: {source_name} ({pdf_path.name})\n\n{chunk}"
            for chunk in pdf_chunks
        )

    if not chunks:
        logger.error("No chunks were created from the PDF documents.")

    return create_vector_database(
        chunks=chunks,
        vectorstores_dir=vectorstores_dir,
        vectorstore_name=vectorstore_name,
        reindex=reindex,
    )


def retrieve_relevant_chunks(
    vector_database: chromadb.Collection,
    user_query: str,
    nb_retrieved_chunks: int = DEFAULT_NB_RETRIEVED_CHUNKS,
) -> list[str]:
    """Retrieve the most relevant chunks based on semantic similarity.

    Args:
        vector_database: The ChromaDB collection to query.
        user_query: The user's question.
        nb_retrieved_chunks: The number of chunks to retrieve.

    Returns:
        A list of retrieved chunks and their PDF source names.
    """

    try:
        results = vector_database.query(
            query_texts=[user_query],
            n_results=nb_retrieved_chunks,
            include=["documents"],
        )
    except Exception as exc:
        logger.error(f"Failed to retrieve context from ChromaDB: {exc}")
        raise

    documents = results.get("documents") or [[]]
    return documents[0]


def retrieve_context(
    vector_database: chromadb.Collection,
    user_query: str,
    nb_retrieved_chunks: int = DEFAULT_NB_RETRIEVED_CHUNKS,
) -> str:
    """Retrieve and format the most relevant chunks as context.

    Args:
        vector_database: The ChromaDB collection to query.
        user_query: The user's question.
        nb_retrieved_chunks: The number of chunks to retrieve.

    Returns:
        A string containing the retrieved chunks and their PDF source names.
    """

    retrieved_chunks = retrieve_relevant_chunks(
        vector_database=vector_database,
        user_query=user_query,
        nb_retrieved_chunks=nb_retrieved_chunks,
    )

    if not retrieved_chunks:
        return "No relevant document chunks were found."

    return "\n------------\n".join(retrieved_chunks)
