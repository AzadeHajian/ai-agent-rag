# rag/splitter.py
# -----------------------------------------------------------
# Splits cleaned documents into chunks small enough to embed and
# precise enough to retrieve.
# -----------------------------------------------------------

import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

CHUNK_SIZE_DEFAULT = 1000
CHUNK_OVERLAP_DEFAULT = 150


def split_documents(documents: list[Document]) -> list[Document]:
    """
    Split Documents into chunks for embedding/retrieval.

    Defaults (1000 chars / 150 overlap) are tuned for Sphinx API-reference
    HTML: large enough to keep a class signature + docstring together in
    one chunk, with overlap so sentences aren't cut across chunk
    boundaries. Override via RAG_CHUNK_SIZE / RAG_CHUNK_OVERLAP in .env.
    """
    chunk_size = int(os.getenv("RAG_CHUNK_SIZE", CHUNK_SIZE_DEFAULT))
    chunk_overlap = int(os.getenv("RAG_CHUNK_OVERLAP", CHUNK_OVERLAP_DEFAULT))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(documents)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i

    return chunks
