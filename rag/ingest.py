# rag/ingest.py
# -----------------------------------------------------------
# The RAG ingestion pipeline: load -> clean -> chunk -> embed -> store.
#
# `ingest_path()` is the single reusable entrypoint used by both:
#   - the CLI:    python -m rag.ingest documents/latest/callbacks
#   - the "Knowledge Base" panel in main.py (Docs Assistant mode)
#
# This module also holds two small helpers for that same panel:
#   - get_ingested_chunk_count()  -> how many chunks are already stored
#   - list_docs_subfolders()      -> which documents/latest/ folders exist
# -----------------------------------------------------------

import argparse
import os
from pathlib import Path

from sqlalchemy import text

from rag.loader import load_html_folder, DOCS_ROOT
from rag.splitter import split_documents
from rag.vectorstore import get_vectorstore, get_engine, COLLECTION_NAME_DEFAULT


def ingest_path(folder_path: str, collection_name: str | None = None) -> dict:
    """
    Ingest every *.html file in `folder_path` (non-recursive):
    load -> clean -> chunk -> embed -> store in the pgvector collection.

    Args:
        folder_path: path to a subfolder of documents/latest/, e.g.
                      "documents/latest/callbacks"
        collection_name: optional override for the PGVector collection
                      name (defaults to RAG_COLLECTION_NAME / "langchain_docs")

    Returns:
        {"folder", "files_loaded", "chunks_created", "chunks_inserted"}
    """
    folder = Path(folder_path)
    if not folder.is_dir():
        raise ValueError(f"Not a directory: {folder_path}")

    documents = load_html_folder(str(folder))
    chunks = split_documents(documents)

    store = get_vectorstore(collection_name)
    if chunks:
        store.add_documents(chunks)

    return {
        "folder": str(folder_path),
        "files_loaded": len(documents),
        "chunks_created": len(chunks),
        "chunks_inserted": len(chunks),
    }


def get_ingested_chunk_count(collection_name: str | None = None) -> int:
    """
    Number of chunks currently stored for `collection_name`
    (default: RAG_COLLECTION_NAME). Returns 0 if nothing has been
    ingested yet (the pgvector tables may not exist at all).
    """
    name = collection_name or os.getenv("RAG_COLLECTION_NAME", COLLECTION_NAME_DEFAULT)
    try:
        engine = get_engine()
        with engine.connect() as conn:
            result = conn.execute(
                text("""
                    SELECT COUNT(*) FROM langchain_pg_embedding e
                    JOIN langchain_pg_collection c ON e.collection_id = c.uuid
                    WHERE c.name = :name
                """),
                {"name": name},
            )
            return result.scalar() or 0
    except Exception:
        return 0


def list_docs_subfolders() -> list[str]:
    """Sorted subfolder names under documents/latest/ (for the folder picker)."""
    if not DOCS_ROOT.is_dir():
        return []
    return sorted(p.name for p in DOCS_ROOT.iterdir() if p.is_dir())


def _cli():
    parser = argparse.ArgumentParser(
        description="Ingest a documents/latest/ subfolder into the pgvector store"
    )
    parser.add_argument("folder", help="Path to folder, e.g. documents/latest/callbacks")
    parser.add_argument("--collection", default=None, help="Override PGVector collection name")
    args = parser.parse_args()

    result = ingest_path(args.folder, collection_name=args.collection)
    print(f"Loaded {result['files_loaded']} file(s)")
    print(f"Created {result['chunks_created']} chunk(s)")
    print(f"Inserted {result['chunks_inserted']} chunk(s) into the vector store")


if __name__ == "__main__":
    _cli()
