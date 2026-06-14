# rag/vectorstore.py
# -----------------------------------------------------------
# Connects to the pgvector-enabled Supabase Postgres database
# used as the vector store for the Docs Assistant (RAG).
#
# DATABASE_URL points at a *different* Supabase project than the
# SQL Assistant's SUPABASE_URL/SUPABASE_ANON_KEY — this one is the
# dedicated pgvector store.
#
# Supabase's pooled connection (port 6543) runs PgBouncer in
# "transaction mode", which does not support psycopg's automatic
# server-side prepared statements. We disable those via
# `prepare_threshold=None` so the same DATABASE_URL works whether
# it points at the transaction pooler or a direct/session connection.
# -----------------------------------------------------------

import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from langchain_postgres import PGVector

from rag.embeddings import get_embeddings

load_dotenv()

COLLECTION_NAME_DEFAULT = "langchain_docs"


def _connection_string() -> str:
    raw = os.getenv("DATABASE_URL")
    if not raw:
        raise ValueError("DATABASE_URL not found in .env")

    # langchain_postgres / SQLAlchemy needs the psycopg3 dialect prefix.
    if raw.startswith("postgresql://"):
        raw = raw.replace("postgresql://", "postgresql+psycopg://", 1)
    return raw


def get_engine() -> Engine:
    """SQLAlchemy engine for the pgvector Supabase project."""
    return create_engine(_connection_string(), connect_args={"prepare_threshold": None})


def get_vectorstore(collection_name: str | None = None) -> PGVector:
    """
    Return a PGVector store bound to the pgvector Supabase project.

    On first use, this auto-creates the `vector` extension and the
    `langchain_pg_collection` / `langchain_pg_embedding` tables.
    """
    collection = collection_name or os.getenv("RAG_COLLECTION_NAME", COLLECTION_NAME_DEFAULT)
    return PGVector(
        embeddings=get_embeddings(),
        collection_name=collection,
        connection=get_engine(),
        use_jsonb=True,
        create_extension=True,
    )
