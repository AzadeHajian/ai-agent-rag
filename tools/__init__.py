# tools/__init__.py
from .supabase_tools import (
    execute_sql,
    list_tables,
    get_table_schema,
    get_sample_rows,
)
from .rag_tools import retrieve_docs

SUPABASE_TOOLS = [
    list_tables,
    get_table_schema,
    get_sample_rows,
    execute_sql,
]

RAG_TOOLS = [
    retrieve_docs,
]