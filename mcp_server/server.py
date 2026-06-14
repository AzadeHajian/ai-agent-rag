# mcp_server/server.py
# -----------------------------------------------------------
# FastMCP server that wraps the Supabase tools and exposes
# them as MCP tools that any MCP client can discover and call.
#
# In simple words:
#   tools/supabase_tools.py  →  raw Python functions
#   mcp_server/server.py     →  those same functions exposed
#                               as MCP tools over the protocol
#
# Run this server with:
#   python -m mcp_server.server
# -----------------------------------------------------------

import os
from dotenv import load_dotenv
from fastmcp import FastMCP
from tools.supabase_tools import (
    execute_sql,
    list_tables,
    get_table_schema,
    get_sample_rows,
)
from tools.rag_tools import retrieve_docs

load_dotenv()

# -----------------------------------------------------------
# Create the FastMCP server instance
# -----------------------------------------------------------
mcp = FastMCP(
    name="sqlspeak-mcp",
    instructions="""
    You have access to two kinds of tools:

    SQL tools — for a Supabase PostgreSQL database:
    1. Call list_tables() to see what tables exist
    2. Call get_table_schema() to understand the columns
    3. Call get_sample_rows() to understand the data
    4. Call execute_sql() with the correct SQL query
    Always prefer SELECT queries unless the user explicitly asks to modify data.

    Documentation tool — for LangChain documentation questions:
    - Call retrieve_documentation(query) to search the pgvector knowledge
      base for relevant chunks before answering.
    """,
)


# -----------------------------------------------------------
# Tool 1: List all tables
# -----------------------------------------------------------
@mcp.tool()
def list_all_tables() -> str:
    """
    List all tables in the Supabase database.
    Always call this first to know what tables are available
    before writing any SQL query.
    """
    return list_tables.invoke({"dummy": ""})


# -----------------------------------------------------------
# Tool 2: Get schema of a table
# -----------------------------------------------------------
@mcp.tool()
def get_schema(table_name: str) -> str:
    """
    Get the column names and data types of a specific table.
    Call this before writing SQL so you know what columns exist.

    Args:
        table_name: The name of the table to inspect.
    """
    return get_table_schema.invoke({"table_name": table_name})


# -----------------------------------------------------------
# Tool 3: Preview rows of a table
# -----------------------------------------------------------
@mcp.tool()
def sample_rows(table_name: str) -> str:
    """
    Get the first 5 rows of a table as a data preview.
    Use this to understand what the data looks like before
    generating a SQL query.

    Args:
        table_name: The name of the table to preview.
    """
    return get_sample_rows.invoke({"table_name": table_name})


# -----------------------------------------------------------
# Tool 4: Execute SQL query
# -----------------------------------------------------------
@mcp.tool()
def run_sql(query: str) -> str:
    """
    Execute a SQL query against the Supabase database.
    Use this to run the final SQL query after exploring
    the tables and schema with the other tools.
    Only use SELECT unless the user explicitly asks to
    insert, update, or delete data.

    Args:
        query: The SQL query string to execute.
    """
    return execute_sql.invoke({"query": query})


# -----------------------------------------------------------
# Tool 5: Retrieve LangChain documentation chunks (RAG)
# -----------------------------------------------------------
@mcp.tool()
def retrieve_documentation(query: str, k: int = 4) -> str:
    """
    Search the LangChain documentation knowledge base (pgvector) for
    chunks relevant to the query, using semantic similarity search.

    Args:
        query: The search query or question.
        k: Number of chunks to retrieve (default 4).
    """
    return retrieve_docs.invoke({"query": query, "k": k})


# -----------------------------------------------------------
# Run the server
# -----------------------------------------------------------
if __name__ == "__main__":
    mcp.run()