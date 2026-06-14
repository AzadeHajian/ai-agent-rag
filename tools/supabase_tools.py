# tools/supabase_tools.py
# -----------------------------------------------------------
# LangChain @tool functions that connect to Supabase.
# The agent uses these tools to explore the database
# and generate + execute SQL queries.
#
# Flow the agent follows:
#   1. list_tables()         → what tables exist?
#   2. get_table_schema()    → what columns does the table have?
#   3. get_sample_rows()     → what does the data look like?
#   4. execute_sql()         → run the final SQL query
# -----------------------------------------------------------

import os
from dotenv import load_dotenv
from langchain_core.tools import tool
from supabase import create_client, Client

load_dotenv()


# -----------------------------------------------------------
# Supabase client — created once, reused by all tools
# -----------------------------------------------------------
def _get_client() -> Client:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_ANON_KEY")

    if not url:
        raise ValueError("SUPABASE_URL not found in .env")
    if not key:
        raise ValueError("SUPABASE_ANON_KEY not found in .env")

    try:
        return create_client(url, key)
    except Exception as e:
        raise RuntimeError(f"Failed to create Supabase client: {e}") from e


# -----------------------------------------------------------
# Tool 1: List all tables
# -----------------------------------------------------------
@tool
def list_tables(dummy: str = "") -> str:
    """
    List all tables in the Supabase database.
    Always call this first before writing any SQL query
    so you know what tables are available.

    Returns:
        A list of table names.
    """
    try:
        client = _get_client()
        response = client.rpc("execute_sql", {
            "query": """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                ORDER BY table_name;
            """
        }).execute()
        return str(response.data)
    except Exception as e:
        return f"Error listing tables: {e}"


# -----------------------------------------------------------
# Tool 2: Get columns and types of a specific table
# -----------------------------------------------------------
@tool
def get_table_schema(table_name: str) -> str:
    """
    Get the column names and data types of a specific table.
    Call this before writing a SQL query so you know
    exactly what columns exist and their types.

    Args:
        table_name: Name of the table to inspect.

    Returns:
        Column names, data types, and nullable info.
    """
    try:
        client = _get_client()
        response = client.rpc("execute_sql", {
            "query": f"""
                SELECT
                    column_name,
                    data_type,
                    is_nullable
                FROM information_schema.columns
                WHERE table_schema = 'public'
                AND table_name = '{table_name}'
                ORDER BY ordinal_position;
            """
        }).execute()

        if not response.data:
            return f"Table '{table_name}' not found or has no columns."
        return str(response.data)
    except Exception as e:
        return f"Error getting schema for '{table_name}': {e}"


# -----------------------------------------------------------
# Tool 3: Preview first rows of a table
# -----------------------------------------------------------
@tool
def get_sample_rows(table_name: str) -> str:
    """
    Get the first 5 rows of a table as a preview.
    Use this to understand what kind of data is in
    the table before generating a SQL query.

    Args:
        table_name: Name of the table to preview.

    Returns:
        First 5 rows of the table.
    """
    try:
        client = _get_client()
        response = client.rpc("execute_sql", {
            "query": f"SELECT * FROM {table_name} LIMIT 5;"
        }).execute()

        if not response.data:
            return f"Table '{table_name}' is empty or does not exist."
        return str(response.data)
    except Exception as e:
        return f"Error getting sample rows for '{table_name}': {e}"


# -----------------------------------------------------------
# Tool 4: Execute a SQL query
# -----------------------------------------------------------
@tool
def execute_sql(query: str) -> str:
    """
    Execute a raw SQL query against the Supabase database.
    Use this to run the final SQL after you have explored
    the tables and schema using the other tools.

    Only use SELECT statements unless the user explicitly
    asks to insert, update, or delete data.

    Args:
        query: The SQL query to execute.

    Returns:
        Query results or an error message.
    """
    try:
        client = _get_client()
        response = client.rpc("execute_sql", {
            "query": query
        }).execute()

        if not response.data:
            return "Query executed successfully but returned no results."
        return str(response.data)
    except Exception as e:
        return f"SQL execution error: {e}"