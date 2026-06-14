# Architecture

- `agent/` — LangGraph `create_react_agent` agents. `SQLAgent` and `RAGAgent`
  share one interface: `__init__(provider)`, `run(user_message, chat_history)`,
  `get_model_name()`, `get_provider()`.
- `llm/` — `get_llm(provider)` -> `AnthropicLLM` | `OpenAILLM` (`BaseLLM`).
  Controls the CHAT model only.
- `rag/` — embeddings (always OpenAI), pgvector vector store, ingestion
  pipeline (`loader` -> `splitter` -> `embeddings` -> `vectorstore` -> `ingest`).
- `tools/` — `@tool` functions, exported via list constants in
  `tools/__init__.py`: `SUPABASE_TOOLS` (SQL agent), `RAG_TOOLS` (Docs agent).
- `mcp_server/` — FastMCP server exposing the same tools over MCP.
- `main.py` — single Streamlit entrypoint, dual-mode via a sidebar radio
  (`app_mode`: "sql" | "docs") and session-state keys prefixed `sql_*` / `docs_*`.
