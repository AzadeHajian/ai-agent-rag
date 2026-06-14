# Gotchas

- `DATABASE_URL` uses Supabase's port-6543 transaction pooler (PgBouncer),
  which doesn't support prepared statements — `rag/vectorstore.get_engine()`
  disables them via `connect_args={"prepare_threshold": None}`. Reuse
  `get_engine()` for any new direct-SQL queries against that database.
- `langchain_postgres.PGVector` requires the `postgresql+psycopg://` dialect
  prefix; `DATABASE_URL` is stored as plain `postgresql://` and rewritten in
  `rag/vectorstore._connection_string()`.
- Embeddings are ALWAYS OpenAI (`rag/embeddings.py`, `text-embedding-3-small`),
  independent of the chat `provider` — Anthropic has no embeddings API.
  `OPENAI_API_KEY` is required even when using Claude for chat.
- There are two separate Supabase projects: `SUPABASE_URL`/`SUPABASE_ANON_KEY`
  (SQL Assistant's data) vs. `DATABASE_URL` (pgvector store for the Docs
  Assistant). Don't conflate them.
- The Docs Assistant only knows what's been ingested via
  `rag.ingest.ingest_path()` (CLI: `python -m rag.ingest <folder>`, or the
  "Knowledge Base" panel in Docs mode). An empty/partial knowledge base is
  expected on a fresh setup, not a bug.
- `rag/loader.py`'s `load_html_folder()` resolves `folder_path` to an
  absolute path before globbing — `Document.metadata["source"]` is computed
  via `path.relative_to(DOCS_ROOT)`, which raises `ValueError` if `path` is
  relative while `DOCS_ROOT` is absolute (see `.claude/debug-log.md`).
