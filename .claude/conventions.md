# Conventions

- New tool: `@tool` function in `tools/`, exported via a list constant in
  `tools/__init__.py`.
- New agent: mirror `agent/agent.py`'s `SQLAgent` interface exactly
  (`__init__(provider)`, `run`, `get_model_name`, `get_provider`).
- Each module that needs env vars calls `load_dotenv()` itself rather than
  relying on it having been called elsewhere.
