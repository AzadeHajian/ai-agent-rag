# rag/embeddings.py
# -----------------------------------------------------------
# Embeddings factory.
#
# Embeddings are ALWAYS OpenAI, regardless of which chat LLM
# provider (OpenAI/Anthropic) is selected — Anthropic has no
# embeddings API. The "provider" choice in the UI only affects
# the chat model used by the agent, never the embedding model.
# -----------------------------------------------------------

import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()

EMBEDDING_MODEL_DEFAULT = "text-embedding-3-small"  # 1536 dims


def get_embeddings() -> OpenAIEmbeddings:
    """
    Return the embeddings client used to turn text into vectors.

    Reads OPENAI_API_KEY (required) and EMBEDDING_MODEL (optional,
    defaults to text-embedding-3-small) from .env.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in .env (required for embeddings)")

    model = os.getenv("EMBEDDING_MODEL", EMBEDDING_MODEL_DEFAULT)
    return OpenAIEmbeddings(model=model, api_key=api_key)
