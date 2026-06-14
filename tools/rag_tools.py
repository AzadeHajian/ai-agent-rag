# tools/rag_tools.py
# -----------------------------------------------------------
# LangChain @tool function for retrieving chunks of LangChain
# documentation from the pgvector knowledge base. Used by RAGAgent.
#
# Flow the agent follows:
#   1. retrieve_docs(query)  -> relevant chunks + sources
#   2. answer the question, grounded in those chunks, citing sources
# -----------------------------------------------------------

from langchain_core.tools import tool
from rag.vectorstore import get_vectorstore


@tool
def retrieve_docs(query: str, k: int = 4) -> str:
    """
    Search the LangChain documentation knowledge base for chunks
    relevant to the query, using semantic similarity search.

    Always call this FIRST before answering any question — never
    answer from general knowledge alone.

    Args:
        query: The user's question or a short search phrase derived from it.
        k: Number of chunks to retrieve (default 4).

    Returns:
        Formatted text containing the retrieved chunks, each labeled
        with its source file and title, or a message saying nothing
        relevant was found.
    """
    try:
        store = get_vectorstore()
        results = store.similarity_search_with_score(query, k=k)

        if not results:
            return "No relevant documents found in the knowledge base."

        formatted = []
        for doc, score in results:
            source = doc.metadata.get("source", "unknown")
            title = doc.metadata.get("title", "")
            formatted.append(
                f"--- Source: {source} | Title: {title} | Relevance: {score:.3f} ---\n"
                f"{doc.page_content}"
            )

        return "\n\n".join(formatted)
    except Exception as e:
        return f"Error retrieving documents: {e}"
