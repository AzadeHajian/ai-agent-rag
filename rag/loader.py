# rag/loader.py
# -----------------------------------------------------------
# Loads LangChain's Sphinx-generated HTML docs (documents/latest/)
# and strips them down to their main content before chunking.
#
# Each HTML page is mostly navigation/sidebar/script boilerplate
# around a `<div role="main">` block that holds the actual API
# reference text — that's the part worth embedding.
# -----------------------------------------------------------

from pathlib import Path
from bs4 import BeautifulSoup
from langchain_core.documents import Document

# All "source" metadata is stored relative to this root, so citations
# look like "callbacks/<file>.html" regardless of which subfolder of
# documents/latest/ was ingested.
DOCS_ROOT = Path(__file__).resolve().parent.parent / "documents" / "latest"


def load_html_folder(folder_path: str) -> list[Document]:
    """
    Load every *.html file directly inside `folder_path` (non-recursive)
    as a LangChain Document, with Sphinx boilerplate stripped out.

    Metadata per Document:
        source:    path relative to documents/latest/, e.g.
                   "callbacks/langchain.callbacks....html"
        title:     page title (without the trailing "— LangChain x.y.z")
        category:  name of the immediate parent folder, e.g. "callbacks"
        file_name: the HTML file's name
    """
    folder = Path(folder_path).resolve()
    docs: list[Document] = []

    for path in sorted(folder.glob("*.html")):
        html = path.read_text(encoding="utf-8")
        soup = BeautifulSoup(html, "lxml")

        main = soup.find("div", {"role": "main"})
        content = main if main is not None else soup

        for tag in content.find_all(["script", "style"]):
            tag.decompose()

        text = content.get_text(separator=" ", strip=True)

        title = soup.title.get_text(strip=True) if soup.title else path.stem
        title = title.split(" — ")[0]

        docs.append(Document(
            page_content=text,
            metadata={
                "source": str(path.relative_to(DOCS_ROOT)),
                "title": title,
                "category": path.parent.name,
                "file_name": path.name,
            },
        ))

    return docs
