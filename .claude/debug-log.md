# Debug Log

Running log of non-obvious bugs and their fixes. Newest entries first. When
you fix something that wasn't obvious from the code/error message alone, add
an entry here: symptom, root cause, fix (with file references).

---

## 2026-06-14 — `rag.ingest` ValueError: path not in subpath of `DOCS_ROOT`

**Symptom:** `python -m rag.ingest documents/latest/callbacks` failed with:
```
ValueError: 'documents/latest/callbacks/langchain.callbacks.streaming_aiter.AsyncIteratorCallbackHandler.html'
is not in the subpath of '/mnt/.../documents/latest' OR one path is relative
and the other is absolute.
```

**Root cause:** `rag/loader.py` builds the `source` metadata via
`path.relative_to(DOCS_ROOT)`, where `DOCS_ROOT` is an absolute path
(`Path(__file__).resolve().parent.parent / "documents" / "latest"`). The CLI
passes a relative folder path (`"documents/latest/callbacks"`), so
`folder.glob("*.html")` yielded relative `Path` objects — `relative_to()`
can't compare a relative path against an absolute one.

**Fix:** `rag/loader.py` — `load_html_folder()` now does
`folder = Path(folder_path).resolve()` before globbing, so every yielded
`path` is absolute and comparable to `DOCS_ROOT`.
