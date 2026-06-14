# main.py
# -----------------------------------------------------------
# AI Assistant Suite - Streamlit UI
# Two modes, switchable from the sidebar:
#   - SQL Assistant  -> SQLAgent  (text-to-SQL over Supabase)
#   - Docs Assistant -> RAGAgent  (RAG over LangChain docs, pgvector)
# Only file that has UI code — agent logic stays in agent/
# -----------------------------------------------------------

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st

# On Streamlit Community Cloud there is no .env file — secrets are configured
# via the app's "Secrets" dashboard instead. Accessing st.secrets (if a
# secrets.toml exists) makes Streamlit copy top-level secrets into
# os.environ, so the existing os.getenv() calls in agent/, llm/, rag/, and
# tools/ keep working unchanged both locally and when deployed.
st.secrets.load_if_toml_exists()

from agent import SQLAgent, RAGAgent
from rag.ingest import get_ingested_chunk_count, list_docs_subfolders, ingest_path

AGENT_CLASSES = {"sql": SQLAgent, "docs": RAGAgent}

# -----------------------------------------------------------
# Page config — must be first Streamlit command
# -----------------------------------------------------------
st.set_page_config(
    page_title="AI Assistant Suite",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------
# Custom CSS
# -----------------------------------------------------------
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(120deg, #10b981, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #64748b;
        margin-bottom: 2rem;
    }
    .result-card {
        padding: 1.5rem;
        border-radius: 10px;
        background: white;
        border: 1px solid #e2e8f0;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(120deg, #10b981, #3b82f6);
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        border: none;
        font-size: 1rem;
    }
    .stButton>button:hover {
        background: linear-gradient(120deg, #059669, #2563eb);
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------
# Session state
# -----------------------------------------------------------
if "app_mode" not in st.session_state:
    st.session_state.app_mode = "sql"

for _mode, _AgentClass in AGENT_CLASSES.items():
    st.session_state.setdefault(f"{_mode}_chat_history", [])
    st.session_state.setdefault(f"{_mode}_current_result", None)
    st.session_state.setdefault(f"{_mode}_query_count", 0)
    st.session_state.setdefault(f"{_mode}_provider", "openai")
    st.session_state.setdefault(f"{_mode}_last_query", "")
    if f"{_mode}_agent" not in st.session_state:
        st.session_state[f"{_mode}_agent"] = _AgentClass(provider="openai")

st.session_state.setdefault("kb_last_ingest_result", None)

# -----------------------------------------------------------
# Sidebar: mode switch
# -----------------------------------------------------------
MODE_LABELS = {"sql": "🗄️ SQL Assistant", "docs": "📚 Docs Assistant (RAG)"}

with st.sidebar:
    st.markdown("### 🧭 Mode")
    mode = st.radio(
        label="Choose assistant:",
        options=["sql", "docs"],
        format_func=lambda x: MODE_LABELS[x],
        index=list(AGENT_CLASSES).index(st.session_state.app_mode),
        key="app_mode_radio",
    )
    if mode != st.session_state.app_mode:
        st.session_state.app_mode = mode
        st.rerun()

    st.divider()
    st.markdown("### ⚙️ Settings")
    st.divider()

    # ---- Model selector (per-mode) ----
    st.markdown("### 🤖 Model")
    provider_key = f"{mode}_provider"
    agent_key = f"{mode}_agent"

    provider = st.radio(
        label="Choose LLM provider:",
        options=["openai", "anthropic"],
        format_func=lambda x: "🟢 GPT-4o (OpenAI)" if x == "openai" else "🟣 Claude (Anthropic)",
        index=0 if st.session_state[provider_key] == "openai" else 1,
        key=f"{mode}_provider_radio",
    )

    if provider != st.session_state[provider_key]:
        st.session_state[provider_key] = provider
        st.session_state[agent_key] = AGENT_CLASSES[mode](provider=provider)
        st.success(f"Switched to {provider}!")

    st.divider()

    # ---- Current model info ----
    st.markdown("### 📋 Current Model")
    st.code(st.session_state[agent_key].get_model_name())

    if mode == "docs":
        st.caption(
            "ℹ️ Embeddings always use OpenAI (text-embedding-3-small), "
            "regardless of the chat model above."
        )

    st.divider()

    # ---- Data source info (mode-aware) ----
    if mode == "sql":
        st.markdown("### 🗄️ Database")
        st.info("Connected to Supabase\nProject: unsigymegludckhspfpm\neu-central-1")
    else:
        st.markdown("### 📚 Knowledge Source")
        st.info("pgvector store on Supabase\nProject: pmmplfxyogjssefzstiw\neu-central-1")

    st.divider()

    # ---- Statistics (per-mode) ----
    st.markdown("### 📈 Statistics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Queries", st.session_state[f"{mode}_query_count"])
    with col2:
        st.metric("History", len(st.session_state[f"{mode}_chat_history"]) // 2)

    st.divider()

    # ---- Clear chat (per-mode) ----
    if st.button("🗑️ Clear Chat", key=f"{mode}_clear_chat"):
        st.session_state[f"{mode}_chat_history"] = []
        st.session_state[f"{mode}_current_result"] = None
        st.session_state[f"{mode}_query_count"] = 0
        st.rerun()

    # ---- Knowledge Base panel (Docs mode only) ----
    if mode == "docs":
        st.divider()
        st.markdown("### 📚 Knowledge Base")

        st.metric("Ingested chunks", get_ingested_chunk_count())

        folders = list_docs_subfolders()
        if folders:
            selected_folder = st.selectbox(
                "Folder to ingest:", options=folders, key="kb_selected_folder"
            )

            if st.button("📥 Ingest selected folder"):
                with st.spinner(f"Ingesting documents/latest/{selected_folder}..."):
                    try:
                        result = ingest_path(f"documents/latest/{selected_folder}")
                        st.session_state.kb_last_ingest_result = result
                    except Exception as e:
                        st.session_state.kb_last_ingest_result = {"error": str(e)}
                st.rerun()
        else:
            st.caption("No subfolders found under documents/latest/")

        if st.session_state.kb_last_ingest_result:
            result = st.session_state.kb_last_ingest_result
            if "error" in result:
                st.error(f"Ingestion failed: {result['error']}")
            else:
                st.success(
                    f"Ingested {result['files_loaded']} file(s) from "
                    f"`{result['folder']}` → {result['chunks_inserted']} chunk(s) added"
                )

# -----------------------------------------------------------
# Header (mode-aware)
# -----------------------------------------------------------
if mode == "sql":
    st.markdown('<div class="main-header">🗄️ SQLSpeak</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Converts your questions into SQL queries and retrieves the results from Supabase</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="main-header">📚 Docs Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Ask questions about LangChain — answered from your ingested documentation</div>', unsafe_allow_html=True)

# -----------------------------------------------------------
# Example questions (mode-aware)
# -----------------------------------------------------------
st.markdown("---")
with st.expander("💡 Example Questions & Tips", expanded=False):
    if mode == "sql":
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**📋 Explore**")
            st.code("What tables exist in the database?")
            st.code("Show me the schema of all tables")
        with col2:
            st.markdown("**🔍 Query**")
            st.code("Show me the first 5 rows of each table")
            st.code("How many records are in each table?")
        with col3:
            st.markdown("**📊 Analyze**")
            st.code("Which table has the most records?")
            st.code("Show me all columns that store dates")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**📚 Callbacks**")
            st.code("What does AsyncIteratorCallbackHandler do?")
            st.code("How do I log callbacks with LoggingCallbackHandler?")
        with col2:
            st.markdown("**🧠 General**")
            st.code("What is the purpose of a callback handler in LangChain?")
            st.code("How can I stream tokens as they're generated?")
        st.caption(
            "ℹ️ Answers are grounded only in documents you've ingested via the "
            "Knowledge Base panel in the sidebar. If the knowledge base is "
            "empty, ingest a folder first."
        )

# -----------------------------------------------------------
# Chat history (mode-aware)
# -----------------------------------------------------------
st.markdown("---")
st.markdown("### 💬 Chat")

chat_history_key = f"{mode}_chat_history"
agent = st.session_state[f"{mode}_agent"]
assistant_avatar = "🗄️" if mode == "sql" else "📚"

for message in st.session_state[chat_history_key]:
    role = message["role"]
    content = message["content"]

    if role == "user":
        with st.chat_message("user"):
            st.write(content)

    elif role == "assistant":
        with st.chat_message("assistant", avatar=assistant_avatar):
            parts = content.split("```")
            for i, part in enumerate(parts):
                if i % 2 == 0:
                    if part.strip():
                        st.markdown(part)
                else:
                    lines = part.strip().split("\n")
                    language = lines[0].lower() if lines else ""
                    if language in ["sql", "postgresql"]:
                        sql_code = "\n".join(lines[1:]) if len(lines) > 1 else part
                        st.code(sql_code, language="sql")
                    else:
                        st.code(part)

# -----------------------------------------------------------
# Query input — pinned to the bottom, submits on Enter
# -----------------------------------------------------------
placeholder = "Ask your request..." if mode == "sql" else "Ask a question about LangChain..."
user_query = st.chat_input(placeholder, key=f"{mode}_chat_input")

# -----------------------------------------------------------
# Process query
# -----------------------------------------------------------
if user_query:
    with st.spinner("🤖 Thinking..."):
        try:
            progress_bar = st.progress(0)
            status_text = st.empty()

            if mode == "sql":
                status_text.text("🔌 Connecting to Supabase...")
                progress_bar.progress(25)
                status_text.text("🤖 Agent is exploring the database...")
            else:
                status_text.text("🔌 Connecting to knowledge base...")
                progress_bar.progress(25)
                status_text.text("🤖 Agent is searching the documentation...")
            progress_bar.progress(50)

            # Run the agent
            response = agent.run(
                user_message=user_query,
                chat_history=st.session_state[chat_history_key],
            )

            status_text.text("📊 Processing results...")
            progress_bar.progress(75)

            # Save to history
            st.session_state[chat_history_key].append({
                "role": "user",
                "content": user_query,
            })
            st.session_state[chat_history_key].append({
                "role": "assistant",
                "content": response,
            })

            st.session_state[f"{mode}_current_result"] = response
            st.session_state[f"{mode}_last_query"] = user_query
            st.session_state[f"{mode}_query_count"] += 1

            progress_bar.progress(100)
            status_text.text("✅ Done!")

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            if mode == "sql":
                st.info("💡 **Troubleshooting:**\n- Check your .env keys\n- Make sure Supabase is reachable\n- Check the terminal for detailed errors")
            else:
                st.info("💡 **Troubleshooting:**\n- Check your .env keys (OPENAI_API_KEY is required for embeddings)\n- Make sure the pgvector database is reachable\n- Try ingesting a documents/latest/ folder via the Knowledge Base panel")

    st.rerun()

# -----------------------------------------------------------
# Result actions
# -----------------------------------------------------------
current_result_key = f"{mode}_current_result"
if st.session_state[current_result_key]:
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📋 Copy Result", key=f"{mode}_copy"):
            st.code(st.session_state[current_result_key])
    with col2:
        if st.button("💾 Save to File", key=f"{mode}_save"):
            from datetime import datetime
            prefix = "sqlspeak" if mode == "sql" else "docs_assistant"
            filename = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(filename, "w") as f:
                f.write(f"Query: {st.session_state[f'{mode}_last_query']}\n\n")
                f.write(f"Result:\n{st.session_state[current_result_key]}")
            st.success(f"✅ Saved to {filename}")
    with col3:
        if st.button("🔄 New Query", key=f"{mode}_new_query"):
            st.session_state[current_result_key] = None
            st.rerun()

# -----------------------------------------------------------
# Footer
# -----------------------------------------------------------
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**🗄️ SQL Assistant**")
    st.markdown("Supabase • PostgreSQL")
with col2:
    st.markdown("**📚 Docs Assistant**")
    st.markdown("pgvector • RAG • LangChain docs")
with col3:
    st.markdown("**🤖 Powered by**")
    st.markdown("Claude • GPT-4o • LangChain")
