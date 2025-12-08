import asyncio
import os
import queue
import threading
import time
from datetime import datetime
from typing import List

import pandas as pd
import requests
import streamlit as st
import websockets
from dotenv import load_dotenv

load_dotenv()

# API configuration
API_BASE_URL = os.getenv("API_BASE_URL")

# Import components
from components import (
    chunking_settings,
    collection_card,
    collection_selector,
    file_uploader_with_preview,
)

# Page configuration
st.set_page_config(
    page_title="RAG System - Collections",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .tab-header {
        font-size: 1.8rem;
        color: #43A047;
        margin-top: 1rem;
        margin-bottom: 1.5rem;
    }
    .collection-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    .collection-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.2rem;
        border: 1px solid #e0e0e0;
        transition: transform 0.2s;
    }
    .collection-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .source-card {
        background-color: #f5f5f5;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.8rem;
        border-left: 4px solid #1E88E5;
    }
    .answer-box {
        background-color: #E8F5E8;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .ingestion-success {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""",
    unsafe_allow_html=True,
)


def check_api_health():
    """Check if API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        return response.status_code == 200, response.json() if response.ok else {}
    except:
        return False, {}


def get_collections():
    """Get list of collections from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/collections")
        if response.status_code == 200:
            return response.json()
        return {"collections": []}
    except:
        return {"collections": []}


def upload_to_collection(
    collection_name: str, files: List, chunk_size: int, chunk_overlap: int
):
    """Upload files to a specific collection"""
    if not files:
        return {"success": False, "error": "No files selected"}

    try:
        files_data = []
        for file in files:
            files_data.append(
                (
                    "files",
                    (
                        file.name,
                        file.getvalue(),
                        file.type or "application/octet-stream",
                    ),
                )
            )

        response = requests.post(
            f"{API_BASE_URL}/ingest",
            files=files_data,
            data={
                "collection_name": collection_name,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
            },
        )

        if response.status_code == 200:
            return response.json()
        else:
            return {"success": False, "error": f"Upload failed: {response.text}"}

    except Exception as e:
        return {"success": False, "error": str(e)}


def query_collection(question: str, collection_name: str, top_k: int = 4):
    """Query a specific collection"""
    payload = {"question": question, "collection_name": collection_name, "top_k": top_k}

    try:
        response = requests.post(
            f"{API_BASE_URL}/query",
            json=payload,
            headers={"Content-Type": "application/json"},
        )

        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API Error: {response.text}"}

    except Exception as e:
        return {"error": str(e)}


def set_current_collection(collection_name: str):
    """Set the current collection"""
    try:
        response = requests.put(
            f"{API_BASE_URL}/collection/current",
            params={"collection_name": collection_name},
        )
        return response.status_code == 200
    except:
        return False


def create_new_collection(collection_name: str):
    """Create a new collection"""
    try:
        response = requests.post(f"{API_BASE_URL}/collections/{collection_name}")
        return response.status_code == 200
    except:
        return False


def render_ingestion_tab():
    """Render the ingestion tab"""
    st.markdown(
        '<h2 class="tab-header">📤 Document Ingestion</h2>', unsafe_allow_html=True
    )

    # Get current collections
    collections_data = get_collections()
    collections = collections_data.get("collections", [])

    # Create two columns: collections list and ingestion form
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### 📚 Collections")

        # Display collections
        if collections:
            for collection in collections:
                with st.container():
                    col_name = collection_card(collection)

                    # Select button
                    if st.button(
                        f"Select {col_name}",
                        key=f"select_{col_name}",
                        width="stretch",
                    ):
                        st.session_state.selected_collection = col_name
                        st.rerun()
        else:
            st.info("No collections yet. Create one below!")

        # Create new collection
        st.markdown("---")
        st.markdown("### 🆕 Create Collection")

        new_collection = st.text_input("New collection name")
        if st.button("Create Collection", width="stretch"):
            if new_collection and new_collection.strip():
                if create_new_collection(new_collection.strip()):
                    st.success(f"Collection '{new_collection}' created!")
                    st.rerun()
                else:
                    st.error("Failed to create collection")
            else:
                st.warning("Please enter a collection name")

    with col2:
        st.markdown("### 📤 Upload Documents")

        # Get selected collection from session state or use first available
        selected_collection = st.session_state.get("selected_collection")

        if selected_collection:
            st.success(f"**Selected Collection:** `{selected_collection}`")
        else:
            if collections:
                selected_collection = collections[0]["name"]
                st.session_state.selected_collection = selected_collection
            else:
                st.warning("Please create or select a collection first")
                return

        # File upload section
        st.markdown("#### 1. Select Files")

        # Get supported extensions from API health
        health_status, health_data = check_api_health()
        supported_extensions = [".pdf", ".txt", ".md", ".docx", ".pptx", ".csv"]

        uploaded_files = file_uploader_with_preview(supported_extensions)

        if uploaded_files:
            # Chunking settings
            st.markdown("#### 2. Configure Chunking")
            chunk_size, chunk_overlap = chunking_settings()

            # Upload button
            st.markdown("#### 3. Ingest")

            col1, col2 = st.columns([3, 1])
            with col2:
                if uploaded_files and st.button(
                    "🚀 Ingest Documents", type="primary", width="stretch"
                ):
                    with st.spinner(
                        f"Starting ingestion for '{selected_collection}'..."
                    ):
                        result = upload_to_collection(
                            collection_name=selected_collection,
                            files=uploaded_files,
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap,
                        )

                        if "task_id" in result:
                            st.success(
                                f"✅ Ingestion started! Task ID: `{result['task_id']}`"
                            )
                            st.info(f"Track progress in the 📊 Task Monitoring tab")

                            # Show task link
                            st.markdown(
                                f"[View Task Details](#task-{result['task_id']})"
                            )

                            # Auto-refresh to show task in monitoring tab
                            time.sleep(2)
                            st.rerun()
                        else:
                            st.error(
                                f"❌ Failed to start ingestion: {result.get('error', 'Unknown error')}"
                            )


def render_query_tab():
    """Render the query tab"""
    st.markdown(
        '<h2 class="tab-header">💬 Query Collections</h2>', unsafe_allow_html=True
    )

    # Get collections
    collections_data = get_collections()
    collections = collections_data.get("collections", [])

    if not collections:
        st.warning(
            "No collections available. Please create and ingest documents in the Ingestion tab first."
        )
        return

    # Collection selection
    st.markdown("### 1. Select Collection to Query")

    collection_names = [c["name"] for c in collections]
    current_collection = collections_data.get("current_collection")

    # Two-column layout for collection selection
    col1, col2 = st.columns([3, 1])

    with col1:
        selected_query_collection = st.selectbox(
            "Choose collection",
            options=collection_names,
            index=collection_names.index(current_collection)
            if current_collection in collection_names
            else 0,
            key="query_collection_select",
        )

    with col2:
        if st.button("Set as Current", width="stretch"):
            if set_current_collection(selected_query_collection):
                st.success(f"Set '{selected_query_collection}' as current collection!")
                st.rerun()
            else:
                st.error("Failed to set collection")

    # Show collection info
    selected_collection_info = next(
        (c for c in collections if c["name"] == selected_query_collection), None
    )

    if selected_collection_info:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Files", selected_collection_info["file_count"])
        with col2:
            st.metric("Indexed Documents", selected_collection_info["vector_count"])
        with col3:
            status = (
                "✅ Indexed"
                if selected_collection_info["vector_exists"]
                else "⏳ Not Indexed"
            )
            st.metric("Status", status)

    # Query interface
    st.markdown("### 2. Ask Questions")

    # Initialize chat history
    if "query_messages" not in st.session_state:
        st.session_state.query_messages = []

    if "query_history" not in st.session_state:
        st.session_state.query_history = []

    # Display chat history for current collection
    for message in st.session_state.query_messages:
        if message.get("collection") == selected_query_collection:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

                if message.get("sources"):
                    with st.expander(
                        f"📚 View Sources ({len(message['sources'])} documents)"
                    ):
                        for i, source in enumerate(message["sources"], 1):
                            st.markdown(
                                f"**Source {i}:** {source.get('source', 'Unknown')}"
                            )
                            st.markdown(
                                f"**Collection:** {source.get('collection', 'Unknown')}"
                            )
                            st.markdown(
                                f"**Content:** {source.get('content', '')[:250]}..."
                            )

    # Query input
    query = st.chat_input(f"Ask about documents in '{selected_query_collection}'...")

    if query:
        # Add to chat history
        st.session_state.query_messages.append(
            {"role": "user", "content": query, "collection": selected_query_collection}
        )

        # Display user message
        with st.chat_message("user"):
            st.markdown(query)

        # Get response
        with st.chat_message("assistant"):
            with st.spinner(f"Searching in '{selected_query_collection}'..."):
                result = query_collection(query, selected_query_collection)

                if "error" in result:
                    st.error(f"Error: {result['error']}")
                else:
                    # Display answer
                    st.markdown("### Answer")
                    st.markdown(
                        f'<div class="answer-box">{result["answer"]}</div>',
                        unsafe_allow_html=True,
                    )

                    # Display sources
                    if result.get("sources"):
                        st.markdown(
                            f"### 📚 Sources ({len(result['sources'])} documents found)"
                        )

                        for i, source in enumerate(result["sources"], 1):
                            with st.expander(
                                f"Source {i}: {source.get('source', 'Unknown')}"
                            ):
                                st.markdown(
                                    f"**Collection:** {source.get('collection', selected_query_collection)}"
                                )
                                if source.get("score"):
                                    st.markdown(
                                        f"**Relevance Score:** {source['score']:.3f}"
                                    )
                                st.markdown(f"**Content:**")
                                st.markdown(source["content"])

                    # Add to session history
                    st.session_state.query_messages.append(
                        {
                            "role": "assistant",
                            "content": result["answer"],
                            "sources": result.get("sources", []),
                            "collection": selected_query_collection,
                        }
                    )

                    st.session_state.query_history.append(
                        {
                            "collection": selected_query_collection,
                            "question": query,
                            "answer": result["answer"],
                            "timestamp": time.time(),
                        }
                    )

    # Query history sidebar
    with st.sidebar:
        if st.session_state.query_history:
            st.markdown("### 📜 Query History")

            # Filter by current collection
            current_collection_history = [
                q
                for q in st.session_state.query_history
                if q["collection"] == selected_query_collection
            ]

            if current_collection_history:
                for i, query_item in enumerate(
                    reversed(current_collection_history[-5:]), 1
                ):
                    with st.expander(f"Query {i}"):
                        st.markdown(f"**Q:** {query_item['question'][:50]}...")
                        st.markdown(f"**A:** {query_item['answer'][:100]}...")

                        if st.button(
                            f"Load {i}", key=f"load_{i}_{selected_query_collection}"
                        ):
                            st.session_state.query_messages.append(
                                {
                                    "role": "user",
                                    "content": query_item["question"],
                                    "collection": selected_query_collection,
                                }
                            )
                            st.session_state.query_messages.append(
                                {
                                    "role": "assistant",
                                    "content": query_item["answer"],
                                    "collection": selected_query_collection,
                                }
                            )
                            st.rerun()
            else:
                st.info(f"No queries yet for {selected_query_collection}")


def render_task_monitoring_tab():
    """Render task monitoring tab"""
    st.markdown(
        '<h2 class="tab-header">📊 Task Monitoring</h2>', unsafe_allow_html=True
    )

    # Task stats
    try:
        response = requests.get(f"{API_BASE_URL}/ingest/tasks/stats")
        if response.status_code == 200:
            stats = response.json()

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Tasks", stats["total_tasks"])
            with col2:
                st.metric("Completed", stats["completed_tasks"])
            with col3:
                st.metric("Failed", stats["failed_tasks"])
            with col4:
                st.metric("Active", stats["active_tasks"] + stats["processing_tasks"])
    except:
        pass

    # Filter options
    col1, col2, col3 = st.columns(3)
    with col1:
        status_filter = st.selectbox(
            "Status Filter",
            ["all", "started", "processing", "completed", "failed", "cancelled"],
            key="status_filter",
        )
    with col2:
        collection_filter = st.text_input("Collection Filter", key="collection_filter")
    with col3:
        limit = st.number_input("Tasks per page", min_value=1, max_value=100, value=20)

    # Load tasks
    try:
        params = {"limit": limit}
        if status_filter != "all":
            params["status"] = status_filter
        if collection_filter:
            params["collection_name"] = collection_filter

        response = requests.get(
            f"{API_BASE_URL}/ingest/tasks", params=params, timeout=10
        )

        if response.status_code == 200:
            data = response.json()
            tasks = data["tasks"]

            if tasks:
                # Create DataFrame for display
                task_data = []
                for task in tasks:
                    created = datetime.fromisoformat(
                        task["created_at"].replace("Z", "+00:00")
                    )
                    updated = datetime.fromisoformat(
                        task["updated_at"].replace("Z", "+00:00")
                    )

                    task_data.append(
                        {
                            "Task ID": task["task_id"][:8] + "...",
                            "Collection": task["collection_name"],
                            "Status": task["status"],
                            "Progress": f"{task.get('progress', 0)}%",
                            "Files": f"{task.get('files_processed', 0)}/{task.get('total_files', 0)}",
                            "Chunks": task.get("chunks_created", 0),
                            "Created": created.strftime("%Y-%m-%d %H:%M"),
                            "Duration": str(updated - created).split(".")[0]
                            if task.get("completed_at")
                            else "Running",
                        }
                    )

                # Display table
                df = pd.DataFrame(task_data)
                st.dataframe(df, width="stretch", hide_index=True)

                # Task details section
                st.markdown("### Task Details")
                selected_task_id = st.selectbox(
                    "Select Task to View Details",
                    [task["task_id"] for task in tasks],
                    format_func=lambda x: f"{x[:8]}... - {next(t['collection_name'] for t in tasks if t['task_id'] == x)}",
                )

                if selected_task_id:
                    task_response = requests.get(
                        f"{API_BASE_URL}/ingest/tasks/{selected_task_id}", timeout=5
                    )
                    if task_response.status_code == 200:
                        task_details = task_response.json()

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Status", task_details["status"].upper())
                        with col2:
                            st.metric("Progress", f"{task_details.get('progress', 0)}%")
                        with col3:
                            st.metric(
                                "Files",
                                f"{task_details.get('files_processed', 0)}/{task_details.get('total_files', 0)}",
                            )

                        # Display files
                        if task_details.get("files"):
                            st.markdown("#### Files")
                            files_df = pd.DataFrame(task_details["files"])
                            st.dataframe(files_df, width="stretch", hide_index=True)

                        # Display metadata
                        if task_details.get("metadata"):
                            with st.expander("View Metadata"):
                                st.json(task_details["metadata"])

                        # Action buttons
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("🔄 Refresh Task", width="stretch"):
                                st.rerun()
                        with col2:
                            if task_details["status"] in ["started", "processing"]:
                                if st.button(
                                    "❌ Cancel Task",
                                    type="secondary",
                                    width="stretch",
                                ):
                                    response = requests.delete(
                                        f"{API_BASE_URL}/ingest/tasks/{selected_task_id}",
                                        timeout=5,
                                    )
                                    if response.status_code == 200:
                                        st.success("Task cancelled!")
                                        time.sleep(1)
                                        st.rerun()
            else:
                st.info("No tasks found matching your filters.")

    except Exception as e:
        st.error(f"Error loading tasks: {e}")


def render_dashboard_tab():
    """Render the dashboard tab"""
    st.markdown(
        '<h2 class="tab-header">📊 System Dashboard</h2>', unsafe_allow_html=True
    )

    # Check API health
    health_status, health_data = check_api_health()

    if not health_status:
        st.error("❌ API is not reachable")
        st.info(f"Make sure the API is running at {API_BASE_URL}")
        return

    # System overview
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("API Status", "✅ Healthy" if health_status else "❌ Unhealthy")

    with col2:
        st.metric("Collections", len(health_data.get("available_collections", [])))

    with col3:
        current = health_data.get("current_collection", "None")
        st.metric("Current Collection", current)

    # Collections overview
    st.markdown("### 📚 Collections Overview")

    collections_data = get_collections()
    collections = collections_data.get("collections", [])

    if collections:
        # Create a DataFrame for better visualization
        collection_stats = []
        for collection in collections:
            collection_stats.append(
                {
                    "Collection": collection["name"],
                    "Files": collection["file_count"],
                    "Indexed Docs": collection["vector_count"],
                    "Status": "✅ Indexed"
                    if collection["vector_exists"]
                    else "⏳ Pending",
                    "Path": collection["data_path"].split("/")[-1],
                }
            )

        df = pd.DataFrame(collection_stats)
        st.dataframe(df, width="stretch")

        # Visualization
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Documents per Collection")
            chart_data = df[["Collection", "Indexed Docs"]].set_index("Collection")
            st.bar_chart(chart_data)

        with col2:
            st.markdown("#### Collection Status")
            status_counts = df["Status"].value_counts()
            st.dataframe(status_counts)
    else:
        st.info("No collections available. Go to the Ingestion tab to create one.")

    # System information
    st.markdown("### ⚙️ System Information")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**API Information**")
        st.json(
            {
                "version": health_data.get("api_version", "Unknown"),
                "embedding_model": health_data.get("embedding_model", "Unknown"),
                "current_collection": health_data.get("current_collection", "None"),
            }
        )

    with col2:
        st.markdown("**Directory Structure**")
        st.code("""
        rag-system/
        ├── data/collections/    # Collection-based documents
        │   ├── collection_1/
        │   ├── collection_2/
        │   └── ...
        ├── vector_store/        # Vector embeddings
        │   └── chroma_db/
        │       ├── collection_1/
        │       ├── collection_2/
        │       └── ...
        └── ...
        """)


def main():
    # Header
    st.markdown(
        '<h1 class="main-header">📚 Multi-Collection RAG System</h1>',
        unsafe_allow_html=True,
    )

    st.markdown("""
    This system organizes documents into collections. Each collection has its own vector store
    and can be queried independently. Upload documents to collections in the **Ingestion** tab,
    then query them in the **Query** tab.
    """)

    # Sidebar - API Status
    with st.sidebar:
        st.markdown("### 🔌 API Status")

        health_status, health_data = check_api_health()

        if health_status:
            st.success("✅ API Connected")

            # Quick stats
            st.markdown(
                f"**Collections:** {len(health_data.get('available_collections', []))}"
            )
            st.markdown(f"**Current:** {health_data.get('current_collection', 'None')}")
            st.markdown(
                f"**Embeddings:** {health_data.get('embedding_model', 'Unknown')}"
            )
        else:
            st.error("❌ API Disconnected")
            st.info(f"Run: `uvicorn api.main:app --reload`")

        st.markdown("---")
        st.markdown("### 🚀 Quick Actions")

        if st.button("Refresh All", width="stretch"):
            st.rerun()

        if st.button("Clear Chat History", width="stretch"):
            if "query_messages" in st.session_state:
                st.session_state.query_messages = []
            st.rerun()

        st.markdown("---")
        st.markdown("### 📖 How to Use")

        with st.expander("Instructions"):
            st.markdown("""
            1. **Ingestion Tab**: Create collections and upload documents
            2. **Query Tab**: Select a collection and ask questions
            3. **Dashboard**: View system statistics and collection info

            **Features:**
            - Multiple collections
            - Configurable chunking
            - Collection-based querying
            - Document source tracking
            """)

    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📤 Ingestion", "💬 Query", "📊 Task Monitoring", "📊 Dashboard"]
    )

    with tab1:
        render_ingestion_tab()

    with tab2:
        render_query_tab()

    with tab3:
        render_task_monitoring_tab()

    with tab4:
        render_dashboard_tab()


if __name__ == "__main__":
    # Initialize session state
    if "selected_collection" not in st.session_state:
        st.session_state.selected_collection = None
    if "query_messages" not in st.session_state:
        st.session_state.query_messages = []
    if "query_history" not in st.session_state:
        st.session_state.query_history = []

    main()
