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
    evaluation_metrics_row,
    evaluation_summary_table,
    file_uploader_with_preview,
    metric_explanation,
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
    .eval-score-good {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
    }
    .eval-score-fair {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
    }
    .eval-score-poor {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border: 1px solid #dee2e6;
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


def query_collection(
    question: str, collection_name: str, top_k: int = 4, evaluate: bool = False
):
    """Query a specific collection with optional evaluation"""
    if evaluate:
        # Use query/evaluate endpoint
        payload = {
            "question": question,
            "collection_name": collection_name,
            "top_k": top_k,
            "evaluate": True,
        }
        endpoint = f"{API_BASE_URL}/query/evaluate"
    else:
        # Use regular query endpoint
        payload = {
            "question": question,
            "collection_name": collection_name,
            "top_k": top_k,
        }
        endpoint = f"{API_BASE_URL}/query"

    try:
        response = requests.post(
            endpoint,
            json=payload,
            headers={"Content-Type": "application/json"},
        )

        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API Error: {response.text}"}

    except Exception as e:
        return {"error": str(e)}


def evaluate_response(
    question: str, answer: str, contexts: list, reference_answer: str = None
):
    """Evaluate a response using RAGAS metrics"""
    payload = {"question": question, "answer": answer, "contexts": contexts}

    if reference_answer:
        payload["reference_answer"] = reference_answer

    try:
        response = requests.post(
            f"{API_BASE_URL}/evaluate",
            json=payload,
            headers={"Content-Type": "application/json"},
        )

        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API Error: {response.text}"}

    except Exception as e:
        return {"error": str(e)}


def batch_evaluate(items: list, push_to_langfuse: bool = True):
    """Batch evaluate multiple items"""
    payload = {"items": items, "push_to_langfuse": push_to_langfuse}

    try:
        response = requests.post(
            f"{API_BASE_URL}/evaluate/batch",
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


def render_evaluation_tab():
    """Render the evaluation tab"""
    st.markdown(
        '<h2 class="tab-header">📊 Evaluation & Testing</h2>', unsafe_allow_html=True
    )

    st.markdown("""
    Evaluate your RAG system's performance using RAGAS metrics. Test individual queries or batch evaluate multiple questions.
    """)

    # Evaluation mode selector
    eval_mode = st.radio(
        "Evaluation Mode",
        ["Single Query Evaluation", "Query with Evaluation", "Batch Evaluation"],
        horizontal=True,
    )

    if eval_mode == "Single Query Evaluation":
        st.markdown("### Evaluate an Existing Response")
        st.info("Manually enter a question, answer, and contexts to evaluate quality.")

        question = st.text_input("Question", placeholder="What is machine learning?")
        answer = st.text_area(
            "Answer", placeholder="Machine learning is...", height=100
        )

        st.markdown("**Contexts (Retrieved Documents)**")
        num_contexts = st.number_input(
            "Number of contexts", min_value=1, max_value=10, value=2
        )

        contexts = []
        for i in range(num_contexts):
            context = st.text_area(
                f"Context {i + 1}",
                placeholder=f"Enter context paragraph {i + 1}...",
                height=80,
                key=f"context_{i}",
            )
            if context:
                contexts.append(context)

        if st.button("🎯 Evaluate Response", type="primary"):
            if question and answer and contexts:
                with st.spinner("Evaluating response..."):
                    result = evaluate_response(question, answer, contexts)

                    if "error" in result:
                        st.error(f"Error: {result['error']}")
                    else:
                        st.success("✅ Evaluation Complete!")

                        # Display scores
                        st.markdown("### 📊 Evaluation Scores")

                        scores = result.get("scores", {})

                        # Create metrics row
                        cols = st.columns(len(scores))
                        for idx, (metric_name, score) in enumerate(scores.items()):
                            with cols[idx]:
                                if score is not None:
                                    # Color code based on score
                                    if score >= 0.8:
                                        st.metric(
                                            metric_name.replace("_", " ").title(),
                                            f"{score:.3f}",
                                            delta="Good",
                                            delta_color="normal",
                                        )
                                    elif score >= 0.6:
                                        st.metric(
                                            metric_name.replace("_", " ").title(),
                                            f"{score:.3f}",
                                            delta="Fair",
                                            delta_color="off",
                                        )
                                    else:
                                        st.metric(
                                            metric_name.replace("_", " ").title(),
                                            f"{score:.3f}",
                                            delta="Poor",
                                            delta_color="inverse",
                                        )
                                else:
                                    st.metric(
                                        metric_name.replace("_", " ").title(), "N/A"
                                    )

                        # Score interpretation
                        st.markdown("### 📖 Score Interpretation")
                        with st.expander("What do these scores mean?"):
                            st.markdown("""
                            **Faithfulness (0-1):** Measures if the answer is factually consistent with the contexts.
                            - Good: > 0.8
                            - Fair: 0.6 - 0.8
                            - Poor: < 0.6

                            **Context Precision (0-1):** Measures if relevant contexts are ranked higher.
                            - Good: > 0.8
                            - Fair: 0.6 - 0.8
                            - Poor: < 0.6

                            **LLM Context Precision (0-1):** Evaluates retrieval quality without reference.
                            - Good: > 0.75
                            - Fair: 0.6 - 0.75
                            - Poor: < 0.6

                            ⚠️ **Note:** AnswerRelevancy requires Ollama LLM for evaluation.
                            """)
            else:
                st.warning(
                    "Please fill in all fields (question, answer, and at least one context)"
                )

    elif eval_mode == "Query with Evaluation":
        st.markdown("### Query and Evaluate in Real-Time")
        st.info("Query a collection and automatically evaluate the response quality.")

        # Get collections
        collections_data = get_collections()
        collections = collections_data.get("collections", [])

        if not collections:
            st.warning(
                "No collections available. Please create and ingest documents first."
            )
            return

        collection_names = [c["name"] for c in collections]
        selected_collection = st.selectbox(
            "Select Collection", options=collection_names, key="eval_query_collection"
        )

        question = st.text_input(
            "Ask a question",
            placeholder="What is retrieval augmented generation?",
            key="eval_question",
        )

        if st.button("🔍 Query & Evaluate", type="primary"):
            if question:
                with st.spinner("Querying and evaluating..."):
                    result = query_collection(
                        question, selected_collection, evaluate=True
                    )

                    if "error" in result:
                        st.error(f"Error: {result['error']}")
                    else:
                        # Display answer
                        st.markdown("### 💬 Answer")
                        st.markdown(
                            f'<div class="answer-box">{result["answer"]}</div>',
                            unsafe_allow_html=True,
                        )

                        # Display evaluation scores
                        if result.get("evaluation_scores"):
                            st.markdown("### 📊 Evaluation Scores")

                            scores = result["evaluation_scores"]
                            cols = st.columns(len(scores))

                            for idx, (metric_name, score) in enumerate(scores.items()):
                                with cols[idx]:
                                    if score is not None:
                                        color = (
                                            "🟢"
                                            if score >= 0.8
                                            else "🟡"
                                            if score >= 0.6
                                            else "🔴"
                                        )
                                        st.metric(
                                            f"{color} {metric_name.replace('_', ' ').title()}",
                                            f"{score:.3f}",
                                        )

                        # Display sources
                        if result.get("sources"):
                            st.markdown(
                                f"### 📚 Sources ({len(result['sources'])} documents)"
                            )
                            for i, source in enumerate(result["sources"], 1):
                                with st.expander(
                                    f"Source {i}: {source.get('source', 'Unknown')}"
                                ):
                                    st.markdown(
                                        f"**Content:** {source['content'][:300]}..."
                                    )

                        # Processing time
                        if result.get("processing_time"):
                            st.caption(
                                f"⏱️ Processing time: {result['processing_time']:.2f}s"
                            )
            else:
                st.warning("Please enter a question")

    else:  # Batch Evaluation
        st.markdown("### Batch Evaluation")
        st.info(
            "Evaluate multiple question-answer pairs at once. Upload a CSV or enter test cases manually."
        )

        batch_input_method = st.radio(
            "Input Method", ["Manual Entry", "Upload CSV"], horizontal=True
        )

        if batch_input_method == "Manual Entry":
            st.markdown("**Enter Test Cases**")

            if "test_cases" not in st.session_state:
                st.session_state.test_cases = []

            with st.form("add_test_case"):
                col1, col2 = st.columns(2)
                with col1:
                    test_question = st.text_input("Question")
                    test_answer = st.text_area("Answer", height=100)
                with col2:
                    test_context = st.text_area("Context", height=150)

                if st.form_submit_button("Add Test Case"):
                    if test_question and test_answer and test_context:
                        st.session_state.test_cases.append(
                            {
                                "question": test_question,
                                "answer": test_answer,
                                "contexts": [test_context],
                            }
                        )
                        st.success("Test case added!")
                        st.rerun()

            if st.session_state.test_cases:
                st.markdown(
                    f"**Current Test Cases: {len(st.session_state.test_cases)}**"
                )

                for i, case in enumerate(st.session_state.test_cases):
                    with st.expander(f"Test Case {i + 1}: {case['question'][:50]}..."):
                        st.markdown(f"**Q:** {case['question']}")
                        st.markdown(f"**A:** {case['answer'][:100]}...")
                        if st.button(f"Remove", key=f"remove_{i}"):
                            st.session_state.test_cases.pop(i)
                            st.rerun()

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🎯 Evaluate All", type="primary"):
                        st.info(
                            "Batch evaluation via API coming soon! Use CLI tool for now."
                        )
                        st.code(
                            "python evaluation/run_evaluation.py --mode file --file test_data.json"
                        )
                with col2:
                    if st.button("Clear All"):
                        st.session_state.test_cases = []
                        st.rerun()

        else:  # Upload CSV
            st.markdown("**Upload Test Dataset**")
            st.info("CSV should have columns: question, answer, contexts")

            uploaded_file = st.file_uploader("Choose CSV file", type=["csv"])

            if uploaded_file:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.dataframe(df.head())

                    st.info(
                        f"Found {len(df)} test cases. Batch evaluation via API coming soon!"
                    )
                    st.code(
                        "python evaluation/run_evaluation.py --mode file --file your_data.csv"
                    )
                except Exception as e:
                    st.error(f"Error reading CSV: {e}")


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

    # Evaluation toggle
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("")
    with col2:
        enable_evaluation = st.checkbox(
            "📊 Enable Evaluation",
            help="Evaluate response quality with RAGAS metrics (slower)",
        )

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
            spinner_text = f"Searching in '{selected_query_collection}'..."
            if enable_evaluation:
                spinner_text += " and evaluating..."

            with st.spinner(spinner_text):
                result = query_collection(
                    query, selected_query_collection, evaluate=enable_evaluation
                )

                if result.get("error"):
                    st.error(f"Error: {result['error']}")
                else:
                    # Display answer
                    st.markdown("### Answer")
                    st.markdown(
                        f'<div class="answer-box">{result["answer"]}</div>',
                        unsafe_allow_html=True,
                    )

                    # Display evaluation scores if enabled
                    if enable_evaluation and result.get("evaluation_scores"):
                        st.markdown("### 📊 Evaluation Scores")

                        scores = result["evaluation_scores"]
                        # Filter out None scores
                        valid_scores = {
                            k: v for k, v in scores.items() if v is not None
                        }

                        if valid_scores:
                            cols = st.columns(len(valid_scores))

                            for idx, (metric_name, score) in enumerate(
                                valid_scores.items()
                            ):
                                with cols[idx]:
                                    # Visual indicator
                                    if score >= 0.8:
                                        color = "🟢"
                                        status = "Good"
                                    elif score >= 0.6:
                                        color = "🟡"
                                        status = "Fair"
                                    else:
                                        color = "🔴"
                                        status = "Poor"

                                    st.metric(
                                        f"{color} {metric_name.replace('_', ' ').title()}",
                                        f"{score:.3f}",
                                        delta=status,
                                    )
                        else:
                            st.info(
                                "No evaluation scores available. Some metrics may require reference answers."
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
                            "evaluation_scores": result.get("evaluation_scores"),
                        }
                    )

                    st.session_state.query_history.append(
                        {
                            "collection": selected_query_collection,
                            "question": query,
                            "answer": result["answer"],
                            "timestamp": time.time(),
                            "evaluation_scores": result.get("evaluation_scores"),
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
    """Task monitoring with real-time updates"""
    st.markdown(
        '<h2 class="tab-header">📊 Task Monitoring</h2>', unsafe_allow_html=True
    )

    # Initialize session state for real-time updates
    if "task_updates" not in st.session_state:
        st.session_state.task_updates = {}

    # Real-time update button
    if st.button("🔄 Enable Real-time Updates", key="realtime_toggle"):
        st.session_state.realtime_enabled = not st.session_state.get(
            "realtime_enabled", False
        )
        st.rerun()

    # Get tasks list
    try:
        response = requests.get(f"{API_BASE_URL}/ingest/tasks", timeout=5)

        if response.status_code == 200:
            data = response.json()
            tasks = data["tasks"]

            if tasks:
                # Display active tasks first
                active_tasks = [
                    t for t in tasks if t["status"] in ["started", "processing"]
                ]
                completed_tasks = [t for t in tasks if t["status"] == "completed"]

                # Active tasks section
                if active_tasks:
                    st.markdown("### 🔵 Active Tasks")

                    for task in active_tasks:
                        task_id = task["task_id"]

                        # Create a card for each active task
                        with st.container():
                            col1, col2, col3 = st.columns([3, 1, 1])

                            with col1:
                                st.markdown(f"**{task['collection_name']}**")
                                st.caption(f"ID: `{task_id[:8]}...`")

                            with col2:
                                # Get latest status
                                status_response = requests.get(
                                    f"{API_BASE_URL}/ingest/tasks/{task_id}/status",
                                    timeout=2,
                                )

                                if status_response.status_code == 200:
                                    status_data = status_response.json()
                                    progress = status_data.get("progress", 0)

                                    # Progress bar
                                    st.progress(progress / 100)
                                    st.caption(f"{progress}%")
                                else:
                                    st.caption("Updating...")

                            with col3:
                                if st.button("📊 Details", key=f"details_{task_id}"):
                                    st.session_state.selected_task = task_id
                                    st.rerun()

                        # Real-time updates for this task
                        if st.session_state.get("realtime_enabled"):
                            # Poll for updates every 2 seconds
                            if "last_update" not in st.session_state:
                                st.session_state.last_update = 0

                            current_time = time.time()
                            if current_time - st.session_state.last_update > 2:
                                # Update progress
                                try:
                                    ws_url = f"ws://localhost:8000/ws/tasks/{task_id}"
                                    pass
                                except:
                                    pass

                                st.session_state.last_update = current_time

                # Completed tasks section
                if completed_tasks:
                    st.markdown("### ✅ Completed Tasks")

                    # Show recent completed tasks
                    for task in completed_tasks[:5]:  # Show only 5 most recent
                        with st.expander(f"{task['collection_name']} - Completed"):
                            st.write(f"**Task ID:** `{task['task_id']}`")
                            st.write(
                                f"**Files:** {task.get('files_processed', 0)}/{task.get('total_files', 0)}"
                            )
                            st.write(
                                f"**Completed:** {task.get('completed_at', 'N/A')}"
                            )

                            if st.button("View Details", key=f"view_{task['task_id']}"):
                                st.session_state.selected_task = task["task_id"]
                                st.rerun()

                # Task details view
                if "selected_task" in st.session_state:
                    st.markdown("---")
                    st.markdown("### 🔍 Task Details")

                    task_id = st.session_state.selected_task

                    # Display detailed task info
                    try:
                        response = requests.get(
                            f"{API_BASE_URL}/ingest/tasks/{task_id}", timeout=5
                        )

                        if response.status_code == 200:
                            task_details = response.json()

                            # Status badge with color
                            status = task_details["status"]
                            status_colors = {
                                "completed": "🟢",
                                "failed": "🔴",
                                "processing": "🟡",
                                "started": "🟡",
                                "cancelled": "⚫",
                            }

                            col1, col2 = st.columns([1, 3])
                            with col1:
                                st.metric(
                                    "Status",
                                    f"{status_colors.get(status, '⚪')} {status.upper()}",
                                )
                            with col2:
                                if status == "processing":
                                    progress = task_details.get("progress", 0)
                                    st.progress(progress / 100)
                                    st.caption(f"Progress: {progress}%")

                            # Task info
                            info_cols = st.columns(2)
                            with info_cols[0]:
                                st.write(
                                    "**Collection:**", task_details["collection_name"]
                                )
                                st.write(
                                    "**Files:**",
                                    f"{task_details.get('files_processed', 0)}/{task_details.get('total_files', 0)}",
                                )
                            with info_cols[1]:
                                st.write("**Created:**", task_details["created_at"])
                                if task_details.get("completed_at"):
                                    st.write(
                                        "**Completed:**", task_details["completed_at"]
                                    )

                            # Action buttons
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                if st.button("🔄 Refresh", key="refresh_details"):
                                    st.rerun()
                            with col2:
                                if st.button("📋 Copy ID", key="copy_id"):
                                    st.code(task_id)
                            with col3:
                                if st.button("← Back to List", key="back_list"):
                                    del st.session_state.selected_task
                                    st.rerun()

                    except Exception as e:
                        st.error(f"Error loading task details: {e}")
            else:
                st.info("📭 No tasks found")

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
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "📤 Ingestion",
            "💬 Query",
            "📊 Evaluation",
            "📊 Task Monitoring",
            "📊 Dashboard",
        ]
    )

    with tab1:
        render_ingestion_tab()

    with tab2:
        render_query_tab()

    with tab3:
        render_evaluation_tab()

    with tab4:
        render_task_monitoring_tab()

    with tab5:
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
