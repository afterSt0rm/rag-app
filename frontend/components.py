from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st


def collection_card(collection_info: Dict[str, Any], key_suffix: str = ""):
    """Display a collection card with stats"""
    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown(f"### 📁 {collection_info['name']}")
        st.caption(f"Path: `{collection_info['data_path']}`")

    with col2:
        if collection_info["vector_exists"]:
            st.metric("Documents", collection_info["vector_count"])
        else:
            st.warning("Not indexed")

    # File stats
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Files", collection_info["file_count"])
    with col2:
        if collection_info["vector_exists"]:
            st.success("✓ Indexed")
        else:
            st.error("✗ Not indexed")

    return collection_info["name"]


def file_uploader_with_preview(
    supported_extensions: List[str], key: str = "file_uploader"
):
    """Enhanced file uploader with preview"""
    uploaded_files = st.file_uploader(
        "Choose files to upload",
        type=[ext.replace(".", "") for ext in supported_extensions],
        accept_multiple_files=True,
        key=key,
    )

    if uploaded_files:
        st.markdown("**Selected Files:**")
        file_data = []

        for file in uploaded_files:
            file_data.append(
                {
                    "Filename": file.name,
                    "Size": f"{len(file.getvalue()) / 1024:.1f} KB",
                    "Type": file.type or "Unknown",
                }
            )

        df = pd.DataFrame(file_data)
        st.dataframe(df, use_container_width=True)

    return uploaded_files


def chunking_settings(default_size: int = 1000, default_overlap: int = 200):
    """UI for chunking settings"""
    st.markdown("### ⚙️ Chunking Settings")

    col1, col2 = st.columns(2)

    with col1:
        chunk_size = st.slider(
            "Chunk Size (characters)",
            min_value=100,
            max_value=5000,
            value=default_size,
            step=100,
            help="Size of each text chunk",
        )

    with col2:
        chunk_overlap = st.slider(
            "Chunk Overlap (characters)",
            min_value=0,
            max_value=1000,
            value=default_overlap,
            step=50,
            help="Overlap between chunks to preserve context",
        )

    # Preview
    st.info(
        f"**Preview:** Each document will be split into chunks of **{chunk_size}** characters with **{chunk_overlap}** characters overlap."
    )

    return chunk_size, chunk_overlap


def collection_selector(
    available_collections: List[str], current_collection: Optional[str] = None
):
    """Collection selector dropdown"""
    col1, col2 = st.columns([3, 1])

    with col1:
        if available_collections:
            selected = st.selectbox(
                "Select Collection",
                options=available_collections,
                index=available_collections.index(current_collection)
                if current_collection in available_collections
                else 0,
                key="collection_select",
            )
        else:
            st.warning("No collections available")
            selected = None

    with col2:
        st.markdown("")
        st.markdown("")
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()

    return selected


def evaluation_score_card(metric_name: str, score: float, description: str = ""):
    """Display an evaluation score with visual indicator"""
    # Determine color and status based on score
    if score >= 0.8:
        color = "🟢"
        status = "Good"
        bg_color = "#d4edda"
    elif score >= 0.6:
        color = "🟡"
        status = "Fair"
        bg_color = "#fff3cd"
    else:
        color = "🔴"
        status = "Poor"
        bg_color = "#f8d7da"

    st.markdown(
        f"""
        <div style="background-color: {bg_color}; padding: 1rem; border-radius: 8px; margin-bottom: 0.5rem;">
            <h4>{color} {metric_name.replace("_", " ").title()}</h4>
            <h2>{score:.3f}</h2>
            <p><strong>Status:</strong> {status}</p>
            {f'<p style="font-size: 0.9em; color: #666;">{description}</p>' if description else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )


def evaluation_metrics_row(scores: Dict[str, float]):
    """Display evaluation scores in a row"""
    cols = st.columns(len(scores))

    for idx, (metric_name, score) in enumerate(scores.items()):
        with cols[idx]:
            if score is not None:
                # Determine status
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
                st.metric(metric_name.replace("_", " ").title(), "N/A")


def evaluation_summary_table(evaluation_results: List[Dict]):
    """Display a summary table of evaluation results"""
    if not evaluation_results:
        st.info("No evaluation results yet.")
        return

    # Prepare data for table
    table_data = []
    for result in evaluation_results:
        row = {
            "Question": result.get("question", "")[:50] + "...",
            "Timestamp": result.get("timestamp", ""),
        }

        # Add scores
        scores = result.get("evaluation_scores", {})
        for metric, score in scores.items():
            if score is not None:
                row[metric.replace("_", " ").title()] = f"{score:.3f}"
            else:
                row[metric.replace("_", " ").title()] = "N/A"

        table_data.append(row)

    df = pd.DataFrame(table_data)
    st.dataframe(df, use_container_width=True)


def metric_explanation():
    """Display explanation of evaluation metrics"""
    with st.expander("📖 Understanding Evaluation Metrics"):
        st.markdown("""
        ### Evaluation Metrics Explained

        **Faithfulness (0-1)**
        - Measures if the answer is factually consistent with the retrieved contexts
        - Higher is better
        - ✅ Good: > 0.8 | ⚠️ Fair: 0.6-0.8 | ❌ Poor: < 0.6
        - Use case: Detect hallucinations

        **Context Precision (0-1)**
        - Measures if relevant contexts are ranked higher in retrieval
        - Higher is better
        - ✅ Good: > 0.8 | ⚠️ Fair: 0.6-0.8 | ❌ Poor: < 0.6
        - Use case: Optimize retrieval strategy

        **LLM Context Precision Without Reference (0-1)**
        - Evaluates retrieval quality without needing ground truth
        - Higher is better
        - ✅ Good: > 0.75 | ⚠️ Fair: 0.6-0.75 | ❌ Poor: < 0.6
        - Use case: General quality assessment

        **Answer Relevancy (0-1)** *(Requires Ollama)*
        - Measures how relevant the answer is to the question
        - Higher is better
        - ✅ Good: > 0.85 | ⚠️ Fair: 0.7-0.85 | ❌ Poor: < 0.7
        - Use case: Ensure on-topic responses

        ⚠️ **Note:** By default, the system uses Cerebras-compatible metrics.
        For AnswerRelevancy, Ollama must be configured for evaluation.

        See [evaluation/METRICS_COMPATIBILITY.md](evaluation/METRICS_COMPATIBILITY.md) for details.
        """)


def test_case_card(test_case: Dict, index: int, on_remove=None):
    """Display a test case card with optional remove button"""
    with st.container():
        st.markdown(
            f"""
            <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 8px; margin-bottom: 0.8rem;">
                <h4>Test Case {index + 1}</h4>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns([4, 1])

        with col1:
            st.markdown(f"**Q:** {test_case.get('question', '')[:100]}...")
            st.markdown(f"**A:** {test_case.get('answer', '')[:100]}...")
            st.caption(f"Contexts: {len(test_case.get('contexts', []))}")

        with col2:
            if on_remove and st.button("🗑️ Remove", key=f"remove_tc_{index}"):
                on_remove(index)
