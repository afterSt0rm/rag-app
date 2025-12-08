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
