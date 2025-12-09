import asyncio
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import uvicorn
from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from ingestion.ingestion_pipeline import get_ingestion_pipeline
from query.config import QueryConfig
from query.query_pipeline import get_query_pipeline

from api.database import (
    add_task_files,
    cleanup_old_tasks,
    create_ingestion_task,
    delete_task,
    get_task,
    get_task_status,
    init_database,
    list_tasks,
    update_file_status,
    update_task_progress,
    update_task_status,
)
from api.models import (
    CollectionInfo,
    CollectionIngestRequest,
    CollectionsResponse,
    HealthResponse,
    IngestionTaskRequest,
    IngestionTaskResponse,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    TasksListResponse,
    TaskStatsResponse,
    TaskStatus,
    TaskStatusResponse,
    TaskUpdateRequest,
)

# Initialize FastAPI app
app = FastAPI(
    title="RAG System API",
    description="Multi-Collection RAG System with Ollama and Gemini",
    version="2.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state for current collection
_current_collection = None

# Create a thread pool for CPU-bound tasks
thread_pool = ThreadPoolExecutor(max_workers=4)


@app.get("/collections", response_model=CollectionsResponse)
async def list_collections():
    """List all available collections"""
    pipeline = get_ingestion_pipeline()
    stats = pipeline.list_all_collection_stats()

    collections_info = []
    for stat in stats:
        if stat["exists"]:
            collections_info.append(CollectionInfo(**stat))

    return CollectionsResponse(
        collections=collections_info, current_collection=_current_collection
    )


@app.post("/collections/{collection_name}")
async def create_collection(collection_name: str):
    """Create a new collection"""
    collection_name = collection_name.replace(" ", "")
    pipeline = get_ingestion_pipeline()
    collection_path = pipeline.collection_manager.create_collection(collection_name)

    return {
        "success": True,
        "message": f"Collection '{collection_name}' created",
        "path": str(collection_path),
    }


async def process_single_file(file_path: Path, filename: str, task_id: str):
    """Process a single file with progress updates"""
    try:
        # Update file status
        update_file_status(task_id, filename, "processing")

        # Simulate some processing time
        await asyncio.sleep(0.5)

        # Update progress
        return True
    except Exception as e:
        update_file_status(task_id, filename, "failed")
        raise


async def run_ingestion_background(
    task_id: str,
    collection_name: str,
    file_paths: List[Path],
    chunk_size: int,
    chunk_overlap: int,
    filenames: List[str],
):
    """Run ingestion with real-time progress updates"""
    total_files = len(file_paths)

    try:
        # Update to processing
        update_task_status(
            task_id=task_id,
            status="processing",
            message=f"Starting to process {total_files} files...",
            progress=5,
        )

        # Process files one by one with progress updates
        processed_count = 0
        for i, (file_path, filename) in enumerate(zip(file_paths, filenames), 1):
            try:
                # Update current file
                update_task_progress(
                    task_id=task_id,
                    progress=int((i / total_files) * 90),  # Leave 10% for final steps
                    current_file=filename,
                    current_action="loading",
                )

                # Process the file
                await process_single_file(file_path, filename, task_id)
                update_file_status(task_id, filename, "loaded")

                processed_count += 1

                # Update progress
                update_task_status(
                    task_id=task_id,
                    status="processing",
                    files_processed=processed_count,
                    progress=int((i / total_files) * 90),
                    message=f"Processed {i}/{total_files} files",
                )

                # Small delay to allow status updates
                await asyncio.sleep(0.1)

            except Exception as e:
                update_file_status(task_id, filename, "failed")
                raise

        # Run the actual ingestion
        update_task_progress(
            task_id=task_id, progress=95, current_action="creating_vector_store"
        )

        # Run ingestion in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        pipeline = get_ingestion_pipeline()

        result = await loop.run_in_executor(
            thread_pool,
            lambda: pipeline.ingest_to_collection(
                collection_name=collection_name,
                file_paths=file_paths,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            ),
        )

        # Update final status
        if result.get("success"):
            update_task_status(
                task_id=task_id,
                status="completed",
                message=result.get("message"),
                files_processed=processed_count,
                chunks_created=result.get("chunks_created", 0),
                progress=100,
                metadata={
                    "collection_path": result.get("collection_path"),
                    "vector_store_path": result.get("vector_store_path"),
                    "processing_time": result.get("processing_time"),
                },
            )
        else:
            update_task_status(
                task_id=task_id,
                status="failed",
                message=result.get("message"),
                error_details=result.get("error"),
                files_processed=processed_count,
                progress=100,
            )

    except Exception as e:
        update_task_status(
            task_id=task_id,
            status="failed",
            message="Ingestion failed with error",
            error_details=str(e),
            progress=100,
        )
        raise
    finally:
        # Clean up temp files
        for file_path in file_paths:
            if file_path.exists():
                file_path.unlink()

        temp_dir = Path("./temp_uploads")
        if temp_dir.exists():
            temp_dir.rmdir()


@app.post("/ingest", response_model=IngestionTaskResponse)
async def ingest_documents(
    background_tasks: BackgroundTasks,
    collection_name: str = Form(...),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(200),
    files: List[UploadFile] = File(...),
):
    """Ingest documents with real-time progress tracking"""
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    try:
        # Save uploaded files temporarily
        temp_dir = Path("./temp_uploads")
        temp_dir.mkdir(exist_ok=True)

        file_paths = []
        filenames = []

        for file in files:
            file_path = temp_dir / file.filename
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            file_paths.append(file_path)
            filenames.append(file.filename)

        # Create task record
        task_id = create_ingestion_task(
            collection_name=collection_name,
            total_files=len(files),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        # Add files to task
        add_task_files(task_id, filenames)

        # Add background task
        background_tasks.add_task(
            run_ingestion_background,
            task_id=task_id,
            collection_name=collection_name,
            file_paths=file_paths,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            filenames=filenames,
        )

        return IngestionTaskResponse(
            task_id=task_id,
            status="started",
            message=f"Ingestion started for {len(files)} files",
            collection_name=collection_name,
            created_at=datetime.utcnow(),
            estimated_time=len(files) * 3,  # 3 seconds per file estimate
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ingest/tasks", response_model=TasksListResponse)
async def list_ingestion_tasks(
    limit: int = 50,
    offset: int = 0,
    status: Optional[str] = None,
    collection_name: Optional[str] = None,
):
    """List all ingestion tasks"""
    try:
        result = list_tasks(
            limit=limit, offset=offset, status=status, collection_name=collection_name
        )

        # Convert to response models
        tasks = []
        for task_data in result["tasks"]:
            task_details = get_task(task_data["task_id"])
            if task_details:
                tasks.append(TaskStatusResponse(**task_details))

        return TasksListResponse(
            tasks=tasks, total=result["total"], limit=limit, offset=offset
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ingest/tasks/{task_id}", response_model=TaskStatusResponse)
async def get_ingestion_task(task_id: str):
    """Get status of a specific ingestion task"""
    task = get_task(task_id)

    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    return TaskStatusResponse(**task)


@app.delete("/ingest/tasks/{task_id}")
async def cancel_ingestion_task(task_id: str):
    """Cancel an ingestion task"""
    task = get_task(task_id)

    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    if task["status"] not in ["started", "processing"]:
        raise HTTPException(
            status_code=400, detail=f"Cannot cancel task with status: {task['status']}"
        )

    update_task_status(
        task_id=task_id, status="cancelled", message="Task cancelled by user"
    )

    return {"success": True, "message": f"Task {task_id} cancelled"}


@app.get("/ingest/tasks/stats", response_model=TaskStatsResponse)
async def get_task_statistics():
    """Get ingestion task statistics"""
    try:
        stats = get_task_stats()
        return TaskStatsResponse(**stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/ingest/tasks/{task_id}")
async def update_task(task_id: str, update_request: TaskUpdateRequest):
    """Update task status (for testing or manual intervention)"""
    task = get_task(task_id)

    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    update_task_status(
        task_id=task_id,
        status=update_request.status.value if update_request.status else None,
        message=update_request.message,
        progress=update_request.progress,
    )

    return {"success": True, "message": f"Task {task_id} updated"}


@app.post("/query", response_model=QueryResponse, response_model_exclude_none=True)
async def query_rag_endpoint(request: QueryRequest):
    """Query a specific collection"""
    start_time = time.time()

    try:
        # Use collection from request or default to current
        collection_name = request.collection_name or _current_collection

        if not collection_name:
            raise HTTPException(
                status_code=400,
                detail="No collection specified. Please provide a collection_name or set a current collection.",
            )

        # Validate top_k parameter
        top_k = request.top_k
        if top_k is not None:
            if not isinstance(top_k, int):
                raise HTTPException(status_code=400, detail="top_k must be an integer")
            if top_k <= 0:
                raise HTTPException(
                    status_code=400, detail="top_k must be a positive integer"
                )
            if top_k > 20:
                raise HTTPException(
                    status_code=400, detail="top_k cannot exceed 20 documents"
                )

        # Get query pipeline for this collection
        pipeline = get_query_pipeline(collection_name)

        # Execute query
        result = pipeline.query(request.question, collection_name, top_k)

        processing_time = time.time() - start_time

        # Convert to response model
        sources = []
        for source in result.get("sources", []):
            sources.append(
                {
                    "content": source["content"],
                    "source": source["source"],
                    "collection": source.get("collection", collection_name),
                    "score": source.get("score"),
                }
            )

        return QueryResponse(
            question=result["question"],
            answer=result["answer"],
            sources=sources,
            doc_count=result["doc_count"],
            collection=result["collection"],
            processing_time=processing_time,
            error=result.get("error"),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/collection/current")
async def set_current_collection(collection_name: str):
    """Set the current collection"""
    global _current_collection

    # Verify collection exists
    pipeline = get_ingestion_pipeline()
    if collection_name not in pipeline.collection_manager.list_collections():
        raise HTTPException(
            status_code=404, detail=f"Collection '{collection_name}' not found"
        )

    _current_collection = collection_name

    # Pre-load the collection in query pipeline
    get_query_pipeline(collection_name)

    return {
        "success": True,
        "message": f"Current collection set to '{collection_name}'",
        "collection": collection_name,
    }


@app.get("/collection/current/info")
async def get_current_collection_info():
    """Get information about the current collection"""
    if not _current_collection:
        return {"current_collection": None, "loaded": False}

    pipeline = get_query_pipeline(_current_collection)
    info = pipeline.get_collection_info()

    return {"current_collection": _current_collection, **info}


@app.get("/search")
async def search_documents(
    query: str, collection_name: Optional[str] = None, k: int = 3
):
    """Search for similar documents in a collection"""
    collection_name = collection_name or _current_collection

    if not collection_name:
        raise HTTPException(status_code=400, detail="No collection specified")

    try:
        pipeline = get_query_pipeline(collection_name)
        docs = pipeline.similarity_search(query, k=k)

        results = []
        for doc in docs:
            results.append(
                {
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", "Unknown"),
                    "collection": doc.metadata.get("collection", collection_name),
                    "metadata": doc.metadata,
                }
            )

        return {"query": query, "collection": collection_name, "results": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Health check endpoint
@app.get("/health")
async def health_check():
    """Check API health and available collections"""
    ingestion_pipeline = get_ingestion_pipeline()
    collections = ingestion_pipeline.collection_manager.list_collections()

    return {
        "status": "healthy",
        "api_version": "2.0.0",
        "available_collections": collections,
        "current_collection": _current_collection,
        "embedding_model": ingestion_pipeline.config.embedding_model,
    }


@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    print("🚀 Starting RAG System API with Background Task Tracking...")

    # Create necessary directories
    Path("./data/collections").mkdir(parents=True, exist_ok=True)
    Path("./vector_store/chroma_db").mkdir(parents=True, exist_ok=True)
    Path("./temp_uploads").mkdir(parents=True, exist_ok=True)

    # Initialize database
    init_database()

    # Clean up old tasks on startup
    cleanup_old_tasks(days_old=7)

    print("✅ API is ready with background task tracking!")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("👋 Shutting down RAG System API...")

    # Clean up temp directory
    temp_dir = Path("./temp_uploads")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    print("✅ Cleanup completed!")


if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
