from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., description="The question to ask")
    collection_name: Optional[str] = Field(None, description="Collection to query")
    top_k: Optional[int] = Field(4, description="Number of documents to retrieve")


class SourceDocument(BaseModel):
    content: str
    source: str
    collection: str
    score: Optional[float] = None


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceDocument]
    doc_count: int
    collection: str
    processing_time: Optional[float] = None
    error: Optional[str] = None


class CollectionIngestRequest(BaseModel):
    collection_name: str = Field(..., description="Name of the collection")
    chunk_size: Optional[int] = Field(1000, description="Chunk size for splitting")
    chunk_overlap: Optional[int] = Field(200, description="Chunk overlap for splitting")


class IngestResponse(BaseModel):
    success: bool
    message: str
    collection: str
    files_processed: Optional[int] = None
    chunks_created: Optional[int] = None
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
    collection_path: Optional[str] = None
    vector_store_path: Optional[str] = None
    processing_time: Optional[float] = None
    error: Optional[str] = None


class CollectionInfo(BaseModel):
    name: str
    file_count: int
    vector_exists: bool
    vector_count: int
    data_path: str
    vector_path: str


class CollectionsResponse(BaseModel):
    collections: List[CollectionInfo]
    current_collection: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    vector_store: Dict[str, Any]
    models: Dict[str, str]


class TaskStatus(str, Enum):
    """Ingestion task status"""

    STARTED = "started"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskFile(BaseModel):
    """File in an ingestion task"""

    filename: str
    file_size: Optional[int] = None
    status: str = "pending"


class IngestionTaskRequest(BaseModel):
    """Request to start an ingestion task"""

    collection_name: str = Field(..., description="Name of the collection")
    chunk_size: Optional[int] = Field(1000, description="Chunk size for splitting")
    chunk_overlap: Optional[int] = Field(200, description="Chunk overlap for splitting")


class IngestionTaskResponse(BaseModel):
    """Response with task ID"""

    task_id: str
    status: str
    message: str
    collection_name: str
    created_at: datetime
    estimated_time: Optional[int] = Field(None, description="Estimated time in seconds")


class TaskStatusResponse(BaseModel):
    """Detailed task status response"""

    task_id: str
    collection_name: str
    status: str
    message: Optional[str]
    files_processed: int = 0
    chunks_created: int = 0
    total_files: int = 0
    progress: int = 0
    created_at: datetime
    updated_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    error_details: Optional[str]
    metadata: Optional[Dict[str, Any]]
    files: List[TaskFile] = []
    chunk_size: Optional[int]
    chunk_overlap: Optional[int]
    current_file: Optional[str] = None
    current_action: Optional[str] = None


class TasksListResponse(BaseModel):
    """List of tasks with pagination"""

    tasks: List[TaskStatusResponse]
    total: int
    limit: int
    offset: int


class TaskStatsResponse(BaseModel):
    """Task statistics"""

    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    active_tasks: int
    processing_tasks: int
    total_files_processed: int
    total_chunks_created: int
    recent_collections: List[Dict[str, Any]]


class TaskUpdateRequest(BaseModel):
    """Request to update task status"""

    status: Optional[TaskStatus] = None
    message: Optional[str] = None
    progress: Optional[int] = Field(None, ge=0, le=100)


class EvaluationRequest(BaseModel):
    """Request to evaluate a query with optional reference answer"""

    question: str = Field(..., description="The question that was asked")
    answer: str = Field(..., description="The generated answer")
    contexts: List[str] = Field(..., description="Retrieved context strings")
    reference_answer: Optional[str] = Field(
        None, description="Optional ground truth answer"
    )
    trace_id: Optional[str] = Field(
        None, description="Optional Langfuse trace ID to attach scores"
    )


class EvaluationScores(BaseModel):
    """Evaluation scores from RAGAS metrics"""

    faithfulness: Optional[float] = None
    answer_relevancy: Optional[float] = None
    context_precision: Optional[float] = None
    llm_context_precision_without_reference: Optional[float] = None


class EvaluationResponse(BaseModel):
    """Response with evaluation scores"""

    question: str
    answer: str
    scores: Dict[str, Optional[float]]
    evaluated_at: str
    trace_id: Optional[str] = None


class QueryWithEvaluationRequest(BaseModel):
    """Request to query and evaluate in one call"""

    question: str = Field(..., description="The question to ask")
    collection_name: Optional[str] = Field(None, description="Collection to query")
    top_k: Optional[int] = Field(4, description="Number of documents to retrieve")
    reference_answer: Optional[str] = Field(
        None, description="Optional reference answer for evaluation"
    )
    evaluate: bool = Field(
        True, description="Whether to evaluate the response with RAGAS"
    )


class QueryWithEvaluationResponse(BaseModel):
    """Response with query results and evaluation scores"""

    question: str
    answer: str
    sources: List[SourceDocument]
    doc_count: int
    collection: str
    processing_time: Optional[float] = None
    evaluation_scores: Optional[Dict[str, Optional[float]]] = None
    evaluated_at: Optional[str] = None
    trace_id: Optional[str] = None
    error: Optional[str] = None


class BatchEvaluationRequest(BaseModel):
    """Request to evaluate multiple query-answer pairs"""

    items: List[Dict[str, Any]] = Field(
        ...,
        description="List of items with 'question', 'answer', 'contexts', and optional 'reference_answer'",
    )
    push_to_langfuse: bool = Field(
        True, description="Whether to create Langfuse traces"
    )


class BatchEvaluationResponse(BaseModel):
    """Response with batch evaluation results"""

    results: List[Dict[str, Any]]
    total_evaluated: int
    average_scores: Dict[str, Optional[float]]
    evaluated_at: str
