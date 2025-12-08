"""SQLite database operations for tracking ingestion tasks"""

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

DATABASE_PATH = Path("./data/ingestion_tasks.db")


def init_database():
    """Initialize the database and create tables"""
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)

    with get_db_connection() as conn:
        # Create ingestion tasks table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ingestion_tasks (
                task_id TEXT PRIMARY KEY,
                collection_name TEXT NOT NULL,
                status TEXT NOT NULL,
                message TEXT,
                files_processed INTEGER DEFAULT 0,
                chunks_created INTEGER DEFAULT 0,
                total_files INTEGER DEFAULT 0,
                progress INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                error_details TEXT,
                metadata TEXT,
                chunk_size INTEGER,
                chunk_overlap INTEGER
            )
        """)

        # Create task files table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS task_files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                file_size INTEGER,
                status TEXT DEFAULT 'pending',
                FOREIGN KEY (task_id) REFERENCES ingestion_tasks (task_id) ON DELETE CASCADE
            )
        """)

        # Create indexes
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_tasks_status ON ingestion_tasks(status)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_tasks_collection ON ingestion_tasks(collection_name)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_tasks_created ON ingestion_tasks(created_at)"
        )

        conn.commit()
    print(f"✅ Database initialized at: {DATABASE_PATH}")


@contextmanager
def get_db_connection():
    """Context manager for database connections"""
    conn = sqlite3.connect(str(DATABASE_PATH))
    conn.row_factory = sqlite3.Row  # Return rows as dictionaries
    try:
        yield conn
    finally:
        conn.close()


def create_ingestion_task(
    collection_name: str,
    total_files: int,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> str:
    """Create a new ingestion task and return task ID"""
    task_id = str(uuid.uuid4())

    with get_db_connection() as conn:
        conn.execute(
            """
            INSERT INTO ingestion_tasks
            (task_id, collection_name, status, total_files, chunk_size, chunk_overlap, started_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (
                task_id,
                collection_name,
                "started",
                total_files,
                chunk_size,
                chunk_overlap,
                datetime.utcnow().isoformat(),
            ),
        )

        conn.commit()

    return task_id


def update_task_status(
    task_id: str,
    status: str,
    message: Optional[str] = None,
    files_processed: Optional[int] = None,
    chunks_created: Optional[int] = None,
    progress: Optional[int] = None,
    error_details: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
):
    """Update task status and progress"""
    with get_db_connection() as conn:
        updates = ["status = ?", "updated_at = ?"]
        params = [status, datetime.utcnow().isoformat()]

        if message is not None:
            updates.append("message = ?")
            params.append(message)

        if files_processed is not None:
            updates.append("files_processed = ?")
            params.append(files_processed)

        if chunks_created is not None:
            updates.append("chunks_created = ?")
            params.append(chunks_created)

        if progress is not None:
            updates.append("progress = ?")
            params.append(progress)

        if error_details is not None:
            updates.append("error_details = ?")
            params.append(error_details)

        if metadata is not None:
            updates.append("metadata = ?")
            params.append(json.dumps(metadata))

        if status == "completed":
            updates.append("completed_at = ?")
            params.append(datetime.utcnow().isoformat())

        params.append(task_id)

        query = f"UPDATE ingestion_tasks SET {', '.join(updates)} WHERE task_id = ?"
        conn.execute(query, params)
        conn.commit()


def add_task_files(task_id: str, filenames: List[str]):
    """Add files to a task"""
    with get_db_connection() as conn:
        for filename in filenames:
            conn.execute(
                """
                INSERT INTO task_files (task_id, filename)
                VALUES (?, ?)
            """,
                (task_id, filename),
            )
        conn.commit()


def update_file_status(task_id: str, filename: str, status: str):
    """Update status of a specific file in a task"""
    with get_db_connection() as conn:
        conn.execute(
            """
            UPDATE task_files
            SET status = ?
            WHERE task_id = ? AND filename = ?
        """,
            (status, task_id, filename),
        )
        conn.commit()


def get_task(task_id: str) -> Optional[Dict[str, Any]]:
    """Get task details by ID"""
    with get_db_connection() as conn:
        cursor = conn.execute(
            """
            SELECT
                task_id, collection_name, status, message,
                files_processed, chunks_created, total_files,
                progress, created_at, updated_at,
                started_at, completed_at, error_details,
                metadata, chunk_size, chunk_overlap
            FROM ingestion_tasks
            WHERE task_id = ?
        """,
            (task_id,),
        )

        row = cursor.fetchone()
        if not row:
            return None

        task = dict(row)

        # Parse metadata JSON
        if task.get("metadata"):
            task["metadata"] = json.loads(task["metadata"])

        # Get associated files
        cursor = conn.execute(
            """
            SELECT filename, file_size, status
            FROM task_files
            WHERE task_id = ?
            ORDER BY id
        """,
            (task_id,),
        )

        task["files"] = [dict(file_row) for file_row in cursor.fetchall()]

        return task


def list_tasks(
    limit: int = 50,
    offset: int = 0,
    status: Optional[str] = None,
    collection_name: Optional[str] = None,
) -> Dict[str, Any]:
    """List tasks with optional filters"""
    with get_db_connection() as conn:
        # Build query
        query = """
            SELECT
                task_id, collection_name, status, message,
                files_processed, chunks_created, total_files,
                progress, created_at, updated_at,
                started_at, completed_at
            FROM ingestion_tasks
            WHERE 1=1
        """
        params = []

        if status:
            query += " AND status = ?"
            params.append(status)

        if collection_name:
            query += " AND collection_name = ?"
            params.append(collection_name)

        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        cursor = conn.execute(query, params)
        tasks = [dict(row) for row in cursor.fetchall()]

        # Get total count
        count_query = "SELECT COUNT(*) as total FROM ingestion_tasks WHERE 1=1"
        count_params = []

        if status:
            count_query += " AND status = ?"
            count_params.append(status)

        if collection_name:
            count_query += " AND collection_name = ?"
            count_params.append(collection_name)

        cursor = conn.execute(count_query, count_params)
        total = cursor.fetchone()["total"]

        return {"tasks": tasks, "total": total, "limit": limit, "offset": offset}


def delete_task(task_id: str) -> bool:
    """Delete a task and its files"""
    with get_db_connection() as conn:
        cursor = conn.execute(
            "DELETE FROM ingestion_tasks WHERE task_id = ?", (task_id,)
        )
        conn.commit()
        return cursor.rowcount > 0


def cleanup_old_tasks(days_old: int = 30):
    """Clean up old completed/failed tasks"""
    with get_db_connection() as conn:
        cutoff_date = (
            datetime.utcnow().replace(day=datetime.utcnow().day - days_old).isoformat()
        )

        conn.execute(
            """
            DELETE FROM ingestion_tasks
            WHERE status IN ('completed', 'failed', 'cancelled')
            AND updated_at < ?
        """,
            (cutoff_date,),
        )

        conn.commit()
        print(f"✅ Cleaned up old tasks")


def update_task_progress(
    task_id: str,
    progress: int,
    current_file: Optional[str] = None,
    current_action: Optional[str] = None,
):
    """Update task progress incrementally"""
    with get_db_connection() as conn:
        updates = ["progress = ?", "updated_at = ?"]
        params = [progress, datetime.utcnow().isoformat()]

        if current_file:
            updates.append("current_file = ?")
            params.append(current_file)

        if current_action:
            updates.append("current_action = ?")
            params.append(current_action)

        params.append(task_id)

        query = f"UPDATE ingestion_tasks SET {', '.join(updates)} WHERE task_id = ?"
        conn.execute(query, params)
        conn.commit()


def get_task_status(task_id: str) -> Optional[Dict[str, Any]]:
    """Quick status check without full details"""
    with get_db_connection() as conn:
        cursor = conn.execute(
            """
            SELECT status, progress, current_file, current_action, updated_at
            FROM ingestion_tasks
            WHERE task_id = ?
        """,
            (task_id,),
        )

        row = cursor.fetchone()
        return dict(row) if row else None
