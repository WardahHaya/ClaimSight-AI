import json
import os
import sqlite3
from typing import Any, Dict, List, Optional

from backend.config import SQLITE_PATH


def get_db_connection() -> sqlite3.Connection:
    """
    Establish and return a SQLite connection with row access by column name.
    """
    os.makedirs(os.path.dirname(SQLITE_PATH), exist_ok=True)
    connection = sqlite3.connect(SQLITE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def _ensure_column(connection: sqlite3.Connection, column_name: str, definition: str) -> None:
    existing_columns = {
        row["name"]
        for row in connection.execute("PRAGMA table_info(chat_history)").fetchall()
    }
    if column_name not in existing_columns:
        connection.execute(f"ALTER TABLE chat_history ADD COLUMN {column_name} {definition}")


def init_db() -> None:
    """
    Initialize the chat history schema and apply lightweight migrations.
    """
    with get_db_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                image_analysis TEXT,
                image_data_url TEXT,
                pipeline_route TEXT,
                response_mode TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        _ensure_column(connection, "image_analysis", "TEXT")
        _ensure_column(connection, "image_data_url", "TEXT")
        _ensure_column(connection, "pipeline_route", "TEXT")
        _ensure_column(connection, "response_mode", "TEXT")
        connection.execute("CREATE INDEX IF NOT EXISTS idx_session ON chat_history(session_id)")
        connection.commit()


def save_message(
    session_id: str,
    role: str,
    content: str,
    image_analysis: Optional[str] = None,
    image_data_url: Optional[str] = None,
    pipeline_route: Optional[List[str]] = None,
    response_mode: Optional[str] = None,
) -> None:
    """
    Save a message turn to SQLite using parameterized queries.
    """
    with get_db_connection() as connection:
        connection.execute(
            """
            INSERT INTO chat_history (
                session_id,
                role,
                content,
                image_analysis,
                image_data_url,
                pipeline_route,
                response_mode
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                role,
                content,
                image_analysis,
                image_data_url,
                json.dumps(pipeline_route) if pipeline_route else None,
                response_mode,
            ),
        )
        connection.commit()


def get_history(session_id: str, limit: int = 20) -> List[Dict[str, Any]]:
    """
    Return the latest messages for a session in chronological order.
    """
    init_db()
    with get_db_connection() as connection:
        rows = connection.execute(
            """
            SELECT role, content, image_analysis, image_data_url, pipeline_route, response_mode
            FROM chat_history
            WHERE session_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (session_id, limit),
        ).fetchall()

    history: List[Dict[str, Any]] = []
    for row in reversed(rows):
        history.append(
            {
                "role": row["role"],
                "content": row["content"],
                "image_analysis": row["image_analysis"],
                "image_data_url": row["image_data_url"],
                "pipeline_route": json.loads(row["pipeline_route"]) if row["pipeline_route"] else None,
                "response_mode": row["response_mode"],
            }
        )
    return history


def clear_history(session_id: str) -> None:
    """
    Delete all message history for a session.
    """
    with get_db_connection() as connection:
        connection.execute("DELETE FROM chat_history WHERE session_id = ?", (session_id,))
        connection.commit()
