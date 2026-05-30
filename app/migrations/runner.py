"""Lightweight versioned SQL migrations for edge Postgres (no Alembic)."""

import logging
import re
from pathlib import Path
from typing import List

from sqlalchemy import text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)

MIGRATIONS_DIR = Path(__file__).parent / "versions"
MIGRATIONS_TABLE = "edge_schema_migrations"
_VERSION_PATTERN = re.compile(r"^\d{3}_")


def _ensure_migrations_table(conn) -> None:
    conn.execute(
        text(
            f"""
            CREATE TABLE IF NOT EXISTS {MIGRATIONS_TABLE} (
                version VARCHAR(255) PRIMARY KEY,
                applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
    )


def _applied_versions(conn) -> set[str]:
    rows = conn.execute(text(f"SELECT version FROM {MIGRATIONS_TABLE}"))
    return {row[0] for row in rows}


def _migration_files() -> List[Path]:
    if not MIGRATIONS_DIR.is_dir():
        return []
    files = [
        path
        for path in MIGRATIONS_DIR.glob("*.sql")
        if _VERSION_PATTERN.match(path.name)
    ]
    return sorted(files, key=lambda path: path.name)


def _split_sql_statements(sql: str) -> List[str]:
    statements: List[str] = []
    for chunk in sql.split(";"):
        lines = [
            line
            for line in chunk.splitlines()
            if line.strip() and not line.strip().startswith("--")
        ]
        statement = "\n".join(lines).strip()
        if statement:
            statements.append(statement)
    return statements


def _execute_sql_script(conn, sql: str) -> None:
    for statement in _split_sql_statements(sql):
        conn.execute(text(statement))


def run_migrations(engine: Engine) -> List[str]:
    """Apply pending SQL migrations in lexical order. Returns newly applied versions."""
    files = _migration_files()
    if not files:
        return []

    applied_new: List[str] = []
    with engine.begin() as conn:
        _ensure_migrations_table(conn)
        applied = _applied_versions(conn)

        for path in files:
            version = path.stem
            if version in applied:
                continue

            sql = path.read_text(encoding="utf-8").strip()
            if not sql:
                logger.warning("Skipping empty migration %s", version)
                continue

            logger.info("Applying migration %s", version)
            _execute_sql_script(conn, sql)
            conn.execute(
                text(f"INSERT INTO {MIGRATIONS_TABLE} (version) VALUES (:version)"),
                {"version": version},
            )
            applied_new.append(version)
            logger.info("Applied migration %s", version)

    return applied_new
