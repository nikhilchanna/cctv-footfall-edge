#!/usr/bin/env python3
"""Apply pending edge DB migrations without starting the API server."""

from app.database import engine, init_db
from app.migrations.runner import run_migrations


def main() -> None:
    init_db(create_only=True)
    applied = run_migrations(engine)
    if applied:
        print("Applied migrations:", ", ".join(applied))
    else:
        print("No pending migrations.")


if __name__ == "__main__":
    main()
