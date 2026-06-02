import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# Use environment variable or fallback to a default local postgres url
SQLALCHEMY_DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "postgresql://postgres:Dmimpact%40123@localhost:5432/footfall_db"
)

engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def init_db(*, create_only: bool = False) -> None:
    """Create tables for new installs; optionally skip versioned migrations."""
    Base.metadata.create_all(bind=engine)
    if not create_only:
        from app.migrations.runner import run_migrations

        run_migrations(engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
