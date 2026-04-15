import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import declarative_base, sessionmaker

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL not set. Create a .env file (see .env.example).")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

Base = declarative_base()


def ensure_schema() -> None:
    """
    Apply a tiny compatibility migration for local/dev databases that predate
    user ownership on portfolios.
    """
    inspector = inspect(engine)

    if "portfolios" not in inspector.get_table_names():
        return

    portfolio_columns = {column["name"] for column in inspector.get_columns("portfolios")}
    if "user_id" in portfolio_columns:
        return

    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE portfolios ADD COLUMN user_id INTEGER"))
