from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from rediris.common.config import settings
from rediris.common.utils.logging import setup_logger

logger = setup_logger(__name__)

engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,
    pool_size=100,
    max_overflow=50,
    pool_recycle=3600,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def recreate_engine_and_session(database_url: str):

    global engine, SessionLocal
    logger.info(f"Recreating database engine with URL: {database_url}")
    engine = create_engine(
        database_url,
        pool_pre_ping=True,
        pool_size=100,
        max_overflow=50,
        pool_recycle=3600,
    )
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    logger.info("Database engine and session recreated successfully")


def get_db() -> Session:

    logger.debug("get_db() called - creating database session")
    try:
        db = SessionLocal()
        logger.debug("Database session created successfully")
        try:
            yield db
        finally:
            db.close()
            logger.debug("Database session closed")
    except Exception as e:
        logger.error(f"Error creating database session: {e}", exc_info=True)
        raise

