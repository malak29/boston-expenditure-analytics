from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.pool import NullPool
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from app.core.config import settings

engine = create_async_engine(
    settings.database_url,
    poolclass=NullPool if settings.debug else None,
    pool_pre_ping=True,
    pool_recycle=300,
    echo=settings.debug
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False
)

async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

@asynccontextmanager
async def get_db_transaction():
    async with AsyncSessionLocal() as session:
        async with session.begin():
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

class DatabaseManager:
    def __init__(self):
        self.engine = engine
        self.session_factory = AsyncSessionLocal
    
    async def create_tables(self):
        from app.models.database import Base
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    
    async def drop_tables(self):
        from app.models.database import Base
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
    
    async def health_check(self) -> bool:
        try:
            async with self.session_factory() as session:
                await session.execute("SELECT 1")
                return True
        except Exception:
            return False
    
    async def close(self):
        await self.engine.dispose()

db_manager = DatabaseManager()