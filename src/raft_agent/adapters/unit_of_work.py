"""Unit of Work — manages SQLite transaction boundaries and repository access."""
import abc

from sqlalchemy.ext.asyncio import create_async_engine

from src.raft_agent.adapters.repository import (
    AbstractOrderRepository,
    AbstractTrainingRepository,
    SqlAlchemyOrderRepository,
    SqlAlchemyTrainingRepository,
    _ephemeral_metadata,
    _training_metadata,
)

DEFAULT_TRAINING_DB_URL = "sqlite+aiosqlite:///orders_training.db"


class AbstractUnitOfWork(abc.ABC):
    orders: AbstractOrderRepository
    training: AbstractTrainingRepository

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.rollback()
        await self.close()

    @abc.abstractmethod
    async def commit(self) -> None: ...

    @abc.abstractmethod
    async def rollback(self) -> None: ...

    @abc.abstractmethod
    async def close(self) -> None: ...


class SqlAlchemyUnitOfWork(AbstractUnitOfWork):
    def __init__(self, ephemeral_engine=None, training_engine=None) -> None:
        self._ephemeral_engine = ephemeral_engine or create_async_engine("sqlite+aiosqlite:///:memory:")
        self._training_engine = training_engine or create_async_engine(DEFAULT_TRAINING_DB_URL)

    async def __aenter__(self):
        async with self._ephemeral_engine.begin() as conn:
            await conn.run_sync(_ephemeral_metadata.create_all)
        async with self._training_engine.begin() as conn:
            await conn.run_sync(_training_metadata.create_all)
        self._conn = await self._ephemeral_engine.connect()
        self.orders = SqlAlchemyOrderRepository(self._conn)
        self._training_conn = await self._training_engine.connect()
        self.training = SqlAlchemyTrainingRepository(self._training_conn)
        return self

    async def commit(self) -> None:
        await self._conn.commit()

    async def rollback(self) -> None:
        if hasattr(self, "_conn"):
            await self._conn.rollback()

    async def close(self) -> None:
        if hasattr(self, "_conn"):
            await self._conn.close()
        if hasattr(self, "_training_conn"):
            await self._training_conn.close()
