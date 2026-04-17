"""Unit of Work — manages SQLite transaction boundaries and repository access."""
import abc

from sqlalchemy import create_engine

from raft_llm.adapters.repository import (
    AbstractOrderRepository,
    SqlAlchemyOrderRepository,
    create_tables,
)


class AbstractUnitOfWork(abc.ABC):
    orders: AbstractOrderRepository

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.rollback()

    @abc.abstractmethod
    def commit(self) -> None: ...

    @abc.abstractmethod
    def rollback(self) -> None: ...

    @abc.abstractmethod
    def close(self) -> None: ...


class SqlAlchemyUnitOfWork(AbstractUnitOfWork):
    def __init__(self, engine=None) -> None:
        if engine is None:
            engine = create_engine("sqlite:///:memory:")
        create_tables(engine)
        self._conn = engine.connect()
        self.orders = SqlAlchemyOrderRepository(self._conn)

    def commit(self) -> None:
        self._conn.commit()

    def rollback(self) -> None:
        self._conn.rollback()

    def close(self) -> None:
        self._conn.close()
