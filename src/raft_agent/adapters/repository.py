"""SQLAlchemy Core repositories for Order persistence.

Two stores:
- Ephemeral (orders_table): in-memory, per agent run; used for parse→store→query.
- Training (training_orders_table): persistent file-based SQLite; accumulates data
  across runs and is the sole source used to train the linear regression model.
"""
import abc

from sqlalchemy import Column, Float, MetaData, String, Table, text
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from src.raft_agent.domain.models import Order
from src.raft_agent.service_layer.parsers import normalize_sql_quotes

# ---------------------------------------------------------------------------
# Ephemeral table (in-memory, per-run)
# ---------------------------------------------------------------------------

_ephemeral_metadata = MetaData()

orders_table = Table(
    "orders",
    _ephemeral_metadata,
    Column("order_id", String, primary_key=True),
    Column("buyer", String),
    Column("state", String),
    Column("total", Float),
)


# ---------------------------------------------------------------------------
# Training table (persistent across runs)
# ---------------------------------------------------------------------------

_training_metadata = MetaData()

training_orders_table = Table(
    "training_orders",
    _training_metadata,
    Column("order_id", String, primary_key=True),
    Column("buyer", String),
    Column("state", String),
    Column("total", Float, nullable=False),
)


# ---------------------------------------------------------------------------
# Ephemeral repository
# ---------------------------------------------------------------------------


class AbstractOrderRepository(abc.ABC):
    @abc.abstractmethod
    async def add_all(self, orders: list[Order]) -> None: ...

    @abc.abstractmethod
    async def execute_query(self, sql: str) -> list[dict]: ...


class SqlAlchemyOrderRepository(AbstractOrderRepository):
    def __init__(self, conn) -> None:
        self._conn = conn

    async def add_all(self, orders: list[Order]) -> None:
        if not orders:
            return
        await self._conn.execute(
            orders_table.insert(),
            [
                {"order_id": o.orderId, "buyer": o.buyer, "state": o.state.value, "total": o.total}
                for o in orders
            ],
        )

    async def execute_query(self, sql: str) -> list[dict]:
        sql = normalize_sql_quotes(sql)
        if not sql.strip().upper().startswith("SELECT"):
            raise ValueError(f"Only SELECT queries are permitted, got: {sql!r}")
        result = await self._conn.execute(text(sql))
        return [dict(row._mapping) for row in result]


# ---------------------------------------------------------------------------
# Training repository
# ---------------------------------------------------------------------------


class AbstractTrainingRepository(abc.ABC):
    @abc.abstractmethod
    async def upsert_all(self, orders: list[Order]) -> None:
        """Insert or replace orders by order_id. Auto-commits."""
        raise NotImplementedError

    @abc.abstractmethod
    async def get_all(self) -> list[Order]:
        """Return all stored training orders."""
        raise NotImplementedError


class SqlAlchemyTrainingRepository(AbstractTrainingRepository):
    def __init__(self, conn) -> None:
        self._conn = conn

    async def upsert_all(self, orders: list[Order]) -> None:
        if not orders:
            return
        rows = [
            {"order_id": o.orderId, "buyer": o.buyer, "state": o.state.value, "total": o.total}
            for o in orders
            if o.total is not None
        ]
        if not rows:
            return
        stmt = sqlite_insert(training_orders_table).values(rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=["order_id"],
            set_={"buyer": stmt.excluded.buyer, "state": stmt.excluded.state, "total": stmt.excluded.total},
        )
        await self._conn.execute(stmt)
        await self._conn.commit()

    async def get_all(self) -> list[Order]:
        result = await self._conn.execute(training_orders_table.select())
        rows = result.mappings().all()
        return [
            Order(orderId=r["order_id"], buyer=r["buyer"], state=r["state"], total=r["total"])
            for r in rows
        ]
