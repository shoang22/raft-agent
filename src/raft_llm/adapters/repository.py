"""SQLAlchemy Core repository for Order persistence."""
import abc

from sqlalchemy import Column, Float, MetaData, String, Table, text

from raft_llm.domain.models import Order

metadata = MetaData()

orders_table = Table(
    "orders",
    metadata,
    Column("order_id", String, primary_key=True),
    Column("buyer", String),
    Column("state", String),
    Column("total", Float),
)


def create_tables(engine) -> None:
    metadata.create_all(engine)


class AbstractOrderRepository(abc.ABC):
    @abc.abstractmethod
    def add_all(self, orders: list[Order]) -> None: ...

    @abc.abstractmethod
    def execute_query(self, sql: str) -> list[dict]: ...


class SqlAlchemyOrderRepository(AbstractOrderRepository):
    def __init__(self, conn) -> None:
        self._conn = conn

    def add_all(self, orders: list[Order]) -> None:
        if not orders:
            return
        self._conn.execute(
            orders_table.insert(),
            [
                {"order_id": o.orderId, "buyer": o.buyer, "state": o.state.value, "total": o.total}
                for o in orders
            ],
        )

    def execute_query(self, sql: str) -> list[dict]:
        if not sql.strip().upper().startswith("SELECT"):
            raise ValueError(f"Only SELECT queries are permitted, got: {sql!r}")
        result = self._conn.execute(text(sql))
        return [dict(row._mapping) for row in result]
