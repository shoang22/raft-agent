"""LLM-powered parsing for order data."""
import logging
import math
import re

import chromadb
from langsmith import expect
from pydantic import ValidationError

from raft_llm.adapters.abstractions import AbstractLLM
from raft_llm.domain.models import FilterCriteria, Order, OrdersOutput

logger = logging.getLogger(__name__)

_PARSE_SYSTEM = (
    "You are a data extraction assistant. "
    "Given raw order text (which may vary in format), extract structured order data. "
    "Each order must have: orderId (string), buyer (string), state (2-letter US state code), "
    "total (number, no currency symbol). "
    "If a field cannot be determined, omit that order entirely."
)

_FILTER_SYSTEM = (
    "You are a query parser. Given a natural language query about orders, "
    "extract filter criteria: state (2-letter state code or null), "
    "min_total (number or null), max_total (number or null), "
    "buyer_name_contains (string or null)."
)

_SQL_SYSTEM = (
    "You are a SQL query generator. Given a natural language query about customer orders, "
    "generate a valid SQLite SELECT query against the following table:\n\n"
    "  orders(order_id TEXT, buyer TEXT, state TEXT, total REAL)\n\n"
    "IMPORTANT: the 'state' column stores 2-letter uppercase US state abbreviations (e.g. 'OH' for Ohio, "
    "'CA' for California, 'TX' for Texas). Always use abbreviations in WHERE clauses, never full state names.\n\n"
    "Return ONLY the SQL query — no explanation, no markdown, no code blocks."
)


class ParseError(Exception):
    pass


def vector_search(query: str, collection: chromadb.Collection):
    tgt = [Order.model_validate(doc) for doc in collection.get()]

    result = collection.query(query_texts=[query], n_results=1)


# each function takes a response and chunks them
def to_target(source: str, llm: AbstractLLM):
    result = llm.invoke_structured(
        [
            {"role": "system", "content": _PARSE_SYSTEM},
            {"role": "user", "content": source},
        ],
        OrdersOutput,
    ) 
    
    


def parse_raw_orders(
    raw_orders: list[str],
    llm: AbstractLLM,
    batch_size: int = 1,
) -> list[Order]:
    """Parse raw order text strings into Order objects using the LLM."""
    if not raw_orders:
        return []

    num_batches = math.ceil(len(raw_orders) / batch_size)
    all_orders: list[Order] = []

    for i in range(num_batches):
        batch = raw_orders[i * batch_size : (i + 1) * batch_size]
        logger.info("Parsing batch %d/%d (%d orders)", i + 1, num_batches, len(batch))
        all_orders.extend(_parse_batch(batch, llm))

    return all_orders


def _parse_batch(batch: list[str], llm: AbstractLLM) -> list[Order]:
    user_content = "\n".join(f"- {line}" for line in batch)
    result = llm.invoke_structured(
        [
            {"role": "system", "content": _PARSE_SYSTEM},
            {"role": "user", "content": user_content},
        ],
        OrdersOutput,
    )
    return result.orders


def extract_filter_criteria(query: str, llm: AbstractLLM) -> FilterCriteria:
    """Use the LLM to extract structured filter criteria from a natural language query."""
    return llm.invoke_structured(
        [
            {"role": "system", "content": _FILTER_SYSTEM},
            {"role": "user", "content": query},
        ],
        FilterCriteria,
    )


def generate_sql_query(query: str, llm: AbstractLLM, max_retries: int = 3) -> str:
    """Use the LLM to translate a natural language query into a SQL SELECT statement."""
    for attempt in range(1, max_retries + 1):
        try:
            response = llm.invoke([
                {"role": "system", "content": _SQL_SYSTEM},
                {"role": "user", "content": query},
            ])
            sql = response.content.strip()
            sql = re.sub(r"^```(?:sql)?\s*", "", sql)
            sql = re.sub(r"\s*```$", "", sql).strip()
            if not sql.upper().startswith("SELECT"):
                raise ParseError(f"LLM returned non-SELECT query: {sql!r}")
            return sql
        except ParseError as e:
            logger.warning("Attempt %d/%d: SQL generation failed: %s", attempt, max_retries, e)
            if attempt == max_retries:
                raise
    raise ParseError("Unreachable")
