"""LLM-powered parsing for order data."""
import asyncio
import logging
import re
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import Field, ValidationError

from src.raft_agent.adapters.abstractions import AbstractLLM
from src.raft_agent.adapters.ml_model import AbstractTotalPredictor
from src.raft_agent.domain.models import FilterCriteria, Order, OrdersOutput, OrderStrict

_PARSE_CHUNK_TEMPLATE = (Path(__file__).parent / "prompts" / "parse_order_chunk.md").read_text()


logger = logging.getLogger(__name__)


_SQL_SYSTEM = (
    "You are a SQL query generator. Given a natural language query about customer orders, "
    "generate a valid SQLite SELECT query against the following table:\n\n"
    "  orders(order_id TEXT, buyer TEXT, state TEXT, total REAL)\n\n"
    "IMPORTANT: the 'state' column stores 2-letter uppercase US state abbreviations (e.g. 'OH' for Ohio, "
    "'CA' for California, 'TX' for Texas). Always use abbreviations in WHERE clauses, never full state names.\n\n"
    "IMPORTANT: every result row must include all four columns — order_id, buyer, state, total. "
    "If you need aggregations (MAX, SUM, COUNT), use a subquery or JOIN so the four columns still appear per row. "
    "Never return a result that omits any of these columns.\n\n"
    "Return ONLY the SQL query — no explanation, no markdown, no code blocks."
)


class ParseError(Exception):
    pass


OrderField = Enum("OrderField", {k: k for k in Order.model_fields})


class OrderChunk(Order):
    last_field: OrderField = Field(..., description="The name of the last field successfully parsed from this order. Used for combining chunked orders.")


def _build_parse_prompt(last_field: str, chunk: str, error_history: list[str]) -> str:
    prompt = _PARSE_CHUNK_TEMPLATE.replace("{last_field}", last_field).replace("{chunk}", chunk)
    if error_history:
        lines = "\n\n".join(f"Attempt {i + 1} error: {e}" for i, e in enumerate(error_history))
        prompt += (
            f"\n\n## Previous Attempt Errors\n\n"
            f"The following errors occurred on previous attempts. Adjust your response to avoid them:\n\n{lines}"
        )
    return prompt


async def direct_extraction(query: str, llm: AbstractLLM, max_retries: int = 3) -> Order:
    n_tokens_prompt = llm.count_tokens(_PARSE_CHUNK_TEMPLATE)
    n_tokens_schema = llm.count_tokens(str(OrderChunk.model_json_schema()))
    chunk_size = (llm.context_window - n_tokens_prompt - n_tokens_schema) // 2
    chunk_size = chunk_size - chunk_size // 5  # reserve ~20% for structured output JSON template overhead
    if chunk_size <= 0:
        raise ParseError(
            f"Context window ({llm.context_window} tokens) is too small to fit prompt "
            f"({n_tokens_prompt}) and schema ({n_tokens_schema}) overhead"
        )
    chunks = chunk_by_words(query, chunk_size=chunk_size, llm=llm)
    logger.info(f"Created {len(chunks)} chunks.")
    prompt_budget = n_tokens_prompt + chunk_size

    error_history: list[str] = []
    for attempt in range(1, max_retries + 1):
        last_field = "null"
        order = Order()
        attempt_error: Exception | None = None

        for chunk in chunks:
            chunk_tokens = llm.count_tokens(chunk)
            if chunk_tokens > chunk_size:
                raise ParseError(f"Chunk exceeds chunk_size (count: {chunk_tokens} - limit: {chunk_size}): {chunk!r}")

            prompt = _build_parse_prompt(last_field, chunk, error_history)
            n_tokens_full = llm.count_tokens(prompt)
            if n_tokens_full > prompt_budget:
                raise ParseError(
                    f"Retry prompt too large for context window on attempt {attempt} "
                    f"({n_tokens_full} tokens > {prompt_budget} prompt budget)"
                )

            try:
                partial_order = await llm.invoke_structured(
                    [
                        {"role": "system", "content": "You are a precise data extractor. Return only values explicitly present in the text."},
                        {"role": "user", "content": prompt},
                    ],
                    OrderChunk,
                )
                for k, v in partial_order.model_dump(exclude={"last_field"}).items():
                    if v is None:
                        continue
                    if isinstance(v, Enum):
                        v = v.value
                    candidates = {str(v)}
                    if isinstance(v, float):
                        candidates.add(f"{v:g}")
                    if not any(c in query for c in candidates):
                        raise ParseError(
                            f"Parsed value {v!r} for field {k} not found in original query: {query!r}"
                        )
            except Exception as e:
                logger.warning("Extraction attempt %d/%d failed on chunk: %s", attempt, max_retries, e)
                attempt_error = e
                break

            last_field = partial_order.last_field.value
            order = _merge_partial(order, partial_order)

        if attempt_error is None:
            try:
                OrderStrict.model_validate(order.model_dump())
                return order
            except ValidationError:
                attempt_error = ParseError(f"Invalid model result: {order}")

        error_history.append(str(attempt_error))
        logger.warning("Extraction attempt %d/%d failed: %s", attempt, max_retries, attempt_error)
        if attempt == max_retries:
            raise ParseError(f"Failed to extract after {max_retries} attempts") from attempt_error
    raise ParseError("Unreachable")


def _merge_partial(base: Order, partial: OrderChunk) -> Order:
    """Merge a parsed chunk into the accumulated order.

    String fields are concatenated because multi-word values (e.g. buyer "John Smith")
    may be split across word-boundary chunks. Non-string fields (USState enum, float)
    are always single tokens so they overwrite.
    """
    updates: dict = {}
    for field_name, value in partial.model_dump(exclude={"last_field"}).items():
        if value is None:
            continue
        existing = getattr(base, field_name)
        if existing is not None and isinstance(existing, str) and isinstance(value, str):
            updates[field_name] = existing + " " + value
        else:
            updates[field_name] = value
    return base.model_copy(update=updates)


async def parse_raw_orders(
    raw_orders: list[str],
    llm: AbstractLLM,
    predictor: Optional[AbstractTotalPredictor] = None,
) -> list[Order]:
    """Parse raw order text strings into Order objects using the LLM.

    Orders missing a total are imputed via the predictor when provided.
    Orders that cannot be imputed (model not yet trained) are dropped with a warning.
    """
    if not raw_orders:
        return []

    logger.info("Parsing %d orders concurrently", len(raw_orders))
    all_orders = list(
        await asyncio.gather(*(direct_extraction(order, llm) for order in raw_orders))
    )

    return _impute_totals(all_orders, predictor)


def _impute_totals(
    orders: list[Order], predictor: Optional[AbstractTotalPredictor]
) -> list[Order]:
    result: list[Order] = []
    for order in orders:
        if order.total is not None:
            result.append(order)
            continue
        if predictor is None:
            logger.warning("Order %s has no total and no predictor configured; skipping", order.orderId)
            continue
        predicted = predictor.predict(order)
        if predicted is None:
            logger.warning(
                "Cannot predict total for order %s (model not yet trained); skipping", order.orderId
            )
            continue
        logger.info("Predicted total %.2f for order %s", predicted, order.orderId)
        result.append(order.model_copy(update={"total": predicted}))
    return result


def chunk_by_words(text: str, chunk_size: int, llm: AbstractLLM) -> list[str]:
    """Split text into chunks whose token count does not exceed chunk_size.

    Words are added greedily; a chunk is flushed whenever adding the next word
    would push it over the limit.  A single word that exceeds chunk_size on its
    own is emitted as a chunk by itself.
    """
    words = text.split()
    if not words:
        return []

    word_tokens = [llm.count_tokens(words[0])] + [
        llm.count_tokens(" " + w) for w in words[1:]
    ]

    chunks: list[str] = []
    current_words: list[str] = []
    current_tokens = 0

    for word, tokens in zip(words, word_tokens):
        if current_words and current_tokens + tokens > chunk_size:
            chunks.append(" ".join(current_words))
            current_words = [word]
            current_tokens = tokens
        else:
            current_words.append(word)
            current_tokens += tokens

    if current_words:
        chunks.append(" ".join(current_words))

    return chunks


async def generate_sql_query(query: str, llm: AbstractLLM, max_retries: int = 3) -> str:
    """Use the LLM to translate a natural language query into a SQL SELECT statement."""
    for attempt in range(1, max_retries + 1):
        try:
            response = await llm.invoke([
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
