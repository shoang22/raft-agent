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
from src.raft_agent.domain.models import PartialOrder, Order

_PARSE_CHUNK_TEMPLATE = (Path(__file__).parent / "prompts" / "parse_order_chunk.md").read_text()


logger = logging.getLogger(__name__)

# Maps every Unicode quote-lookalike to its ASCII equivalent.
_SINGLE_QUOTE_CHARS = (
    "\u0060"  # GRAVE ACCENT
    "\u00b4"  # ACUTE ACCENT
    "\u02b9"  # MODIFIER LETTER PRIME
    "\u02bb"  # MODIFIER LETTER TURNED COMMA
    "\u02bc"  # MODIFIER LETTER APOSTROPHE
    "\u02bd"  # MODIFIER LETTER REVERSED COMMA
    "\u02be"  # MODIFIER LETTER RIGHT HALF RING
    "\u02bf"  # MODIFIER LETTER LEFT HALF RING
    "\u2018"  # LEFT SINGLE QUOTATION MARK
    "\u2019"  # RIGHT SINGLE QUOTATION MARK
    "\u201a"  # SINGLE LOW-9 QUOTATION MARK
    "\u201b"  # SINGLE HIGH-REVERSED-9 QUOTATION MARK
    "\u2032"  # PRIME
    "\u2035"  # REVERSED PRIME
    "\u2039"  # SINGLE LEFT-POINTING ANGLE QUOTATION MARK
    "\u203a"  # SINGLE RIGHT-POINTING ANGLE QUOTATION MARK
    "\u055a"  # ARMENIAN APOSTROPHE
    "\u05f3"  # HEBREW PUNCTUATION GERESH
    "\uff07"  # FULLWIDTH APOSTROPHE
)
_DOUBLE_QUOTE_CHARS = (
    "\u00ab"  # LEFT-POINTING DOUBLE ANGLE QUOTATION MARK
    "\u00bb"  # RIGHT-POINTING DOUBLE ANGLE QUOTATION MARK
    "\u201c"  # LEFT DOUBLE QUOTATION MARK
    "\u201d"  # RIGHT DOUBLE QUOTATION MARK
    "\u201e"  # DOUBLE LOW-9 QUOTATION MARK
    "\u201f"  # DOUBLE HIGH-REVERSED-9 QUOTATION MARK
    "\u2033"  # DOUBLE PRIME
    "\u2036"  # REVERSED DOUBLE PRIME
    "\uff02"  # FULLWIDTH QUOTATION MARK
)
_QUOTE_TABLE = str.maketrans(
    _SINGLE_QUOTE_CHARS + _DOUBLE_QUOTE_CHARS,
    "'" * len(_SINGLE_QUOTE_CHARS) + '"' * len(_DOUBLE_QUOTE_CHARS),
)


def normalize_sql_quotes(sql: str) -> str:
    """Replace Unicode typographic quote variants with ASCII quote characters."""
    return sql.translate(_QUOTE_TABLE)



_SQL_SCHEMA_DEFAULT = "orders(order_id TEXT, buyer TEXT, state TEXT, total REAL)"

_SQL_SYSTEM_TEMPLATE = (
    "You are a SQL query generator. Given a natural language query about customer orders, "
    "generate a valid SQLite SELECT query against the following table:\n\n"
    "  {schema}\n\n"
    "IMPORTANT: the 'state' column stores 2-letter uppercase US state abbreviations (e.g. 'OH' for Ohio, "
    "'CA' for California, 'TX' for Texas). Always use abbreviations in WHERE clauses, never full state names.\n\n"
    "IMPORTANT: every result row must include all four columns — order_id, buyer, state, total. "
    "If you need aggregations (MAX, SUM, COUNT), use a subquery or JOIN so the four columns still appear per row. "
    "Never return a result that omits any of these columns.\n\n"
    "Return ONLY the SQL query — no explanation, no markdown, no code blocks."
)


class ParseError(Exception):
    pass


OrderField = Enum("OrderField", {k: k for k in PartialOrder.model_fields})


class OrderChunk(PartialOrder):
    last_field: str = Field(..., description="The name of the last field successfully parsed from this order. Used for combining chunked orders.")


def build_error_message(error_history: list[str]) -> str: 
    lines = "\n\n".join(f"Attempt {i + 1} error: {e}" for i, e in enumerate(error_history))
    return (
        f"\n\n## Previous Attempt Errors\n\n"
        f"The following errors occurred on previous attempts. Adjust your response to avoid them:\n\n{lines}"
    )


async def direct_extraction(query: str, llm: AbstractLLM, max_retries: int = 3) -> Order:
    n_tokens_prompt = llm.count_tokens(_PARSE_CHUNK_TEMPLATE)
    n_tokens_schema = llm.count_tokens(str(OrderChunk.model_json_schema()))

    error_history: list[str] = []
    for attempt in range(1, max_retries + 1):
        # Recompute chunk size each attempt: error history is appended to every chunk's prompt,
        # so the available space for chunk content shrinks as errors accumulate.
        error_message = ""
        if error_history:
            error_message = build_error_message(error_history=error_history)
            n_tokens_history = llm.count_tokens(error_message)
        else:
            n_tokens_history = 0
        chunk_size = (llm.context_window - n_tokens_prompt - n_tokens_schema - n_tokens_history) // 2
        chunk_size = chunk_size - chunk_size // 5  # reserve ~20% for structured output JSON overhead
        if chunk_size <= 0:
            raise ParseError(
                f"Context window ({llm.context_window} tokens) is too small: "
                f"prompt={n_tokens_prompt}, schema={n_tokens_schema}, error_history={n_tokens_history}"
            )

        chunks = chunk_by_words(query, chunk_size=chunk_size, llm=llm)
        logger.info("Attempt %d/%d: %d chunks (chunk_size=%d, history=%d tokens).",
                    attempt, max_retries, len(chunks), chunk_size, n_tokens_history)

        last_field = "null"
        order = PartialOrder()
        attempt_error: Exception | None = None

        for chunk in chunks:
            chunk_tokens = llm.count_tokens(chunk)
            if chunk_tokens > chunk_size:
                raise ParseError(f"Chunk exceeds chunk_size (count: {chunk_tokens} - limit: {chunk_size}): {chunk!r}")

            prompt = _PARSE_CHUNK_TEMPLATE.replace("{last_field}", last_field).replace("{chunk}", chunk)
            prompt += error_message
            try:
                resp = await llm.invoke_structured(
                    [
                        {"role": "user", "content": prompt},
                    ],
                    OrderChunk,
                )

                if resp.get('parsing_error'):
                    raise ParseError(
                        f"LLM failed to return structured output for chunk (last_field={last_field!r}): {chunk!r}\n\n"
                        "Error: {}\n\n Raw LLM output: {}".format(resp.get('parsing_error'), resp.get('raw'))
                    )
                
                partial_order = resp['parsed']
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
                            f"Parsed value {v!r} for field {k!r} not found in original query"
                        )
            except Exception as e:
                logger.warning("Extraction attempt %d/%d failed on chunk: %s", attempt, max_retries, e)
                attempt_error = e
                break

            last_field = partial_order.last_field
            order = _merge_partial(order, partial_order)

        if attempt_error is None:
            try:
                return Order.model_validate(order.model_dump())
            except ValidationError as e:
                attempt_error = ParseError(f"Extracted order failed schema validation: {e}")

        error_history.append(str(attempt_error))
        logger.warning("Extraction attempt %d/%d failed: %s", attempt, max_retries, attempt_error)
        if attempt == max_retries:
            raise ParseError(f"Failed to extract after {max_retries} attempts") from attempt_error
    raise ParseError("Unreachable")


def _merge_partial(base: PartialOrder, partial: OrderChunk) -> PartialOrder:
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


async def generate_sql_query(
    query: str,
    llm: AbstractLLM,
    max_retries: int = 3,
    schema: str = _SQL_SCHEMA_DEFAULT,
) -> str:
    """Use the LLM to translate a natural language query into a SQL SELECT statement."""
    system_prompt = _SQL_SYSTEM_TEMPLATE.format(schema=schema)
    error_history: list[tuple[str, str]] = []  # (bad_sql, error_message)
    for attempt in range(1, max_retries + 1):
        messages: list[dict] = [{"role": "system", "content": system_prompt}]
        user_content = query
        if error_history:
            corrections = "\n\n".join(
                f"Attempt {i + 1} produced invalid SQL:\n{bad_sql}\nError: {err}"
                for i, (bad_sql, err) in enumerate(error_history)
            )
            user_content = (
                f"{query}\n\n## Previous Attempt Errors\n\n"
                f"Adjust your response to avoid these:\n\n{corrections}"
            )
        messages.append({"role": "user", "content": user_content})
        bad_sql = ""
        try:
            response = await llm.invoke(messages)
            bad_sql = response.content.strip()
            logger.info(f"SQL generated: {bad_sql}")
            bad_sql = re.sub(r"^```(?:sql)?\s*", "", bad_sql)
            bad_sql = re.sub(r"\s*```$", "", bad_sql).strip()
            sql = normalize_sql_quotes(bad_sql)
            logger.info(f"SQL final: {sql}")
            if not sql.upper().startswith("SELECT"):
                raise ParseError(f"LLM returned non-SELECT query: {sql!r}")
            return sql
        except ParseError as e:
            logger.warning("Attempt %d/%d: SQL generation failed: %s", attempt, max_retries, e)
            error_history.append((bad_sql, str(e)))
            if attempt == max_retries:
                raise
    raise ParseError("Unreachable")
