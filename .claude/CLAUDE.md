# Architecture Patterns with Python — Principles for LLMs

> This document is a working reference for an LLM to reason about, generate, and review Python software
> using the architectural patterns in the book. It covers the *why*, the *what*, and the *how* — with
> enough detail to apply principles correctly in real code.

---

## Code Change Policy

**Every code change must include:**
1. **Tests updated** — add tests for new behavior, remove tests for deleted behavior, update tests whose assertions no longer match the changed behavior. No change ships without a matching test delta.
2. **Dead code removed** — delete any code rendered unreachable or unused by the change (functions, imports, constants, branches). Do not leave orphaned code behind.
3. **Documentation updated** — update docstrings, inline comments, and any affected sections of CLAUDE.md or other docs to reflect the new behavior. Stale documentation is treated as a bug.

---

## Core Philosophy

Software naturally tends toward chaos — a "Big Ball of Mud" where everything is coupled to everything
else. The patterns in this book are tools to fight entropy. The goal is not architectural purity for its
own sake, but to make a system easier to change, test, and reason about as it grows.

**The three disciplines:**
1. **TDD** — Test-Driven Development: build code that is correct and safe to refactor, with a healthy
   test pyramid (many fast unit tests, few slow E2E tests).
2. **DDD** — Domain-Driven Design: model the business domain explicitly in code, keeping that model
   clean of infrastructure concerns.
3. **Event-Driven Architecture** — decouple components by having them communicate via messages,
   enabling independent failure, scalability, and loose coupling.

**The single most important principle — Dependency Inversion (DIP):**
- High-level modules (domain logic) must not depend on low-level modules (infrastructure).
- Both should depend on abstractions.
- Concretely: the domain model knows nothing about databases, ORMs, HTTP, or message brokers.
  Infrastructure adapts to the domain, not the other way around.

---

## Part I: Building an Architecture to Support Domain Modeling

### 1. Domain Modeling

**What it is:** The domain model is the code that most directly represents the business. It lives at the
center of the architecture, has no external dependencies, and can be unit-tested in pure Python with no
infrastructure.

**Key building blocks:**

#### Entities
Objects with a persistent identity over time. Two `Batch` objects with the same `reference` are the
same batch even if other attributes change.
- Implement `__eq__` based on identity attribute(s).
- `__hash__` should be based on the same identity attribute (make it read-only or treat hash as stable).

```python
class Batch:
    def __eq__(self, other):
        if not isinstance(other, Batch):
            return False
        return other.reference == self.reference

    def __hash__(self):
        return hash(self.reference)
```

#### Value Objects
Objects defined entirely by their data, with no meaningful identity. Two `OrderLine` objects with the
same `orderid`, `sku`, and `qty` are interchangeable.
- Use `@dataclass(frozen=True)` or `NamedTuple`.
- Immutable by definition.

```python
@dataclass(frozen=True)
class OrderLine:
    orderid: str
    sku: str
    qty: int
```

#### Domain Services
Logic that doesn't naturally live on any single entity or value object. Expressed as a plain function.

```python
def allocate(line: OrderLine, batches: List[Batch]) -> str:
    batch = next(b for b in sorted(batches) if b.can_allocate(line))
    batch.allocate(line)
    return batch.reference
```

#### Domain Exceptions
Use custom exceptions named in domain language to express concepts like `OutOfStock`. They are part of
the model, not infrastructure.

**Rules:**
- Domain model classes must have **zero imports from infrastructure layers** (no SQLAlchemy, Flask,
  Redis, etc.).
- All business rules and invariants live here.
- Test domain models with plain unit tests — no database, no fakes needed.
- Prefer behavior over data: design around what the model *does*, not what it *stores*.

---

### 2. The Repository Pattern

**What it is:** An abstraction over persistent storage. The repository pretends all data lives in memory.
It decouples the domain model from the database.

**The key inversion:** The ORM imports and maps to the domain model. The domain model does *not* import
the ORM. This is "classical mapping" in SQLAlchemy.

```python
# WRONG: model depends on ORM
class Batch(Base):
    __tablename__ = 'batches'
    reference = Column(String)

# RIGHT: ORM depends on model
def start_mappers():
    mapper(model.Batch, batches_table)
```

**Abstract base:**

```python
class AbstractRepository(abc.ABC):
    @abc.abstractmethod
    def add(self, entity):
        raise NotImplementedError

    @abc.abstractmethod
    def get(self, reference):
        raise NotImplementedError
```

**Fake for testing (the real payoff):**

```python
class FakeRepository(AbstractRepository):
    def __init__(self, items):
        self._items = set(items)

    def add(self, item):
        self._items.add(item)

    def get(self, reference):
        return next(i for i in self._items if i.reference == reference)
```

**Rules:**
- Each aggregate gets one repository (see Aggregates section).
- The repository interface is minimal: `add()` and `get()`. Add other query methods only as needed.
- If the fake is hard to build, the abstraction is too complex.
- Never expose a `.save()` — mutations happen in-memory; persistence is handled by the Unit of Work.

---

### 3. Abstractions and Coupling

**Coupling:**
- *Local coupling* (cohesion within a module) is good.
- *Global coupling* (changes in one module ripple through many others) is bad and grows superlinearly.

**How abstractions reduce coupling:** Insert a simpler interface between two subsystems. The number of
dependency arrows decreases. Each side can evolve independently.

**Choosing abstractions:**
- Can you represent messy external state as a simple Python data structure (dict, list, set)?
- Can you separate "what to do" (pure logic) from "how to do it" (I/O)?
- Pattern: **Functional Core, Imperative Shell** — pure functions for logic, imperative code at the
  edges to gather inputs and apply outputs.

**Prefer fakes over mocks:**
- Fakes are working in-memory implementations of a dependency.
- Mocks assert on *how* things are called (coupling to implementation).
- Fakes assert on *end state* (coupling to behavior).
- `mock.patch` is a code smell if used extensively; it indicates missing abstractions.

---

### 4. The Service Layer

**What it is:** The service layer (also called use-case layer or orchestration layer) defines the
application's use cases. It sits between the entrypoints (Flask, CLI, tests) and the domain model.

**A typical service function:**
1. Fetch needed objects from the repository.
2. Validate/check preconditions against current state.
3. Call a domain service or method.
4. Commit (via Unit of Work).
5. Return a primitive result.

```python
def allocate(orderid: str, sku: str, qty: int, uow: AbstractUnitOfWork) -> str:
    line = OrderLine(orderid, sku, qty)
    with uow:
        product = uow.products.get(sku)
        if product is None:
            raise InvalidSku(f'Invalid sku {sku}')
        batchref = product.allocate(line)
        uow.commit()
    return batchref
```

**Rules:**
- Service layer functions should take **primitives** (strings, ints), not domain objects, as parameters.
  This keeps callers (tests, API handlers, CLI) decoupled from the domain model.
- Alternatively, take **Command objects** (see Part II) as input.
- Express dependencies as abstract types (`AbstractUnitOfWork`), not concrete implementations.
- The Flask/CLI entrypoint becomes a thin adapter: parse input → call service → return HTTP response.
- Keep "web stuff" (JSON parsing, status codes) out of the service layer entirely.

**Test pyramid with a service layer:**
- **E2E tests (few):** One or two per feature, via HTTP, testing the whole stack.
- **Service-layer tests (many):** Fast unit tests using `FakeUnitOfWork`. Cover all edge cases and
  business logic paths here.
- **Domain model tests (small core):** Fine-grained tests for complex domain logic. Replace with
  service-layer tests over time as coverage moves up.

---

### 5. The Unit of Work Pattern

**What it is:** An abstraction over atomic database operations. It manages the transaction boundary —
either everything commits or nothing does. It also provides access to repositories.

```python
class AbstractUnitOfWork(abc.ABC):
    products: AbstractRepository

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.rollback()

    @abc.abstractmethod
    def commit(self): ...

    @abc.abstractmethod
    def rollback(self): ...
```

**Usage:**
```python
with uow:
    product = uow.products.get(sku)
    product.allocate(line)
    uow.commit()
# rollback happens automatically on exception
```

**Fake for testing:**
```python
class FakeUnitOfWork(AbstractUnitOfWork):
    def __init__(self):
        self.products = FakeRepository([])
        self.committed = False

    def commit(self):
        self.committed = True

    def rollback(self):
        pass
```

**Rules:**
- The service layer takes `uow` as its only infrastructure dependency (ideally).
- `commit()` is explicit; auto-commit is a bad default.
- "Don't mock what you don't own" — fake the UoW (code you wrote), not SQLAlchemy sessions (third-party).
- The UoW tracks which aggregates were loaded (via `seen`) so it can collect domain events.

---

### 6. Aggregates and Consistency Boundaries

**What it is:** An aggregate is a cluster of domain objects treated as a single unit for data changes.
The aggregate root is the only object in the cluster that the outside world holds a reference to.

**Why:** To enforce invariants (business rules that must always be true). For example, "you cannot
allocate more stock than is available" is an invariant of `Product`. A consistency boundary ensures
this invariant cannot be violated, even with concurrent access.

**Rules:**
- One aggregate = one repository.
- Aggregates communicate with other aggregates only via domain events (see Part II), not direct
  references.
- Choose aggregate boundaries to match real-world consistency requirements, not database structure.
- Use optimistic concurrency (version numbers) to prevent lost updates when multiple processes
  modify the same aggregate.

```python
class Product:  # Aggregate root
    def __init__(self, sku, batches=None):
        self.sku = sku
        self.batches = batches or []
        self.version_number = 0
        self.events = []

    def allocate(self, line: OrderLine) -> str:
        # enforces invariants across all batches for this SKU
        ...
        self.version_number += 1
        self.events.append(events.Allocated(...))
        return batchref
```

---

## Part II: Event-Driven Architecture

### 7. Domain Events and the Message Bus

**Domain events** represent things that happened in the past. They are simple immutable dataclasses
named in the past tense.

```python
@dataclass
class OutOfStock(Event):
    sku: str

@dataclass
class Allocated(Event):
    orderid: str
    sku: str
    qty: int
    batchref: str
```

**Why events:** When a use case triggers secondary effects (send email, update another aggregate,
publish to an external system), putting all that logic in one function violates the Single
Responsibility Principle. Events let secondary effects happen asynchronously or optionally, without
entangling them with the primary use case.

**The message bus** routes events to handlers:

```python
# A mapping from event type -> list of handler functions
EVENT_HANDLERS = {
    events.OutOfStock: [handlers.send_out_of_stock_notification],
    events.Allocated:  [handlers.publish_allocated_event],
}
```

**How events are raised and collected (preferred pattern):**
1. Domain model methods append events to `self.events`.
2. After `uow.commit()`, the UoW collects events from all seen aggregates (`uow.collect_new_events()`).
3. The message bus processes the queue, running handlers and collecting any new events they trigger.

**Rules:**
- Events are facts; they describe what happened. They cannot be "rejected."
- Event handlers should be idempotent where possible (safe to retry on failure).
- Use `tenacity` or similar for retry logic on transient failures.
- A chain of events = a chain of separate transactions. This enables eventual consistency.

---

### 8. Commands vs. Events

**Commands** represent a request for the system to do something. Named with imperative verbs.
Commands have one handler. If a command fails, the error is raised to the caller.

**Events** represent facts about the past. Named with past-tense verbs. Events can have multiple
handlers. If an event handler fails, it is logged but processing continues.

| | Command | Event |
|---|---|---|
| **Named** | Imperative (e.g., `Allocate`) | Past tense (e.g., `Allocated`) |
| **Handlers** | Exactly one | Zero or more |
| **On failure** | Raise exception to caller | Log and continue |
| **Sent to** | One specific handler | All interested handlers |

```python
# Dispatch differently
def handle(message):
    if isinstance(message, commands.Command):
        handle_command(message)   # re-raises exceptions
    elif isinstance(message, events.Event):
        handle_event(message)     # swallows/logs exceptions

def handle_command(command):
    handler = COMMAND_HANDLERS[type(command)]
    handler(command)  # raises on error

def handle_event(event):
    for handler in EVENT_HANDLERS[type(event)]:
        try:
            handler(event)
        except Exception:
            logger.exception(...)
            continue  # fail independently
```

**Rule:** A command modifies one aggregate and must succeed or fail atomically. Any downstream cleanup,
notifications, or cross-aggregate updates happen via events, which can fail independently.

---

### 9. Event-Driven Microservice Integration

**The anti-pattern:** Distributed Ball of Mud — services calling each other synchronously via HTTP,
creating temporal coupling (everything must work at the same time) and cascading failures.

**The pattern:** Temporal decoupling via async messaging.
- Services publish events to an external message bus (Redis pub/sub, Kafka, RabbitMQ).
- Services subscribe to events and handle them independently.
- Failure in one service does not cascade synchronously to callers.

**Design principle:** Think in *verbs* (processes), not *nouns* (data objects).
- Not "Order Service" and "Batch Service" — but "ordering" and "allocating."
- Boundaries follow business processes (bounded contexts), not database tables.

**Connascence levels (weakest to strongest):**
- Connascence of Name (shared event name/fields) — acceptable across services.
- Connascence of Execution (shared call order) — avoid across services.
- Connascence of Timing (must happen simultaneously) — avoid across services.

**Thin adapters for external messaging:**

```python
# Incoming: translate external message -> internal command -> message bus
def handle_change_batch_quantity(redis_message):
    data = json.loads(redis_message['data'])
    cmd = commands.ChangeBatchQuantity(ref=data['batchref'], qty=data['qty'])
    bus.handle(cmd)

# Outgoing: translate domain event -> external message
def publish_allocated_event(event: events.Allocated, _):
    redis_client.publish('line_allocated', json.dumps(asdict(event)))
```

---

### 10. Command-Query Responsibility Segregation (CQRS)

**Core insight:** Reads (queries) and writes (commands) have different needs. The domain model is
optimized for writes (enforcing rules, complex state transitions). It is *not* optimized for reads.

| | Read side | Write side |
|---|---|---|
| Behavior | Simple read | Complex business logic |
| Consistency | Can be stale | Must be transactionally consistent |
| Cacheability | Highly cacheable | Uncacheable |

**Principle (CQS):** Functions should either modify state or answer questions, never both.

**Read model options (simplest to most complex):**
1. **Repository + domain model** — simple, but may have N+1 queries and awkward object traversal.
2. **ORM queries directly** — reuses DB config but can be slow and complex.
3. **Raw SQL** — fast, explicit, fine-grained control. Recommended when reads are conceptually
   different from the write model.
4. **Separate denormalized read store** — a separate table (or Redis) updated via event handlers.
   Best for high read-volume, complex queries, or needing horizontal scalability.

```python
# Read-only view using raw SQL — this is fine and often best
def allocations(orderid: str, uow: SqlAlchemyUnitOfWork):
    with uow:
        results = list(uow.session.execute(
            'SELECT sku, batchref FROM allocations_view WHERE orderid = :orderid',
            dict(orderid=orderid)
        ))
    return [{'sku': sku, 'batchref': batchref} for sku, batchref in results]
```

**Updating a read model via event handler:**
```python
EVENT_HANDLERS = {
    events.Allocated: [
        handlers.publish_allocated_event,
        handlers.add_allocation_to_read_model,  # keeps read store in sync
    ],
}
```

**Rules:**
- Split your codebase into clearly separated read-only views and state-modifying command/event handlers.
- Apply CQRS when: read patterns differ significantly from write patterns; high read volume requires
  scalability; the domain model is rich and complex.
- Don't apply CQRS for simple CRUD apps — the overhead is not justified.

---

### 11. Dependency Injection and Bootstrapping

**The problem:** As the number of adapters grows (database, email, Redis, S3, etc.), every entrypoint
(Flask, CLI, tests) needs to assemble and inject the right dependencies. This creates duplication and
coupling between entrypoints and infrastructure choices.

**The solution:** A bootstrap script — a *composition root* — that assembles the application once,
wires up dependencies, and returns a configured message bus.

```python
def bootstrap(
    start_orm: bool = True,
    uow: AbstractUnitOfWork = SqlAlchemyUnitOfWork(),
    notifications: AbstractNotifications = EmailNotifications(),
    publish: Callable = redis_eventpublisher.publish,
) -> MessageBus:
    if start_orm:
        orm.start_mappers()

    dependencies = {'uow': uow, 'notifications': notifications, 'publish': publish}
    injected_event_handlers = {
        event_type: [inject_dependencies(h, dependencies) for h in handlers]
        for event_type, handlers in EVENT_HANDLERS.items()
    }
    injected_command_handlers = {
        cmd_type: inject_dependencies(handler, dependencies)
        for cmd_type, handler in COMMAND_HANDLERS.items()
    }
    return MessageBus(uow=uow,
                      event_handlers=injected_event_handlers,
                      command_handlers=injected_command_handlers)
```

**In tests — override defaults:**
```python
bus = bootstrap.bootstrap(
    start_orm=False,
    uow=FakeUnitOfWork(),
    notifications=FakeNotifications(),
    publish=lambda *args: None,
)
```

**DI approaches for handlers:**
- **Closures/partials** — functional, concise, slightly tricky with mutable state.
- **Classes with `__call__`** — explicit, familiar to OO programmers.
- **Inspect-based injection** — match handler argument names to dependency dict (the book's approach).
- **Manual lambda wiring** — verbose but maximally transparent.

**Adapters — the canonical pattern:**
1. Define an ABC for the interface (`AbstractNotifications`).
2. Implement the real version (`EmailNotifications`).
3. Build a fake for unit tests (`FakeNotifications`).
4. Test the real thing via integration tests against a real (or realistic) external service.

---

## Folder Structure

```
src/
└── myapp/
    ├── domain/
    │   ├── model.py      # Entities, Value Objects, Aggregates
    │   ├── events.py     # Domain events (dataclasses)
    │   └── commands.py   # Commands (dataclasses)
    ├── service_layer/
    │   ├── handlers.py   # Command and event handlers
    │   ├── messagebus.py # MessageBus class
    │   └── unit_of_work.py
    ├── adapters/
    │   ├── orm.py        # SQLAlchemy classical mappings
    │   ├── repository.py # AbstractRepository + SqlAlchemy impl
    │   └── notifications.py
    ├── entrypoints/
    │   ├── flask_app.py  # HTTP adapter (thin)
    │   └── redis_consumer.py
    └── bootstrap.py      # Composition root / DI

tests/
├── unit/         # Fast tests; use fakes; no DB
├── integration/  # Test DB interaction, real adapters
└── e2e/          # Full stack; HTTP + real DB
```

---

## Cheat Sheet: Which Pattern for What Problem?

| Problem | Pattern |
|---|---|
| Domain logic mixed with infrastructure | Domain Model + DIP |
| Domain model coupled to ORM/DB | Repository Pattern |
| Hard to unit test because of DB dependency | Repository + FakeRepository |
| Service layer tightly coupled to session/DB | Unit of Work Pattern |
| Risk of inconsistent state across aggregate | Aggregate + consistency boundary |
| Secondary effects tangled with primary logic | Domain Events + Message Bus |
| Commands vs. notifications unclear | Commands + Events (separate types) |
| Services tightly coupled via synchronous HTTP | Async messaging / event-driven integration |
| Read queries awkward via domain model | CQRS / separate read model |
| Infrastructure wiring duplicated everywhere | Bootstrap / Dependency Injection |

---

## Anti-Patterns to Avoid

- **Big Ball of Mud** — no layering, everything depends on everything.
- **Anemic Domain Model** — all logic in service layer, domain objects are just data bags.
- **Fat Controllers** — business logic in Flask views or API handlers.
- **Model depends on ORM** — domain classes inherit from `Base` or `models.Model`.
- **Distributed Ball of Mud** — microservices calling each other synchronously via nouns.
- **Overuse of mocks** — `mock.patch` everywhere hides missing abstractions and couples tests to
  implementation details.
- **Primitive obsession (unmitigated)** — passing raw strings/ints when a Value Object or Command
  would give type safety and intent.
- **God services** — service layer functions that do too much; violates SRP.
- **SELECT N+1** — using the domain model for read operations that require many joins.

---

## Testing Rules of Thumb

1. **Aim for one E2E test per feature** — tests the full stack (HTTP → DB). Proves it all connects.
2. **Bulk of tests at the service layer** — fast, in-memory, cover all edge cases with fakes.
3. **Small core of domain model tests** — for complex domain logic; replace with service-layer tests
   as the codebase matures.
4. **Error handling is a feature** — test it like any other use case.
5. **Express tests in domain language** — tests should read like executable specifications.
6. **Avoid testing implementation details** — test behavior and end state, not how the sausage is made.
7. **High gear vs. low gear** — use service-layer tests (high gear) for most work; drop to domain model
   tests (low gear) when designing new or gnarly domain logic.

---

## Validation (Appendix E Summary)

- **Syntax validation** (is the input well-formed?) — at the edge/entrypoint, before it enters the
  system. Use dataclass field types, Pydantic, or marshmallow schemas. Apply Postel's Law: be liberal
  in what you accept, strict in what you emit.
- **Semantic validation** (does the input make sense for the domain?) — in command handlers or domain
  services (e.g., "Is this SKU known?").
- **Pragmatic validation** (is the input valid in this context?) — deep in the domain, enforced by
  aggregate invariants (e.g., "Is there enough stock?").

---

## Key Quotes to Internalize

> "A big ball of mud is the natural state of software in the same way that wilderness is the natural
> state of your garden. It takes energy and direction to prevent the collapse."

> "Behavior should come first and drive our storage requirements. Our customers don't care about the
> data model — they care about what the system does."

> "Every line of code that we put in a test is like a blob of glue, holding the system in a particular
> shape. The more low-level tests we have, the harder it will be to change things."

> "Designing for testability really means designing for extensibility."

> "If your app is just a simple CRUD wrapper around a database, you don't need a domain model or
> a repository."